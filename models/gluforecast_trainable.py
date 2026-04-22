import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


FEATURE_COLUMNS = {
    'cgm': ['CGM'],
    'cgm_insulin': ['CGM', 'Insulin'],
    'cgm_carbs': ['CGM', 'Carbs'],
    'cgm_insulin_carbs': ['CGM', 'Insulin', 'Carbs'],
}


def _time_features(ts_seconds: torch.Tensor) -> torch.Tensor:
    """ts_seconds: (B, T), returns (B, T, 4)."""
    day = 24 * 60 * 60
    week = 7 * day
    tod = 2 * torch.pi * (ts_seconds % day) / day
    tow = 2 * torch.pi * (ts_seconds % week) / week
    return torch.stack([
        torch.sin(tod), torch.cos(tod),
        torch.sin(tow), torch.cos(tow),
    ], dim=-1)


class _CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.hd = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        bsz, t, _ = x.shape
        return x.view(bsz, t, self.n_heads, self.hd).transpose(1, 2)

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        bsz, h, t, hd = x.shape
        return x.transpose(1, 2).contiguous().view(bsz, t, h * hd)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.d_model, dim=-1)
        q = self._split(q)
        k = self._split(k)
        v = self._split(v)

        bsz = q.shape[0]
        tq = q.shape[2]
        tk = k.shape[2]
        causal_mask = torch.ones(tq, tk, dtype=torch.bool, device=q.device).tril()
        causal_mask = causal_mask.repeat(bsz, 1, 1).reshape(bsz, 1, tq, tk)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)
            causal_mask = attn_mask & causal_mask
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=causal_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.proj(self._merge(y))


class _Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = _CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class TrainableGluForecastModel(nn.Module):
    """Bench-local GluForecast model adapted to ablation feature sets."""

    def __init__(
        self,
        feature_set: str = 'cgm',
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        max_len: int = 180,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_set = feature_set
        self.feature_cols = FEATURE_COLUMNS[feature_set]
        self.input_proj = nn.Linear(7, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.blocks = nn.ModuleList([
            _Block(d_model, n_heads, dropout=dropout) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 12)

    def _split_channels(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = {name: i for i, name in enumerate(self.feature_cols)}
        cgm = x[:, :, idx['CGM']]
        insulin = x[:, :, idx['Insulin']] if 'Insulin' in idx else torch.zeros_like(cgm)
        carbs = x[:, :, idx['Carbs']] if 'Carbs' in idx else torch.zeros_like(cgm)
        return cgm, insulin, carbs

    def _build_model_input(
        self,
        x: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cgm, insulin, carbs = self._split_channels(x)
        bsz, t = cgm.shape
        if timestamps is None:
            tf = torch.zeros((bsz, t, 4), dtype=x.dtype, device=x.device)
        else:
            ts = timestamps.to(dtype=torch.float32, device=x.device)
            # Evaluate path provides ns epoch; convert when values are large.
            if ts.abs().max().item() > 1e12:
                ts = ts / 1e9
            tf = _time_features(ts)
        return torch.cat(
            [cgm.unsqueeze(-1), insulin.unsqueeze(-1), carbs.unsqueeze(-1), tf],
            dim=-1,
        )

    def forward(
        self,
        x: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x7 = self._build_model_input(x, timestamps=timestamps)
        bsz, t, _ = x7.shape
        h = self.input_proj(x7) + self.pos_emb[:, :t]
        attn_mask = None
        if mask is not None:
            h = h * mask.reshape(bsz, t, 1)
            attn_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
        for blk in self.blocks:
            h = blk(h, attn_mask=attn_mask)
        h = self.ln_f(h)
        delta = self.head(h)  # (B, T, 12)
        # Predict absolute CGM from last step baseline + delta.
        return x7[:, -1, 0].unsqueeze(-1) + delta[:, -1, :]


class TrainableGluForecast:
    """Benchmark-compatible inference wrapper for trained GluForecast checkpoints."""

    def __init__(self, checkpoint_path: str, feature_set: str = 'cgm', device: str = 'cpu'):
        self.feature_set = feature_set
        self.feature_cols = FEATURE_COLUMNS[feature_set]
        self.device = device
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        hp = ckpt.get('gluforecast_hparams') or {}
        self.model = TrainableGluForecastModel(
            feature_set=feature_set,
            d_model=hp.get('d_model', 128),
            n_heads=hp.get('n_heads', 4),
            n_layers=hp.get('n_layers', 4),
            max_len=hp.get('max_len', ckpt.get('seq_len', 180)),
            dropout=hp.get('dropout', 0.1),
        )
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.to(device)
        self.model.eval()

    def predict(self, timestamps, cgm, insulin, carbs):
        channels = {'CGM': cgm, 'Insulin': insulin, 'Carbs': carbs}
        selected = [channels[c] for c in self.feature_cols]
        x = np.stack(selected, axis=-1)
        x_t = torch.tensor(x, dtype=torch.float32, device=self.device)
        ts_t = torch.tensor(timestamps, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            pred = self.model(x_t, timestamps=ts_t)
        return pred.detach().cpu().numpy()
