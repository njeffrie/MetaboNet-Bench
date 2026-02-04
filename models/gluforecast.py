import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

# Glucose normalization bounds (mg/dL)
CGM_LOWER, CGM_UPPER = 38, 402
CGM_HALF_RANGE = (CGM_UPPER - CGM_LOWER) / 2


class Preprocessor:
    """Glucose normalization for model input/output."""

    def normalize(self, glucose_values):
        return (glucose_values - CGM_LOWER) / CGM_HALF_RANGE - 1.0

    def unnormalize(self, glucose_normalized):
        return (glucose_normalized + 1.0) * CGM_HALF_RANGE + CGM_LOWER


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

    def _split(self, x):
        B, T, D = x.shape
        return x.view(B, T, self.n_heads, self.hd).transpose(1, 2)

    def _merge(self, x):
        B, H, T, hd = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * hd)

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        B, T_q, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.d_model, dim=-1)
        q, k, v = self._split(q), self._split(k), self._split(v)

        causal_mask = torch.tril(
            torch.ones(T_q, T_q, dtype=torch.bool, device=x.device)
        ).unsqueeze(0).unsqueeze(0)  # (1, 1, T_q, T_q)
        if attn_mask is not None:
            causal_mask = attn_mask.unsqueeze(1) & causal_mask

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

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        x = x + self.attn(self.ln1(x), attn_mask=attn_mask)
        return x + self.mlp(self.ln2(x))


class GluforecastModel(nn.Module):
    def __init__(
        self,
        d_in: int = 7,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        max_len: int = 180,
    ):
        super().__init__()
        self.input_proj = nn.Linear(d_in, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.blocks = nn.ModuleList([
            _Block(d_model, n_heads, dropout=0.1) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 12)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: (B, T, 7). mask: (B, T) True = valid. Returns (B, T, 12)."""
        B, T, _ = x.shape
        h = self.input_proj(x) + self.pos_emb[:, :T]
        if mask is not None:
            h = h * mask.reshape(B, T, 1)
            attn_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
        else:
            attn_mask = None

        for blk in self.blocks:
            h = blk(h, attn_mask=attn_mask)
        return self.head(self.ln_f(h))


class Gluforecast:
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.device = device
        self.model = GluforecastModel()
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device).eval()
        self.preprocessor = Preprocessor()

    def _time_features(self, ts: torch.Tensor) -> torch.Tensor:
        """ts (B, T) in seconds. Returns (B, T, 4): sin/cos of time-of-day and time-of-week."""
        day = 24 * 3600
        week = 7 * day
        tod = 2 * torch.pi * (ts % day) / day
        tow = 2 * torch.pi * (ts % week) / week
        return torch.stack([torch.sin(tod), torch.cos(tod), torch.sin(tow), torch.cos(tow)], dim=-1)

    def predict(self, timestamps: np.ndarray, cgm: np.ndarray, insulin: np.ndarray, carbs: np.ndarray) -> np.ndarray:
        """Timestamps in nanoseconds. Returns (B, 12) int predictions in mg/dL."""
        ts_sec = torch.tensor(timestamps.astype(np.int64) // 10**9)
        cgm_norm = torch.tensor(self.preprocessor.normalize(cgm), dtype=torch.float32)
        insulin_t = torch.tensor(insulin, dtype=torch.float32)
        carbs_t = torch.tensor(carbs, dtype=torch.float32)

        tf = self._time_features(ts_sec)
        x = torch.cat([
            cgm_norm.unsqueeze(-1),
            insulin_t.unsqueeze(-1),
            carbs_t.unsqueeze(-1),
            tf,
        ], dim=-1).to(self.device)

        max_len = 167  # model input length (truncate if longer)
        if x.shape[1] > max_len:
            x = x[:, -max_len:, :]

        with torch.no_grad():
            delta_hat = self.model(x, mask=None)
            preds = x[:, -1, 0].unsqueeze(-1) + delta_hat[:, -1, :]

        return self.preprocessor.unnormalize(preds).cpu().int().numpy()
