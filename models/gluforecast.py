import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Optional

class Preprocessor:
    def __init__(self, lookback_window: int = 60, prediction_length: int = 12):
        self.lookback_window = lookback_window
        self.prediction_length = prediction_length
        self.UPPER = 402
        self.LOWER = 38
        time_steps = torch.linspace(0, self.lookback_window*5, steps=self.lookback_window)
        insulin_action = 4.26e-5 * torch.pow(time_steps, 1.5) * torch.exp(-1.5 * time_steps / 75.0)
        self.stacked_insulin_action = torch.stack([insulin_action] * self.lookback_window, dim=0)
        for i in range(self.lookback_window):
            self.stacked_insulin_action[i] = torch.nn.functional.pad(self.stacked_insulin_action[i], (i, 0), mode='constant', value=0)[:self.lookback_window]

        self.stacked_insulin_action = self.stacked_insulin_action.unsqueeze(0)

    def normalize(self, glucose_values):
        return (glucose_values - self.LOWER) / ((self.UPPER - self.LOWER)/2) - 1.0

    def unnormalize(self, glucose_normalized):
        return (glucose_normalized + 1.0) * ((self.UPPER - self.LOWER)/2) + self.LOWER

    def preprocess(self, glucose_values: torch.Tensor, insulin_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(glucose_values, torch.Tensor):
            glucose_values = torch.tensor(glucose_values)
        if not isinstance(insulin_values, torch.Tensor):
            insulin_values = torch.tensor(insulin_values)

        if glucose_values.ndim == 1:
            glucose_values = glucose_values.unsqueeze(0)
        if insulin_values.ndim == 1:
            insulin_values = insulin_values.unsqueeze(0)

        glucose_values = self.normalize(glucose_values)
        insulin_values = insulin_values[:, -self.lookback_window:]
        insulin_values = insulin_values.unsqueeze(-1)
        insulin_action = torch.sum(insulin_values * self.stacked_insulin_action, dim=0)

        return glucose_values, insulin_action


# -----------------------------
# Minimal KV-cache support requires a custom attention stack.
# nn.TransformerEncoder does not expose KV caching.
# The following tiny block stack keeps GlucoseTransformer mostly intact.
# -----------------------------

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
        return x.view(B, T, self.n_heads, self.hd).transpose(1, 2)  # (B,H,T,hd)

    def _merge(self, x):
        B, H, T, hd = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * hd)

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        # x: (B,T,D)
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.d_model, dim=-1)
        q = self._split(q)  # (B, H, T_q, hd)
        k = self._split(k)  # (B, H, T_k, hd)
        v = self._split(v)  # (B, H, T_k, hd)

        # Prepare attention mask: scaled_dot_product_attention expects (L, S) or broadcastable
        # q: (B, H, T_q, hd), k: (B, H, T_k, hd)
        # attn_mask should be (T_q, T_k) or (B*H, T_q, T_k) or broadcastable
        B = q.shape[0]
        T_q = q.shape[2]
        T_k = k.shape[2]

        causal_mask = torch.ones(T_q, T_k, dtype=torch.bool).tril(diagonal=0).bool().repeat(B, 1, 1).reshape(B, 1, T_q, T_k).to(q.device)
        if attn_mask is not None:
            # attn_mask is (B, T_q, T_k) from forward, expand for heads: (B, 1, T_q, T_k)
            attn_mask = attn_mask.unsqueeze(1)  # (B, 1, T_q, T_k) - broadcasts to (B, H, T_q, T_k)
            causal_mask = attn_mask & causal_mask
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=causal_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = self._merge(y)
        y = self.proj(y)

        return y


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
        a = self.attn(self.ln1(x), attn_mask=attn_mask)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x

class GluforecastModel(nn.Module):
    def __init__(
        self,
        d_in=7,
        d_model=128,
        n_heads=4,
        n_layers=4,
        max_len=180,
    ):
        super().__init__()

        self.input_proj = nn.Linear(d_in, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))

        # REPLACED: nn.TransformerEncoder(...) with a cacheable block stack
        self.blocks = nn.ModuleList([_Block(d_model, n_heads, dropout=0.1) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)

        self.head = nn.Linear(d_model, 12)

    def forward(self, x, mask: Optional[torch.Tensor] = None, kv_cache: Optional[list] = None, use_cache: bool = False):
        """
        x: (B,T,7)
        mask: (B,T) input mask where True/1 = valid token, False/0 = padding token
        kv_cache: list[dict] of length n_layers, each dict has {"k","v"} or None values.
        Returns:
          y: (B,T,12)
          new_cache: updated kv_cache if use_cache else None
        """
        B, T, _ = x.shape

        h = self.input_proj(x)
        assert h.isfinite().all(), f'h is not finite: {h}'

        h = h + self.pos_emb[:, :T]

        # Apply input mask: zero out embeddings for padding positions
        if mask is not None:
            # mask: (B, T) -> (B, T, 1) for broadcasting
            h = h * mask.reshape(B, T, 1)
            # No cache: simple mask for current tokens
            mask = mask.unsqueeze(1) & mask.unsqueeze(2)  # (B, T, T)
            mask = mask.reshape(B, T, T)

        for i, blk in enumerate(self.blocks):
            h = blk(h, attn_mask=mask)

        h = self.ln_f(h)
        y = self.head(h)
        return y  # (B, T, 12), cache


class Gluforecast:
    def __init__(self, model_path, device='cpu'):
        self.model = GluforecastModel()
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        self.preprocessor = Preprocessor()
        self.model.to(device)
        self.device = device

    def _time_features(self, ts):
        # ts: (B, T) seconds since epoch
        day = 24 * 60 * 60
        week = 7 * day

        tod = 2 * torch.pi * (ts % day) / day
        tow = 2 * torch.pi * (ts % week) / week

        return torch.stack([
            torch.sin(tod), torch.cos(tod),
            torch.sin(tow), torch.cos(tow),
        ], dim=-1)  # (B, T, 4)

    def predict(self, timestamps, cgm, insulin, carbs):
        tf = self._time_features(torch.tensor(timestamps.astype(np.int64)) // np.array([1e9]).astype(np.int64))  # (B, T, 4)
        cgm = torch.tensor(self.preprocessor.normalize(cgm))
        insulin = torch.tensor(insulin)
        carbs = torch.tensor(carbs)

        x = torch.cat([
            cgm.unsqueeze(-1),
            insulin.unsqueeze(-1),
            carbs.unsqueeze(-1),
            tf,
        ], dim=-1)  # (B, T, 7)
        x = x.to(torch.float32)
        x = x.to(self.device)
        T = 180-13 # 15 hours in 5-minute increments
        if x.shape[1] > T:
            x = x[:, -T:, :]
        with torch.no_grad():
            delta_hat = self.model(x, mask=None)
            preds = (x[:, -1, 0].unsqueeze(-1) + delta_hat[:, -1, :])  # (B, 12)

        return self.preprocessor.unnormalize(preds).cpu().int().numpy()