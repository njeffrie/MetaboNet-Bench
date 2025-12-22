import torch
import torch.nn as nn
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

class GlucoseTransformer(nn.Module):
    def __init__(
        self,
        d_in=7,
        d_model=128,
        n_heads=4,
        n_layers=4,
        max_len=512,
    ):
        super().__init__()

        self.input_proj = nn.Linear(d_in, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        B, T, _ = x.shape

        h = self.input_proj(x)
        assert h.isfinite().all(), f'h is not finite: {h}'
        h = h + self.pos_emb[:, :T]

        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device), diagonal=1
        ).bool()
        h = self.transformer(h, mask=causal_mask)
        y = self.head(h).squeeze(-1)
        return y  # (B, T)
    
class GlucoseDecoderModel:
    def __init__(self, model_path):
        self.model = GlucoseTransformer()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.preprocessor = Preprocessor()

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

    def predict(self, ts, cgm, insulin, carbs, ph = 60):
        cgm = torch.tensor(list(self.preprocessor.normalize(cgm)), dtype=torch.float32).unsqueeze(0)
        insulin = torch.tensor(list(insulin), dtype=torch.float32).unsqueeze(0)
        carbs = torch.tensor(list(carbs), dtype=torch.float32).unsqueeze(0)
        ts = ts.astype(np.int64) // 1e9
        new_ts = np.arange(ts[-1], ts[-1] + 60 * ph, 5 * 60)
        ts = np.concatenate([ts, new_ts])
        ts = torch.tensor(ts, dtype=torch.float32).unsqueeze(0)

        prediction_steps = ph//5
        cgm = torch.cat([cgm, torch.zeros([1, prediction_steps], dtype=torch.float32)], dim=1)
        #print(f'ts shape: {ts.shape}, insulin shape: {insulin.shape}, carbs shape: {carbs.shape}, cgm shape: {cgm.shape}')
        tf = self._time_features(ts)  # (B, T, 4)

        x = torch.cat([
            cgm.unsqueeze(-1),
            insulin.unsqueeze(-1),
            carbs.unsqueeze(-1),
            tf,
        ], dim=-1)  # (B, T, 7)
        x = x.to(torch.float32)
        T = 144 # 12 hours in 5-minute increments
        if x.shape[1] > T:
            x = x[:, -T-ph:, :]
        with torch.no_grad():
            for h in range(ph):
                pred = self.model(x[:,:-ph+h])
                x[:,-ph+h+1,0] = pred[:,-1]
        return self.preprocessor.unnormalize(x[:,-ph:,0].squeeze().detach().numpy())