import numpy as np
import torch
import torch.nn as nn


FEATURE_COLUMNS = {
    'cgm': ['CGM'],
    'cgm_insulin': ['CGM', 'Insulin'],
    'cgm_carbs': ['CGM', 'Carbs'],
    'cgm_insulin_carbs': ['CGM', 'Insulin', 'Carbs'],
}


class LSTMModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128, num_layers=2,
                 pred_len=12, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, pred_len)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class LSTM:
    """Benchmark-compatible wrapper for local LSTM checkpoints."""

    def __init__(self, checkpoint_path, feature_set='cgm_insulin_carbs',
                 device='cpu'):
        self.feature_set = feature_set
        self.feature_cols = FEATURE_COLUMNS[feature_set]
        self.device = device

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        lstm_hp = ckpt.get('lstm_hparams') or {}
        dropout = lstm_hp.get('dropout', ckpt.get('dropout', 0.1))
        self.model = LSTMModel(
            input_dim=len(self.feature_cols),
            hidden_dim=ckpt.get('hidden_dim', lstm_hp.get('hidden_dim', 128)),
            num_layers=ckpt.get('num_layers', lstm_hp.get('num_layers', 2)),
            pred_len=ckpt.get('pred_len', 12),
            dropout=dropout,
        )
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.to(device)
        self.model.eval()

    def predict(self, timestamps, cgm, insulin, carbs):
        channels = {'CGM': cgm, 'Insulin': insulin, 'Carbs': carbs}
        x = np.stack([channels[c] for c in self.feature_cols], axis=-1)
        x_t = torch.tensor(x, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            pred = self.model(x_t)
        return pred.cpu().numpy()
