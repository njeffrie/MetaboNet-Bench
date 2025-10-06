import torch
from torch import nn

from transformers import AutoModel, AutoConfig


class SimpleLSTM(nn.Module):
    """
    A simple LSTM-based regressor that consumes the previous len_seq glucose values
    (at 5-minute intervals) and predicts the next len_pred (12) values (1 hour).
    Input shape: (batch_size, len_seq, 1)
    Output shape: (batch_size, len_pred)
    """
    def __init__(self, input_size: int = 1, hidden_size: int = 128, num_layers: int = 2,
                 dropout: float = 0.1, len_seq: int = 180, len_pred: int = 12):
        super().__init__()
        self.len_seq = len_seq
        self.len_pred = len_pred
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.projection = nn.Linear(hidden_size, len_pred)

    def forward(self, batch_x: torch.Tensor) -> torch.Tensor:
        # batch_x: (batch, len_seq, 1)
        lstm_out, hidden = self.lstm(batch_x)
        # Use the last timestep's hidden representation
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_size)
        pred = self.projection(last_hidden)  # (batch, len_pred)
        return pred

class LSTM:
    def __init__(self, huggingface_model_name: str = 'njeffrie/LSTMGlucosePrediction'):
        self.model = AutoModel.from_pretrained(huggingface_model_name, trust_remote_code=True)
        self.config = AutoConfig.from_pretrained(huggingface_model_name, trust_remote_code=True)
        self.model.eval()

    def predict(self, subject_id, timestamps, input_glucose):
        if len(input_glucose) < self.config.len_seq:
            print(f'input_glucose length {len(input_glucose)} is less than config.len_seq {self.config.len_seq}')
        assert(len(input_glucose) >= self.config.len_seq)
        glucose = input_glucose[-self.config.len_seq:].numpy()
        with torch.no_grad():
            pred= self.model(glucose)
        return pred.numpy()
        