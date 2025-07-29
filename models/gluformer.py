from transformers import AutoModel, AutoConfig
from datetime import datetime
import torch

class Gluformer:
    def __init__(self):
        self.model = AutoModel.from_pretrained('njeffrie/Gluformer', trust_remote_code=True)
        self.config = AutoConfig.from_pretrained('njeffrie/Gluformer', trust_remote_code=True)
        self.model.eval()

    def predict(self, subject_id, timestamps, input_glucose):
        seq_len = len(input_glucose)
        pad_amount = self.config.len_seq - seq_len
        if (pad_amount > 0):
            glucose = [0.0] * pad_amount + input_glucose
            timestamps = [datetime.min] * pad_amount + timestamps
        else:
            glucose = input_glucose[:self.config.len_seq]
            timestamps = timestamps[:self.config.len_seq]
        with torch.no_grad():
            pred, log_var = self.model(subject_id, timestamps, glucose)
        return pred.numpy(), log_var.numpy()