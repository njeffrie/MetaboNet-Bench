import torch
from torch import nn

from transformers import AutoModel, AutoConfig


class LSTM:

    def __init__(
            self,
            huggingface_model_name: str = 'njeffrie/LSTMGlucosePrediction', device='cpu'):
        self.model = AutoModel.from_pretrained(huggingface_model_name,
                                               trust_remote_code=True)
        self.config = AutoConfig.from_pretrained(huggingface_model_name,
                                                 trust_remote_code=True)
        self.model.to(device)
        self.model.eval()

    def predict(self, timestamps, cgm, insulin, carbs):
        if cgm.shape[1] < self.config.len_seq:
            print(f'cgm length {len(cgm)} is less than config.len_seq {self.config.len_seq}')
        assert(cgm.shape[1] >= self.config.len_seq)
        glucose = cgm[:, -self.config.len_seq:]
        with torch.no_grad():
            pred = self.model(glucose.to(self.model.device))
        return pred.numpy()
