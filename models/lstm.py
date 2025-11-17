import torch
from torch import nn

from transformers import AutoModel, AutoConfig


class LSTM:

    def __init__(
            self,
            huggingface_model_name: str = 'njeffrie/LSTMGlucosePrediction'):
        self.model = AutoModel.from_pretrained(huggingface_model_name,
                                               trust_remote_code=True)
        self.config = AutoConfig.from_pretrained(huggingface_model_name,
                                                 trust_remote_code=True)
        self.model.eval()

    def predict(self, subject_id, timestamps, input_glucose):
        if len(input_glucose) < self.config.len_seq:
            print(
                f'input_glucose length {len(input_glucose)} is less than config.len_seq {self.config.len_seq}'
            )
        assert (len(input_glucose) >= self.config.len_seq)
        glucose = input_glucose[-self.config.len_seq:].numpy()
        with torch.no_grad():
            pred = self.model(glucose)
        return pred.numpy()
