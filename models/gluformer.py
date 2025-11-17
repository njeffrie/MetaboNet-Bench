from transformers import AutoModel, AutoConfig
from datetime import datetime
import torch
import pandas as pd


class Gluformer:

    def __init__(self, huggingface_model_name: str = 'njeffrie/Gluformer'):
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
        glucose = input_glucose[-self.config.len_seq:]
        timestamps = [
            pd.to_datetime(date) for date in timestamps[-self.config.len_seq:]
        ]
        with torch.no_grad():
            pred, log_var = self.model(subject_id, timestamps, glucose)
        return pred.numpy()
