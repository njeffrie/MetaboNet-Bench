from transformers import AutoModel, AutoConfig
from datetime import datetime
import torch
import pandas as pd


class Gluformer:

    def __init__(self, huggingface_model_name: str = 'njeffrie/Gluformer', device='cpu'):
        self.model = AutoModel.from_pretrained(huggingface_model_name,
                                               trust_remote_code=True)
        self.config = AutoConfig.from_pretrained(huggingface_model_name,
                                                 trust_remote_code=True)
        self.model.eval()
        self.device = device
        self.model.to(self.device)

    def predict(self, timestamps, cgm, insulin, carbs):
        subject_id = 0
        assert(cgm.shape[1] >= self.config.len_seq)
        glucose = cgm[:, -self.config.len_seq:]
        with torch.no_grad():
            pred, log_var = self.model(subject_id, timestamps, glucose)
        return pred.squeeze(-1).cpu().numpy()
