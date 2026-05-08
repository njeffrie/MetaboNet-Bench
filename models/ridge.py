"""Local wrapper for the Ridge MetaboNet HF Hub model.

Mirrors the [models/gluformer.py](gluformer.py) pattern so the benchmark and
``models.models.get_model`` can use Ridge the same way.
"""

from transformers import AutoConfig, AutoModel


class Ridge:

    def __init__(
        self,
        huggingface_model_name: str = 'anonymous-4FAD/Ridge',
        ablation: str = 'all',
        device: str = 'cpu',
    ):
        self.config = AutoConfig.from_pretrained(
            huggingface_model_name, trust_remote_code=True, ablation=ablation)
        self.model = AutoModel.from_pretrained(
            huggingface_model_name, trust_remote_code=True, config=self.config)
        self.model.eval()
        self.device = device
        self.model.to(self.device)
        self.ablation = self.config.ablation

    def predict(self, timestamps, cgm, insulin, carbs):
        return self.model.predict(timestamps, cgm, insulin, carbs)
