"""Local wrapper for the LightGBM MetaboNet HF Hub model.

The package import for ``lightgbm`` happens at module top, BEFORE any imports
that drag in ``torch``. On macOS, importing ``torch`` first and then loading a
LightGBM booster can segfault due to two OpenMP runtimes being mapped into the
process; importing ``lightgbm`` first avoids the duplicate-libomp situation.
See https://github.com/dmlc/xgboost/issues/1715 and lightgbm's macOS notes.
"""

# Pre-import lightgbm so its libomp/lib_lightgbm linkage wins over torch's.
# Absolute import works fine even from this module: Python resolves ``lightgbm``
# via ``sys.modules`` (first against ``models.lightgbm``, then against the
# top-level ``lightgbm`` package on sys.path).
import lightgbm  # noqa: F401

from typing import Optional

from transformers import AutoConfig, AutoModel


class LightGBM:

    def __init__(
        self,
        huggingface_model_name: str = 'anonymous-4FAD/LightGBM',
        ablation: Optional[str] = None,
        device: str = 'cpu',
    ):
        # ``ablation=None`` means "use repo default from config.json".
        config_kwargs = {'trust_remote_code': True}
        if ablation is not None:
            config_kwargs['ablation'] = ablation
        self.config = AutoConfig.from_pretrained(huggingface_model_name, **config_kwargs)
        self.model = AutoModel.from_pretrained(
            huggingface_model_name, trust_remote_code=True, config=self.config)
        self.model.eval()
        # boosters run on CPU; only the sentinel buffer follows ``device``
        self.device = device
        self.model.to(self.device)
        self.ablation = self.config.ablation

    def predict(self, timestamps, cgm, insulin, carbs):
        return self.model.predict(timestamps, cgm, insulin, carbs)
