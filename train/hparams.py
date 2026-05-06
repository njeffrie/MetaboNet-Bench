"""Hyperparameter dataclasses for feature ablation training and Optuna search."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TrainHParams:
    lr: float = 1e-3
    batch_size: int = 64
    weight_decay: float = 1e-4
    max_epochs: int = 50
    patience: int = 5
    grad_clip: float = 1.0


@dataclass
class LSTMHParams:
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1


@dataclass
class UniTSHParams:
    d_model: int = 128
    n_heads: int = 8
    e_layers: int = 2
    patch_len: int = 16
    stride: int = 16
    prompt_num: int = 10
    dropout: float = 0.1

    def __post_init__(self):
        if self.stride != self.patch_len:
            self.stride = self.patch_len


@dataclass
class GluForecastHParams:
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    max_len: int = 180
    dropout: float = 0.1


MODEL_HPARAM_CLASSES = {
    'lstm': LSTMHParams,
    'units': UniTSHParams,
    'gluforecast': GluForecastHParams,
}


@dataclass
class AblationHyperParams:
    train: TrainHParams = field(default_factory=TrainHParams)
    lstm: LSTMHParams = field(default_factory=LSTMHParams)
    units: UniTSHParams = field(default_factory=UniTSHParams)
    gluforecast: GluForecastHParams = field(default_factory=GluForecastHParams)


def default_ablation_hparams() -> AblationHyperParams:
    return AblationHyperParams()


def save_hparams_json(
    hp: AblationHyperParams, path: str | Path, model_type: str,
) -> None:
    """Write only `train` and the section for `model_type` to JSON."""
    if model_type not in MODEL_HPARAM_CLASSES:
        raise ValueError(f'Unknown model_type {model_type!r}')
    payload: dict[str, Any] = {
        'model_type': model_type,
        'train': asdict(hp.train),
        model_type: asdict(getattr(hp, model_type)),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)


def load_hparams_json(path: str | Path) -> tuple[str, AblationHyperParams]:
    """Load a model-specific hparams JSON, filling missing sections with defaults."""
    with open(path) as f:
        d = json.load(f)
    present = [m for m in MODEL_HPARAM_CLASSES if m in d]
    model_type = d.get('model_type') or (present[0] if len(present) == 1 else None)
    if model_type not in MODEL_HPARAM_CLASSES:
        raise ValueError(f'Could not determine model_type from {path}')
    hp = AblationHyperParams(train=TrainHParams(**d.get('train', {})))
    setattr(hp, model_type, MODEL_HPARAM_CLASSES[model_type](**d.get(model_type, {})))
    return model_type, hp
