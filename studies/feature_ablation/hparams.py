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
class AblationHyperParams:
    train: TrainHParams = field(default_factory=TrainHParams)
    lstm: LSTMHParams = field(default_factory=LSTMHParams)
    units: UniTSHParams = field(default_factory=UniTSHParams)


def default_ablation_hparams() -> AblationHyperParams:
    return AblationHyperParams()


def ablation_hparams_to_dict(hp: AblationHyperParams) -> dict[str, Any]:
    return {
        'train': asdict(hp.train),
        'lstm': asdict(hp.lstm),
        'units': asdict(hp.units),
    }


def ablation_hparams_from_dict(d: dict[str, Any]) -> AblationHyperParams:
    return AblationHyperParams(
        train=TrainHParams(**d.get('train', {})),
        lstm=LSTMHParams(**d.get('lstm', {})),
        units=UniTSHParams(**d.get('units', {})),
    )


def save_hparams_json(hp: AblationHyperParams, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(ablation_hparams_to_dict(hp), f, indent=2)


def load_hparams_json(path: str | Path) -> AblationHyperParams:
    with open(path) as f:
        d = json.load(f)
    return ablation_hparams_from_dict(d)
