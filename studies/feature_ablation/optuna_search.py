"""
Optuna hyperparameter search for feature ablation (one study per model × feature set).

1) Run this script to produce best_<model>_<feature_set>.json under --out_dir
2) Run training with: python -m studies.feature_ablation.train --optuna_dir <out_dir> ...

Example:
    python -m studies.feature_ablation.optuna_search \\
        --data_path data/metabonet_train.parquet --device cuda --n_trials 30 \\
        --max_epochs_per_trial 20 --out_dir studies/feature_ablation/optuna
"""

from __future__ import annotations

import os
import sys
import json
import random
from pathlib import Path

import click
import numpy as np
import optuna
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from studies.feature_ablation.train import (
    FEATURE_SETS,
    GlucoseDataset,
    split_train_val,
    make_lstm,
    make_units,
    run_training_loop,
)
from studies.feature_ablation.hparams import (
    AblationHyperParams,
    TrainHParams,
    LSTMHParams,
    UniTSHParams,
    default_ablation_hparams,
    save_hparams_json,
)


def _sample_lstm_hparams(trial: optuna.Trial, max_epochs: int) -> AblationHyperParams:
    base = default_ablation_hparams()
    return AblationHyperParams(
        train=TrainHParams(
            lr=trial.suggest_float('lr', 1e-5, 1e-2, log=True),
            batch_size=trial.suggest_categorical('batch_size', [32, 64, 128]),
            weight_decay=trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
            max_epochs=max_epochs,
            patience=trial.suggest_int('patience', 3, 7),
            grad_clip=trial.suggest_float('grad_clip', 0.5, 2.0),
        ),
        lstm=LSTMHParams(
            hidden_dim=trial.suggest_categorical('hidden_dim', [64, 128, 256]),
            num_layers=trial.suggest_int('num_layers', 1, 3),
            dropout=trial.suggest_float('dropout', 0.0, 0.35),
        ),
        units=base.units,
    )


def _sample_units_hparams(trial: optuna.Trial, max_epochs: int) -> AblationHyperParams:
    base = default_ablation_hparams()
    d_model = trial.suggest_categorical('d_model', [64, 128, 256])
    n_heads = trial.suggest_categorical('n_heads', [4, 8])
    if d_model % n_heads != 0:
        raise optuna.TrialPruned()
    patch_len = trial.suggest_categorical(
        'patch_len', [8, 10, 12, 15, 16, 20, 24, 30])
    return AblationHyperParams(
        train=TrainHParams(
            lr=trial.suggest_float('lr', 1e-5, 1e-2, log=True),
            batch_size=trial.suggest_categorical('batch_size', [16, 32, 64]),
            weight_decay=trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
            max_epochs=max_epochs,
            patience=trial.suggest_int('patience', 3, 7),
            grad_clip=trial.suggest_float('grad_clip', 0.5, 2.0),
        ),
        lstm=base.lstm,
        units=UniTSHParams(
            d_model=d_model,
            n_heads=n_heads,
            e_layers=trial.suggest_int('e_layers', 1, 4),
            patch_len=patch_len,
            stride=patch_len,
            prompt_num=trial.suggest_int('prompt_num', 4, 16),
            dropout=trial.suggest_float('dropout', 0.0, 0.35),
        ),
    )


def _ablation_from_frozen_params(
    params: dict, model_type: str, max_epochs: int,
) -> AblationHyperParams:
    """Rebuild AblationHyperParams from Optuna trial.params (no re-sampling)."""
    base = default_ablation_hparams()
    if model_type == 'lstm':
        return AblationHyperParams(
            train=TrainHParams(
                lr=params['lr'],
                batch_size=params['batch_size'],
                weight_decay=params['weight_decay'],
                max_epochs=max_epochs,
                patience=params['patience'],
                grad_clip=params['grad_clip'],
            ),
            lstm=LSTMHParams(
                hidden_dim=params['hidden_dim'],
                num_layers=params['num_layers'],
                dropout=params['dropout'],
            ),
            units=base.units,
        )
    if model_type == 'units':
        pl = int(params['patch_len'])
        return AblationHyperParams(
            train=TrainHParams(
                lr=params['lr'],
                batch_size=params['batch_size'],
                weight_decay=params['weight_decay'],
                max_epochs=max_epochs,
                patience=params['patience'],
                grad_clip=params['grad_clip'],
            ),
            lstm=base.lstm,
            units=UniTSHParams(
                d_model=params['d_model'],
                n_heads=params['n_heads'],
                e_layers=params['e_layers'],
                patch_len=pl,
                stride=pl,
                prompt_num=params['prompt_num'],
                dropout=params['dropout'],
            ),
        )
    raise ValueError(model_type)


def run_one_study(
    model_type: str,
    feature_set: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    device: str,
    n_trials: int,
    max_epochs_per_trial: int,
    seed: int,
    out_dir: str,
    study_name: str | None,
) -> None:
    feature_cols = FEATURE_SETS[feature_set]
    input_dim = len(feature_cols)
    train_ds = GlucoseDataset(train_df, feature_cols)
    val_ds = GlucoseDataset(val_df, feature_cols)

    def objective(trial: optuna.Trial) -> float:
        if model_type == 'lstm':
            hp = _sample_lstm_hparams(trial, max_epochs_per_trial)
            model = make_lstm(input_dim, device, hp.lstm)
        elif model_type == 'units':
            hp = _sample_units_hparams(trial, max_epochs_per_trial)
            model = make_units(device, hp.units)
        else:
            raise ValueError(model_type)

        train_loader = DataLoader(
            train_ds, batch_size=hp.train.batch_size, shuffle=True,
            num_workers=0, pin_memory=(device != 'cpu'))
        val_loader = DataLoader(
            val_ds, batch_size=hp.train.batch_size, shuffle=False,
            num_workers=0, pin_memory=(device != 'cpu'))

        best_val_loss, _, _ = run_training_loop(
            model, model_type, train_loader, val_loader, device,
            hp.train, hp.train.max_epochs, hp.train.patience,
            verbose=False,
        )
        return float(best_val_loss)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study_id = study_name or f'{model_type}_{feature_set}'
    study = optuna.create_study(direction='minimize', sampler=sampler, study_name=study_id)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_hp = _ablation_from_frozen_params(
        study.best_trial.params, model_type, max_epochs_per_trial)

    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f'best_{model_type}_{feature_set}.json')
    save_hparams_json(best_hp, json_path)

    meta = {
        'model_type': model_type,
        'feature_set': feature_set,
        'best_value': study.best_value,
        'best_params': study.best_params,
        'n_trials': n_trials,
        'max_epochs_per_trial': max_epochs_per_trial,
        'hparams_file': json_path,
    }
    meta_path = os.path.join(out_dir, f'meta_{model_type}_{feature_set}.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f'\n[{model_type} / {feature_set}] best_val={study.best_value:.6f}')
    print(f'  Saved {json_path}')
    print(f'  Meta {meta_path}')


@click.command()
@click.option('--data_path', type=str, default='data/metabonet_train.parquet')
@click.option('--device', type=str, default='cpu')
@click.option('--n_trials', type=int, default=30)
@click.option('--max_epochs_per_trial', type=int, default=20)
@click.option('--seed', type=int, default=42)
@click.option('--models', type=str, default='lstm,units',
              help='Model types to tune (one study each)')
@click.option('--tune_feature_set', type=str, default='cgm_insulin_carbs',
              help='Feature set used as the tuning objective')
@click.option('--out_dir', type=str, default='studies/feature_ablation/optuna')
@click.option('--study_name', type=str, default=None,
              help='Optional prefix for Optuna study names')
def main(
    data_path, device, n_trials, max_epochs_per_trial, seed, models,
    tune_feature_set, out_dir, study_name,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f'Loading {data_path} ...')
    df = pd.read_parquet(data_path)
    train_df, val_df = split_train_val(df)
    print(f'  Train seq: {train_df["SequenceID"].nunique()}, '
          f'Val seq: {val_df["SequenceID"].nunique()}')

    model_types = [m.strip() for m in models.split(',')]
    fs = tune_feature_set.strip()
    if fs not in FEATURE_SETS:
        raise ValueError(f'--tune_feature_set must be one of {list(FEATURE_SETS.keys())}')

    for mt in model_types:
        sn = f'{study_name}_{mt}' if study_name else None
        run_one_study(
            mt, fs, train_df, val_df, device,
            n_trials, max_epochs_per_trial, seed, out_dir, sn,
        )

    print('\nDone. Run final training with:')
    print(f'  python -m studies.feature_ablation.train --optuna_dir {out_dir} '
          f'--data_path {data_path} --device {device}')


if __name__ == '__main__':
    main()
