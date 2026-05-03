"""
Optuna hyperparameter search for feature ablation.

1) For each --tune_feature_sets entry, writes best_<model>.json under:
       <out_dir>/<feature_set>/
   (one hyperparameter search per ablation feature set).
2) Run training with: python -m studies.feature_ablation.train --optuna_dir <out_dir> ...
3) Model architecture follows default_ablation_hparams().
   Optuna searches only: lr, batch_size, weight_decay (patience and grad_clip fixed to defaults).

Example:
    python -m studies.feature_ablation.optuna_search \\
        --data_path data/metabonet_train.parquet --device cuda --n_trials 30 \\
        --max_epochs_per_trial 20 --out_dir studies/feature_ablation/optuna \\
        --amp --tf32
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
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from studies.feature_ablation.train import (
    FEATURE_SETS,
    GlucoseDataset,
    split_train_val,
    make_lstm,
    make_units,
    make_gluforecast,
    run_training_loop,
    get_dataloader_kwargs,
)
from studies.feature_ablation.hparams import (
    AblationHyperParams,
    TrainHParams,
    default_ablation_hparams,
    save_hparams_json,
)


def _sample_train_only_hparams(trial: optuna.Trial, max_epochs: int) -> TrainHParams:
    """Optuna search space: lr, batch_size, weight_decay only."""
    base = default_ablation_hparams()
    return TrainHParams(
        lr=trial.suggest_float('lr', 1e-5, 1e-3, log=True),
        batch_size=trial.suggest_categorical(
            'batch_size', [256]),
        weight_decay=trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True),
        max_epochs=max_epochs,
        patience=base.train.patience,
        grad_clip=base.train.grad_clip,
    )


def _sample_lstm_hparams(trial: optuna.Trial, max_epochs: int) -> AblationHyperParams:
    base = default_ablation_hparams()
    return AblationHyperParams(
        train=_sample_train_only_hparams(trial, max_epochs),
        lstm=base.lstm,
        units=base.units,
        gluforecast=base.gluforecast,
    )


def _sample_units_hparams(trial: optuna.Trial, max_epochs: int) -> AblationHyperParams:
    base = default_ablation_hparams()
    return AblationHyperParams(
        train=_sample_train_only_hparams(trial, max_epochs),
        lstm=base.lstm,
        units=base.units,
        gluforecast=base.gluforecast,
    )


def _sample_gluforecast_hparams(
    trial: optuna.Trial, max_epochs: int,
) -> AblationHyperParams:
    base = default_ablation_hparams()
    return AblationHyperParams(
        train=_sample_train_only_hparams(trial, max_epochs),
        lstm=base.lstm,
        units=base.units,
        gluforecast=base.gluforecast,
    )


def _ablation_from_frozen_params(
    params: dict, _model_type: str, max_epochs: int,
) -> AblationHyperParams:
    """Rebuild AblationHyperParams from Optuna trial.params (lr, batch_size, weight_decay)."""
    base = default_ablation_hparams()
    train = TrainHParams(
        lr=params['lr'],
        batch_size=params['batch_size'],
        weight_decay=params['weight_decay'],
        max_epochs=max_epochs,
        patience=base.train.patience,
        grad_clip=base.train.grad_clip,
    )
    return AblationHyperParams(
        train=train,
        lstm=base.lstm,
        units=base.units,
        gluforecast=base.gluforecast,
    )


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
    num_workers: int | None,
    use_amp: bool = False,
    use_tf32: bool = False,
    use_pruner: bool = True,
    pruner_n_startup_trials: int = 5,
    pruner_n_warmup_steps: int = 3,
) -> None:
    feature_cols = FEATURE_SETS[feature_set]
    input_dim = len(feature_cols)
    train_ds = GlucoseDataset(train_df, feature_cols)
    val_ds = GlucoseDataset(val_df, feature_cols)

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner: optuna.pruners.BasePruner | None = None
    if use_pruner:
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=pruner_n_startup_trials,
            n_warmup_steps=pruner_n_warmup_steps,
        )
    study_id = study_name or f'{model_type}_{feature_set}'
    study = optuna.create_study(
        direction='minimize',
        sampler=sampler,
        pruner=pruner,
        study_name=study_id,
    )

    pbar = tqdm(
        total=n_trials * max_epochs_per_trial,
        desc=f'Optuna {model_type}/{feature_set}',
        unit='epoch',
        dynamic_ncols=True,
    )
    current_trial = {'number': None, 'epochs': 0}

    def best_value_text() -> str:
        try:
            return f'{study.best_value:.4f}'
        except ValueError:
            return 'n/a'

    def objective_with_progress(trial: optuna.Trial) -> float:
        current_trial['number'] = trial.number
        current_trial['epochs'] = 0

        def update_progress(row: dict) -> None:
            current_trial['epochs'] += 1
            pbar.update(1)
            pbar.set_postfix(
                trial=trial.number,
                epoch=row['epoch'],
                val=f"{row['val_loss']:.4f}",
                best=best_value_text(),
                refresh=False,
            )

        try:
            if model_type == 'lstm':
                hp = _sample_lstm_hparams(trial, max_epochs_per_trial)
                model = make_lstm(input_dim, device, hp.lstm)
            elif model_type == 'units':
                hp = _sample_units_hparams(trial, max_epochs_per_trial)
                model = make_units(device, hp.units)
            elif model_type == 'gluforecast':
                hp = _sample_gluforecast_hparams(trial, max_epochs_per_trial)
                model = make_gluforecast(feature_set, device, hp.gluforecast)
            else:
                raise ValueError(model_type)

            dl_kw = get_dataloader_kwargs(device, num_workers)
            train_loader = DataLoader(
                train_ds, batch_size=hp.train.batch_size, shuffle=True, **dl_kw)
            val_loader = DataLoader(
                val_ds, batch_size=hp.train.batch_size, shuffle=False, **dl_kw)

            trial_for_loop = trial if use_pruner else None
            best_val_loss, _, _ = run_training_loop(
                model, model_type, train_loader, val_loader, device,
                hp.train, hp.train.max_epochs, hp.train.patience,
                verbose=False,
                use_amp=use_amp,
                use_tf32=use_tf32,
                optuna_trial=trial_for_loop,
                progress_callback=update_progress,
            )
            return float(best_val_loss)
        finally:
            remaining = max_epochs_per_trial - current_trial['epochs']
            if remaining > 0:
                pbar.update(remaining)

    try:
        study.optimize(
            objective_with_progress,
            n_trials=n_trials,
            show_progress_bar=False,
        )
    finally:
        pbar.close()

    best_hp = _ablation_from_frozen_params(
        study.best_trial.params, model_type, max_epochs_per_trial)

    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f'best_{model_type}.json')
    save_hparams_json(best_hp, json_path)

    meta = {
        'model_type': model_type,
        'feature_set': feature_set,
        'best_value': study.best_value,
        'best_params': study.best_params,
        'n_trials': n_trials,
        'max_epochs_per_trial': max_epochs_per_trial,
        'hparams_file': json_path,
        'use_amp': use_amp,
        'use_tf32': use_tf32,
        'median_pruner': use_pruner,
        'pruner_n_startup_trials': pruner_n_startup_trials,
        'pruner_n_warmup_steps': pruner_n_warmup_steps,
    }
    meta_path = os.path.join(out_dir, f'meta_{model_type}.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f'\n[{model_type}] tune_feature_set={feature_set} best_val={study.best_value:.6f}')
    print(f'  Saved {json_path}')
    print(f'  Meta {meta_path}')


def _trials_for_model(
    model_type: str,
    n_trials: int,
    n_trials_lstm: int | None,
    n_trials_units: int | None,
    n_trials_gluforecast: int | None,
) -> int:
    if model_type == 'lstm':
        return int(n_trials_lstm if n_trials_lstm is not None else n_trials)
    if model_type == 'units':
        return int(n_trials_units if n_trials_units is not None else n_trials)
    if model_type == 'gluforecast':
        return int(
            n_trials_gluforecast
            if n_trials_gluforecast is not None else n_trials
        )
    raise ValueError(model_type)


@click.command()
@click.option('--data_path', type=str, default='data/metabonet_train.parquet')
@click.option('--device', type=str, default='cpu')
@click.option('--n_trials', type=int, default=30,
              help='Trials per study if --n-trials-lstm / --n-trials-units not set')
@click.option('--n-trials-lstm', 'n_trials_lstm', type=int, default=None,
              help='Optuna trials for LSTM study (default: same as --n_trials)')
@click.option('--n-trials-units', 'n_trials_units', type=int, default=None,
              help='Optuna trials for UniTS study (default: same as --n_trials)')
@click.option('--n-trials-gluforecast', 'n_trials_gluforecast', type=int, default=None,
              help='Optuna trials for GluForecast study (default: same as --n_trials)')
@click.option('--max_epochs_per_trial', type=int, default=20)
@click.option('--seed', type=int, default=42)
@click.option('--models', type=str, default='lstm,units,gluforecast',
              help='Model types to tune (one study each per feature set)')
@click.option('--tune-feature-sets', 'tune_feature_sets', type=str,
              default='cgm,cgm_insulin,cgm_carbs,cgm_insulin_carbs',
              help='Comma-separated FEATURE_SETS keys; one Optuna pass per entry '
                   '(writes under out_dir/<feature_set>/)')
@click.option('--out_dir', type=str, default='studies/feature_ablation/optuna')
@click.option('--study_name', type=str, default=None,
              help='Optional prefix for Optuna study names')
@click.option('--num_workers', type=int, default=None,
              help='DataLoader worker processes (default: auto; e.g. 4–8 on MPS/CUDA)')
@click.option('--amp', is_flag=True, default=False,
              help='CUDA AMP (bf16 when supported)')
@click.option('--tf32', is_flag=True, default=False,
              help='TF32 matmul on CUDA (e.g. H200)')
@click.option('--no-pruner', is_flag=True, default=False,
              help='Disable MedianPruner (full epochs every trial)')
@click.option('--pruner-n-startup-trials', type=int, default=5,
              help='MedianPruner: trials before pruning')
@click.option('--pruner-n-warmup-steps', type=int, default=3,
              help='MedianPruner: epochs (steps) before pruning')
@click.option('--subset-size', type=int, default=None,
              help='Subset size to run the study on')
def main(
    data_path, device, n_trials, n_trials_lstm, n_trials_units, n_trials_gluforecast,
    max_epochs_per_trial, seed, models,
    tune_feature_sets, out_dir, study_name, num_workers,
    amp, tf32, no_pruner, pruner_n_startup_trials, pruner_n_warmup_steps,
    subset_size,
    ):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f'Loading {data_path} ...')
    df = pd.read_parquet(data_path)
    if subset_size is not None:
        df = df.sample(subset_size)
    train_df, val_df = split_train_val(df)
    print(f'  Train seq: {train_df["SequenceID"].nunique()}, '
          f'Val seq: {val_df["SequenceID"].nunique()}')

    model_types = [m.strip() for m in models.split(',')]
    tune_list = [s.strip() for s in tune_feature_sets.split(',') if s.strip()]
    for fs in tune_list:
        if fs not in FEATURE_SETS:
            raise ValueError(
                f'--tune_feature_sets entries must be one of {list(FEATURE_SETS.keys())}: '
                f'bad {fs!r}'
            )

    for fs in tune_list:
        fs_out = os.path.join(out_dir, fs)
        for mt in model_types:
            nt = _trials_for_model(
                mt, n_trials, n_trials_lstm, n_trials_units, n_trials_gluforecast)
            sn = f'{study_name}_{fs}_{mt}' if study_name else None
            run_one_study(
                mt, fs, train_df, val_df, device,
                nt, max_epochs_per_trial, seed, fs_out, sn, num_workers,
                use_amp=amp,
                use_tf32=tf32,
                use_pruner=not no_pruner,
                pruner_n_startup_trials=pruner_n_startup_trials,
                pruner_n_warmup_steps=pruner_n_warmup_steps,
            )

    train_flags = []
    if amp:
        train_flags.append('--amp')
    if tf32:
        train_flags.append('--tf32')
    extra = (' \\\n    ' + ' '.join(train_flags)) if train_flags else ''
    print('\nDone. Outputs are under <out_dir>/<feature_set>/best_<model>.json.')
    print('Run final training with:')
    print(f'  python -m studies.feature_ablation.train --optuna_dir {out_dir} '
          f'--data_path {data_path} --device {device}{extra}')


if __name__ == '__main__':
    main()
