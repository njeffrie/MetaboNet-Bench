"""
Micro-benchmark: seconds per epoch for representative LSTM and heavy UniTS configs.

Use on the target GPU (e.g. H200) before long Optuna + ablation runs. Results can
be fed into estimate_runtime.py for a 24h budget check.

Example:
    python -m studies.feature_ablation.bench --data_path data/metabonet_train.parquet \\
        --device cuda --epochs 2 --amp --tf32
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import click
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from studies.feature_ablation.hparams import AblationHyperParams, LSTMHParams, TrainHParams, UniTSHParams
from studies.feature_ablation.train import TrainJobConfig, split_train_val, train_one


def _lstm_mid_hparams(epochs: int) -> AblationHyperParams:
    return AblationHyperParams(
        train=TrainHParams(
            lr=1e-3,
            batch_size=64,
            weight_decay=1e-4,
            max_epochs=epochs,
            patience=max(epochs + 1, 10),
            grad_clip=1.0,
        ),
        lstm=LSTMHParams(hidden_dim=128, num_layers=2, dropout=0.1),
    )


def _units_heavy_hparams(epochs: int) -> AblationHyperParams:
    return AblationHyperParams(
        train=TrainHParams(
            lr=1e-3,
            batch_size=32,
            weight_decay=1e-4,
            max_epochs=epochs,
            patience=max(epochs + 1, 10),
            grad_clip=1.0,
        ),
        units=UniTSHParams(
            d_model=256,
            n_heads=8,
            e_layers=4,
            patch_len=8,
            stride=8,
            prompt_num=16,
            dropout=0.1,
        ),
    )


@click.command()
@click.option('--data_path', type=str, default='data/metabonet_train.parquet')
@click.option('--device', type=str, default='cuda')
@click.option('--epochs', type=int, default=2, help='Epochs per model (keep small for timing)')
@click.option('--feature_set', type=str, default='cgm_insulin_carbs',
              help='Feature set (same as typical Optuna tune set)')
@click.option('--num_workers', type=int, default=None)
@click.option('--amp', is_flag=True, default=False, help='CUDA AMP (bf16 when supported)')
@click.option('--tf32', is_flag=True, default=False, help='TF32 matmul on CUDA')
@click.option('--json_out', type=str, default=None,
              help='Optional path to write timings JSON for estimate_runtime.py')
def main(
    data_path: str,
    device: str,
    epochs: int,
    feature_set: str,
    num_workers: int | None,
    amp: bool,
    tf32: bool,
    json_out: str | None,
):
    print(f'Loading {data_path} ...')
    df = pd.read_parquet(data_path)
    train_df, val_df = split_train_val(df)
    print(
        f'  Train seq: {train_df["SequenceID"].nunique()}, '
        f'Val seq: {val_df["SequenceID"].nunique()}'
    )

    results: dict = {
        'data_path': data_path,
        'device': device,
        'epochs_requested': epochs,
        'amp': amp,
        'tf32': tf32,
        'feature_set': feature_set,
        'lstm': {},
        'units': {},
    }

    # LSTM (mid config)
    t0 = time.perf_counter()
    hp_l = _lstm_mid_hparams(epochs)
    s_lstm = train_one(
        TrainJobConfig(
            model_type='lstm',
            feature_set=feature_set,
            device=device,
            num_workers=num_workers,
            use_amp=amp,
            use_tf32=tf32,
        ),
        train_df,
        val_df,
        hp_l,
        save_checkpoint=False,
        quiet=True,
    )
    wall_lstm = time.perf_counter() - t0
    ep_l = max(s_lstm['epochs_trained'], 1)
    sec_per_ep_l = wall_lstm / ep_l
    results['lstm'] = {
        'wall_time_s': wall_lstm,
        'epochs_trained': s_lstm['epochs_trained'],
        'sec_per_epoch': sec_per_ep_l,
        'best_val_loss': s_lstm['best_val_loss'],
    }
    print(
        f'\n[LSTM mid] wall={wall_lstm:.2f}s  epochs={s_lstm["epochs_trained"]}  '
        f'~{sec_per_ep_l:.3f}s/epoch  best_val={s_lstm["best_val_loss"]:.6f}'
    )

    if str(device).startswith('cuda'):
        torch.cuda.synchronize()

    # UniTS (heavy config)
    t1 = time.perf_counter()
    hp_u = _units_heavy_hparams(epochs)
    s_units = train_one(
        TrainJobConfig(
            model_type='units',
            feature_set=feature_set,
            device=device,
            num_workers=num_workers,
            use_amp=amp,
            use_tf32=tf32,
        ),
        train_df,
        val_df,
        hp_u,
        save_checkpoint=False,
        quiet=True,
    )
    wall_units = time.perf_counter() - t1
    ep_u = max(s_units['epochs_trained'], 1)
    sec_per_ep_u = wall_units / ep_u
    results['units'] = {
        'wall_time_s': wall_units,
        'epochs_trained': s_units['epochs_trained'],
        'sec_per_epoch': sec_per_ep_u,
        'best_val_loss': s_units['best_val_loss'],
    }
    print(
        f'[UniTS heavy] wall={wall_units:.2f}s  epochs={s_units["epochs_trained"]}  '
        f'~{sec_per_ep_u:.3f}s/epoch  best_val={s_units["best_val_loss"]:.6f}'
    )

    print(
        '\nUse sec/epoch with: python -m studies.feature_ablation.estimate_runtime '
        f'--lstm-sec-per-epoch {sec_per_ep_l:.6f} --units-sec-per-epoch {sec_per_ep_u:.6f} ...'
    )

    if json_out:
        out_path = Path(json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'Wrote {out_path}')


if __name__ == '__main__':
    main()
