"""
Feature ablation training script.

Trains LSTM and UniTS models with different input feature combinations
(CGM, CGM+insulin, CGM+carbs, CGM+insulin+carbs) on the MetaboNet train split.

Hyperparameters can be tuned with Optuna (see optuna_search.py), then loaded via
--optuna_dir or --hparams_json for final training.

Usage:
    python -m studies.feature_ablation.train --data_path data/metabonet_train.parquet --device cuda
    python -m studies.feature_ablation.train --optuna_dir studies/feature_ablation/optuna \\
        --device cuda --amp --tf32
"""

from __future__ import annotations

import os
import sys
import time
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import click
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from torch import amp as torch_amp
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.lstm import LSTMModel
from models.UniTS import build_units_model
from models.gluforecast import GluForecastModel
from studies.feature_ablation.hparams import (
    AblationHyperParams,
    TrainHParams,
    LSTMHParams,
    UniTSHParams,
    GluForecastHParams,
    default_ablation_hparams,
    load_hparams_json,
)


FEATURE_SETS = {
    'cgm':               ['CGM'],
    'cgm_insulin':       ['CGM', 'Insulin'],
    'cgm_carbs':         ['CGM', 'Carbs'],
    'cgm_insulin_carbs': ['CGM', 'Insulin', 'Carbs'],
}

SEQ_LEN = 180
PRED_LEN = 12
MIN_SEQUENCE_LENGTH = SEQ_LEN + PRED_LEN
STEP_SIZE = 12

_CUDNN_BENCHMARK_SET = False
_TF32_CONFIGURED = False


def _maybe_configure_tf32(device: str, use_tf32: bool) -> None:
    """Enable TF32 for matmul on CUDA (Hopper/Ada-friendly); no-op if disabled."""
    global _TF32_CONFIGURED
    if not use_tf32 or _TF32_CONFIGURED:
        return
    if str(device).startswith('cuda'):
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    _TF32_CONFIGURED = True


def get_dataloader_kwargs(device: str, num_workers: int | None = None) -> dict:
    """Parallel workers + pinned memory so the GPU is not starved waiting on the CPU."""
    dev = str(device)
    if dev == 'cpu':
        return {'num_workers': 0, 'pin_memory': False}
    if dev == 'mps':
        nw = num_workers if num_workers is not None else min(4, (os.cpu_count() or 1))
        nw = max(0, nw)
        kw: dict = {'num_workers': nw, 'pin_memory': False}
        if nw > 0:
            kw['persistent_workers'] = True
            kw['prefetch_factor'] = 2
        return kw
    nw = num_workers if num_workers is not None else min(8, (os.cpu_count() or 4))
    nw = max(0, nw)
    kw = {'num_workers': nw, 'pin_memory': True}
    if nw > 0:
        kw['persistent_workers'] = True
        kw['prefetch_factor'] = 2
    return kw


def _maybe_enable_cudnn_benchmark(device: str) -> None:
    global _CUDNN_BENCHMARK_SET
    if str(device).startswith('cuda') and not _CUDNN_BENCHMARK_SET:
        torch.backends.cudnn.benchmark = True
        _CUDNN_BENCHMARK_SET = True


class GlucoseDataset(Dataset):
    """Sliding-window dataset over preprocessed MetaboNet parquet."""

    def __init__(self, df: pd.DataFrame, feature_cols: list[str]):
        self.feature_cols = feature_cols
        self.windows: list[tuple[np.ndarray, np.ndarray]] = []
        self._build_windows(df)

    def _build_windows(self, df: pd.DataFrame):
        for _, seq_data in df.groupby('SequenceID'):
            seq_len = len(seq_data)
            if seq_len < MIN_SEQUENCE_LENGTH:
                continue

            cgm = seq_data['CGM'].values.astype(np.float32)
            features = np.column_stack(
                [seq_data[c].values.astype(np.float32) for c in self.feature_cols]
            )

            for i in range(0, seq_len - MIN_SEQUENCE_LENGTH + 1, STEP_SIZE):
                x = features[i:i + SEQ_LEN]
                y = cgm[i + SEQ_LEN:i + MIN_SEQUENCE_LENGTH]
                self.windows.append((x, y))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x, y = self.windows[idx]
        return torch.from_numpy(x), torch.from_numpy(y)


def split_train_val(df: pd.DataFrame, val_fraction: float = 0.15, seed: int = 42):
    seq_ids = df['SequenceID'].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(seq_ids)
    n_val = max(1, int(len(seq_ids) * val_fraction))
    val_ids = set(seq_ids[:n_val])
    train_mask = ~df['SequenceID'].isin(val_ids)
    return df[train_mask], df[~train_mask]


def make_lstm(input_dim: int, device: str, lstm_hp: LSTMHParams) -> nn.Module:
    model = LSTMModel(
        input_dim=input_dim,
        hidden_dim=lstm_hp.hidden_dim,
        num_layers=lstm_hp.num_layers,
        pred_len=PRED_LEN,
        dropout=lstm_hp.dropout,
    )
    return model.to(device)


def make_units(device: str, units_hp: UniTSHParams) -> nn.Module:
    model = build_units_model(units_hp, seq_len=SEQ_LEN, pred_len=PRED_LEN)
    return model.to(device)


def make_gluforecast(
    feature_set: str, device: str, gluforecast_hp: GluForecastHParams,
) -> nn.Module:
    model = GluForecastModel(
        feature_set=feature_set,
        d_model=gluforecast_hp.d_model,
        n_heads=gluforecast_hp.n_heads,
        n_layers=gluforecast_hp.n_layers,
        max_len=gluforecast_hp.max_len,
        dropout=gluforecast_hp.dropout,
    )
    return model.to(device)


def forward_pred(model: nn.Module, model_type: str, x_batch: torch.Tensor) -> torch.Tensor:
    if model_type == 'lstm':
        return model(x_batch)
    if model_type == 'gluforecast':
        return model(x_batch)
    pred = model(
        x_enc=x_batch, x_mark_enc=None,
        task_id=0, task_name='long_term_forecast',
    )
    return pred[:, :, 0] if pred.ndim == 3 else pred


def run_training_loop(
    model: nn.Module,
    model_type: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    train_hp: TrainHParams,
    max_epochs: int,
    patience: int,
    verbose: bool = True,
    use_amp: bool = False,
    use_tf32: bool = False,
    optuna_trial: optuna.Trial | None = None,
    progress_callback: Callable[[dict], None] | None = None,
) -> tuple[float, dict, list]:
    _maybe_enable_cudnn_benchmark(device)
    _maybe_configure_tf32(device, use_tf32)
    non_blocking = str(device).startswith('cuda')
    use_autocast = bool(use_amp and str(device).startswith('cuda'))
    amp_dtype = torch.bfloat16
    if use_autocast and not torch.cuda.is_bf16_supported():
        amp_dtype = torch.float16
    use_fp16_scaler = use_autocast and amp_dtype == torch.float16
    scaler = torch_amp.GradScaler('cuda', enabled=use_fp16_scaler)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_hp.lr,
        weight_decay=train_hp.weight_decay,
    )
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_state: dict | None = None
    epochs_no_improve = 0
    history = []
    t_loop = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_sum, train_n = 0.0, 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device, non_blocking=non_blocking)
            y_batch = y_batch.to(device, non_blocking=non_blocking)
            optimizer.zero_grad(set_to_none=True)
            if use_autocast:
                with torch.autocast(device_type='cuda', dtype=amp_dtype):
                    pred = forward_pred(model, model_type, x_batch)
                    loss = criterion(pred, y_batch)
            else:
                pred = forward_pred(model, model_type, x_batch)
                loss = criterion(pred, y_batch)
            if use_fp16_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=train_hp.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=train_hp.grad_clip)
                optimizer.step()
            train_loss_sum += loss.item() * x_batch.size(0)
            train_n += x_batch.size(0)

        train_loss = train_loss_sum / max(train_n, 1)

        model.eval()
        val_loss_sum, val_n = 0.0, 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device, non_blocking=non_blocking)
                y_batch = y_batch.to(device, non_blocking=non_blocking)
                if use_autocast:
                    with torch.autocast(device_type='cuda', dtype=amp_dtype):
                        pred = forward_pred(model, model_type, x_batch)
                        v_t = criterion(pred, y_batch)
                else:
                    pred = forward_pred(model, model_type, x_batch)
                    v_t = criterion(pred, y_batch)
                v = float(v_t.item())
                val_loss_sum += v * x_batch.size(0)
                val_n += x_batch.size(0)

        val_loss = val_loss_sum / max(val_n, 1)
        row = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': optimizer.param_groups[0]['lr'],
        }
        history.append(row)
        if verbose:
            print(f"  Epoch {epoch:3d}/{max_epochs}  "
                  f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                  f"lr={row['lr']:.2e}  [{time.time() - t_loop:.0f}s]")
        if progress_callback is not None:
            progress_callback(row)

        if optuna_trial is not None:
            optuna_trial.report(val_loss, step=epoch)
            if optuna_trial.should_prune():
                raise optuna.TrialPruned()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f'  Early stopping at epoch {epoch} '
                          f'(no improvement for {patience} epochs)')
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val_loss, best_state or {}, history


@dataclass
class TrainJobConfig:
    model_type: str
    feature_set: str
    device: str = 'cpu'
    checkpoint_dir: str = 'studies/feature_ablation/checkpoints'
    num_workers: int | None = None
    use_amp: bool = False
    use_tf32: bool = False


def train_one(
    job: TrainJobConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    ablation_hp: AblationHyperParams,
    save_checkpoint: bool = True,
    quiet: bool = False,
) -> dict:
    """Train one variant. If save_checkpoint, write best weights to checkpoint_dir."""
    feature_cols = FEATURE_SETS[job.feature_set]
    input_dim = len(feature_cols)
    run_name = f'{job.model_type}_{job.feature_set}'
    train_hp = ablation_hp.train

    if not quiet:
        print(f'\n{"="*70}')
        print(f'Training {run_name}  (features={feature_cols}, device={job.device})')
        print(f'{"="*70}')

    train_ds = GlucoseDataset(train_df, feature_cols)
    val_ds = GlucoseDataset(val_df, feature_cols)
    if not quiet:
        print(f'  Train windows: {len(train_ds)},  Val windows: {len(val_ds)}')

    dl_kw = get_dataloader_kwargs(job.device, job.num_workers)
    train_loader = DataLoader(
        train_ds, batch_size=train_hp.batch_size, shuffle=True, **dl_kw)
    val_loader = DataLoader(
        val_ds, batch_size=train_hp.batch_size, shuffle=False, **dl_kw)

    if job.model_type == 'lstm':
        model = make_lstm(input_dim, job.device, ablation_hp.lstm)
    elif job.model_type == 'units':
        model = make_units(job.device, ablation_hp.units)
    elif job.model_type == 'gluforecast':
        model = make_gluforecast(job.feature_set, job.device, ablation_hp.gluforecast)
    else:
        raise ValueError(f'Unknown model type: {job.model_type}')

    t_start = time.time()
    best_val_loss, best_state, history = run_training_loop(
        model, job.model_type, train_loader, val_loader, job.device,
        train_hp, train_hp.max_epochs, train_hp.patience,
        verbose=not quiet,
        use_amp=job.use_amp,
        use_tf32=job.use_tf32,
    )

    if best_state:
        model.load_state_dict(best_state)

    total_time = time.time() - t_start
    last_epoch = history[-1]['epoch'] if history else 0

    if not quiet:
        print(f'  Best val_loss={best_val_loss:.4f}  Total time={total_time:.1f}s')

    ckpt_path = os.path.join(job.checkpoint_dir, f'{run_name}.pth')
    if save_checkpoint:
        os.makedirs(job.checkpoint_dir, exist_ok=True)
        ckpt = {
            'model_state_dict': model.state_dict(),
            'model_type': job.model_type,
            'feature_set': job.feature_set,
            'feature_cols': feature_cols,
            'input_dim': input_dim,
            'hidden_dim': ablation_hp.lstm.hidden_dim,
            'num_layers': ablation_hp.lstm.num_layers,
            'dropout': ablation_hp.lstm.dropout,
            'pred_len': PRED_LEN,
            'seq_len': SEQ_LEN,
            'best_val_loss': best_val_loss,
            'epoch': last_epoch,
            'train_hparams': asdict(ablation_hp.train),
            'lstm_hparams': asdict(ablation_hp.lstm),
            'units_hparams': asdict(ablation_hp.units),
            'gluforecast_hparams': asdict(ablation_hp.gluforecast),
        }
        torch.save(ckpt, ckpt_path)
        if not quiet:
            print(f'  Checkpoint saved to {ckpt_path}')

    return {
        'run_name': run_name,
        'best_val_loss': best_val_loss,
        'total_time_s': total_time,
        'epochs_trained': len(history),
        'history': history,
    }


def _optuna_json_path(
    optuna_dir: str, model_type: str, feature_set: str | None = None,
) -> str:
    """Best hyperparameters from Optuna.

    Layout (current): ``<optuna_dir>/<feature_set>/best_<model_type>.json`` per ablation.
    Legacy: ``<optuna_dir>/best_<model_type>.json`` (one JSON shared across feature sets).
    """
    if feature_set:
        return os.path.join(optuna_dir, feature_set, f'best_{model_type}.json')
    return os.path.join(optuna_dir, f'best_{model_type}.json')


def load_hparams_for_run(
    model_type: str,
    base_hp: AblationHyperParams,
    optuna_dir: str | None,
    hparams_json: str | None,
    feature_set: str | None = None,
) -> AblationHyperParams:
    """Load Optuna JSON for this model type (per feature_set when using per-ablation search)."""
    if hparams_json:
        return load_hparams_json(hparams_json)
    if optuna_dir:
        if feature_set:
            path_fs = _optuna_json_path(optuna_dir, model_type, feature_set)
            if os.path.isfile(path_fs):
                return load_hparams_json(path_fs)
        path = _optuna_json_path(optuna_dir, model_type)
        if os.path.isfile(path):
            return load_hparams_json(path)
    print(f'No Optuna hparams found for {model_type} {feature_set}, using base_hp')
    return base_hp


@click.command()
@click.option('--data_path', type=str, default='data/metabonet_train.parquet',
              help='Path to the preprocessed train parquet')
@click.option('--device', type=str, default='cpu',
              help='Device to train on (cpu, cuda, mps)')
@click.option('--epochs', type=int, default=50,
              help='Max epochs (used when not loading Optuna hparams)')
@click.option('--batch_size', type=int, default=64)
@click.option('--lr', type=float, default=1e-3)
@click.option('--patience', type=int, default=5)
@click.option('--models', type=str, default='lstm,units,gluforecast',
              help='Comma-separated model types to train')
@click.option('--feature_sets', type=str,
              default='cgm,cgm_insulin,cgm_carbs,cgm_insulin_carbs',
              help='Comma-separated feature sets to ablate')
@click.option('--optuna_dir', type=str, default=None,
              help='Optuna output dir: per feature_set best_<model>.json under '
                   '<dir>/<feature_set>/ (or legacy flat best_<model>.json)')
@click.option('--hparams_json', type=str, default=None,
              help='Single ablation hyperparameter JSON (trains one combo only)')
@click.option('--num_workers', type=int, default=None,
              help='DataLoader worker processes (default: auto by device; 0 forces single-threaded loading)')
@click.option('--amp', is_flag=True, default=False,
              help='CUDA AMP (bf16 when supported; else fp16 + GradScaler)')
@click.option('--tf32', is_flag=True, default=False,
              help='TF32 matmul on CUDA (recommended on H200 with fp32 matmuls)')
def main(data_path, device, epochs, batch_size, lr, patience, models,
         feature_sets, optuna_dir, hparams_json, num_workers, amp, tf32):
    print(f'Loading data from {data_path} ...')
    df = pd.read_parquet(data_path)
    print(f'  Loaded {len(df)} rows, '
          f'{df["SequenceID"].nunique()} sequences, '
          f'{df["PtID"].nunique()} patients')

    train_df, val_df = split_train_val(df)
    print(f'  Train: {train_df["SequenceID"].nunique()} sequences, '
          f'Val: {val_df["SequenceID"].nunique()} sequences')

    base = default_ablation_hparams()
    base.train.max_epochs = epochs
    base.train.batch_size = batch_size
    base.train.lr = lr
    base.train.patience = patience

    model_types = [m.strip() for m in models.split(',')]
    feat_sets = [f.strip() for f in feature_sets.split(',')]

    if hparams_json:
        ablation_hp = load_hparams_json(hparams_json)
        if len(model_types) != 1 or len(feat_sets) != 1:
            raise ValueError('--hparams_json requires exactly one model and one feature_set')
        ablation_hp.train.max_epochs = epochs
        summaries = [train_one(
            TrainJobConfig(
                model_type=model_types[0], feature_set=feat_sets[0],
                device=device, num_workers=num_workers,
                use_amp=amp, use_tf32=tf32,
            ),
            train_df, val_df, ablation_hp,
        )]
    else:
        summaries = []
        for model_type in model_types:
            for feat_set in feat_sets:
                hp = load_hparams_for_run(
                    model_type, base, optuna_dir, None, feat_set)
                if optuna_dir:
                    hp.train.max_epochs = epochs
                summaries.append(train_one(
                    TrainJobConfig(
                        model_type=model_type, feature_set=feat_set,
                        device=device, num_workers=num_workers,
                        use_amp=amp, use_tf32=tf32,
                    ),
                    train_df, val_df, hp,
                ))

    print(f'\n{"="*70}')
    print('TRAINING SUMMARY')
    print(f'{"="*70}')
    print(f'{"Run":<30s} {"Best Val Loss":>14s} {"Epochs":>8s} {"Time (s)":>10s}')
    print('-' * 70)
    for s in summaries:
        print(f'{s["run_name"]:<30s} {s["best_val_loss"]:>14.4f} '
              f'{s["epochs_trained"]:>8d} {s["total_time_s"]:>10.1f}')

    summary_path = os.path.join('studies/feature_ablation', 'training_summary.json')
    with open(summary_path, 'w') as f:
        serializable = []
        for s in summaries:
            entry = {k: v for k, v in s.items() if k != 'history'}
            serializable.append(entry)
        json.dump(serializable, f, indent=2)
    print(f'\nTraining summary saved to {summary_path}')


if __name__ == '__main__':
    main()
