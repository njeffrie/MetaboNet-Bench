"""
Feature ablation training script.

Trains LSTM and UniTS models with different input feature combinations
(CGM, CGM+insulin, CGM+carbs, CGM+insulin+carbs) on the MetaboNet train split.

Usage:
    python -m studies.feature_ablation.train --data_path data/metabonet_train.parquet --device cuda
"""

import os
import sys
import time
import json
from dataclasses import dataclass
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Ensure project root is on the path so model imports work
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.lstm_trainable import TrainableLSTMModel
from models.units_trainable import build_units_model


FEATURE_SETS = {
    'cgm':               ['CGM'],
    'cgm_insulin':       ['CGM', 'Insulin'],
    'cgm_carbs':         ['CGM', 'Carbs'],
    'cgm_insulin_carbs': ['CGM', 'Insulin', 'Carbs'],
}

SEQ_LEN = 180
PRED_LEN = 12
MIN_SEQUENCE_LENGTH = SEQ_LEN + PRED_LEN  # 192
STEP_SIZE = 12


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

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
    """Hold out a fraction of *sequences* (not rows) for validation."""
    seq_ids = df['SequenceID'].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(seq_ids)
    n_val = max(1, int(len(seq_ids) * val_fraction))
    val_ids = set(seq_ids[:n_val])
    train_mask = ~df['SequenceID'].isin(val_ids)
    return df[train_mask], df[~train_mask]


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def make_lstm(input_dim: int, device: str) -> nn.Module:
    model = TrainableLSTMModel(
        input_dim=input_dim,
        hidden_dim=128,
        num_layers=2,
        pred_len=PRED_LEN,
        dropout=0.1,
    )
    return model.to(device)


def make_units(device: str) -> nn.Module:
    model = build_units_model(seq_len=SEQ_LEN, pred_len=PRED_LEN)
    return model.to(device)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    model_type: str
    feature_set: str
    epochs: int = 50
    batch_size: int = 64
    lr: float = 1e-3
    patience: int = 5
    device: str = 'cpu'
    checkpoint_dir: str = 'studies/feature_ablation/checkpoints'


def train_one(cfg: TrainConfig, train_df: pd.DataFrame, val_df: pd.DataFrame):
    """Train a single model variant and save the best checkpoint."""
    feature_cols = FEATURE_SETS[cfg.feature_set]
    input_dim = len(feature_cols)
    run_name = f'{cfg.model_type}_{cfg.feature_set}'

    print(f'\n{"="*70}')
    print(f'Training {run_name}  (features={feature_cols}, device={cfg.device})')
    print(f'{"="*70}')

    train_ds = GlucoseDataset(train_df, feature_cols)
    val_ds = GlucoseDataset(val_df, feature_cols)
    print(f'  Train windows: {len(train_ds)},  Val windows: {len(val_ds)}')

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=0, pin_memory=(cfg.device != 'cpu'))
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=0, pin_memory=(cfg.device != 'cpu'))

    if cfg.model_type == 'lstm':
        model = make_lstm(input_dim, cfg.device)
    elif cfg.model_type == 'units':
        model = make_units(cfg.device)
    else:
        raise ValueError(f'Unknown model type: {cfg.model_type}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    criterion = nn.MSELoss()

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.checkpoint_dir, f'{run_name}.pth')

    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = []
    t_start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        # --- train ---
        model.train()
        train_loss_sum, train_n = 0.0, 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(cfg.device)
            y_batch = y_batch.to(cfg.device)

            if cfg.model_type == 'lstm':
                pred = model(x_batch)
            else:
                # UniTS expects (B, T, V) and returns (B, pred_len, V)
                pred = model(
                    x_enc=x_batch, x_mark_enc=None,
                    task_id=0, task_name='long_term_forecast',
                )
                pred = pred[:, :, 0] if pred.ndim == 3 else pred

            loss = criterion(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item() * x_batch.size(0)
            train_n += x_batch.size(0)

        scheduler.step()
        train_loss = train_loss_sum / train_n

        # --- validate ---
        model.eval()
        val_loss_sum, val_n = 0.0, 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(cfg.device)
                y_batch = y_batch.to(cfg.device)

                if cfg.model_type == 'lstm':
                    pred = model(x_batch)
                else:
                    pred = model(
                        x_enc=x_batch, x_mark_enc=None,
                        task_id=0, task_name='long_term_forecast',
                    )
                    pred = pred[:, :, 0] if pred.ndim == 3 else pred

                val_loss_sum += criterion(pred, y_batch).item() * x_batch.size(0)
                val_n += x_batch.size(0)

        val_loss = val_loss_sum / max(val_n, 1)
        elapsed = time.time() - t_start

        history.append({
            'epoch': epoch, 'train_loss': train_loss,
            'val_loss': val_loss, 'lr': scheduler.get_last_lr()[0],
        })
        print(f'  Epoch {epoch:3d}/{cfg.epochs}  '
              f'train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  '
              f'lr={scheduler.get_last_lr()[0]:.2e}  [{elapsed:.0f}s]')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            ckpt = {
                'model_state_dict': model.state_dict(),
                'model_type': cfg.model_type,
                'feature_set': cfg.feature_set,
                'feature_cols': feature_cols,
                'input_dim': input_dim,
                'hidden_dim': 128,
                'num_layers': 2,
                'pred_len': PRED_LEN,
                'seq_len': SEQ_LEN,
                'best_val_loss': best_val_loss,
                'epoch': epoch,
            }
            torch.save(ckpt, ckpt_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                print(f'  Early stopping at epoch {epoch} '
                      f'(no improvement for {cfg.patience} epochs)')
                break

    total_time = time.time() - t_start
    print(f'  Best val_loss={best_val_loss:.4f}  Total time={total_time:.1f}s')
    print(f'  Checkpoint saved to {ckpt_path}')

    return {
        'run_name': run_name,
        'best_val_loss': best_val_loss,
        'total_time_s': total_time,
        'epochs_trained': len(history),
        'history': history,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option('--data_path', type=str, default='data/metabonet_train.parquet',
              help='Path to the preprocessed train parquet')
@click.option('--device', type=str, default='cpu',
              help='Device to train on (cpu, cuda, mps)')
@click.option('--epochs', type=int, default=50)
@click.option('--batch_size', type=int, default=64)
@click.option('--lr', type=float, default=1e-3)
@click.option('--patience', type=int, default=5)
@click.option('--models', type=str, default='lstm,units',
              help='Comma-separated model types to train')
@click.option('--feature_sets', type=str,
              default='cgm,cgm_insulin,cgm_carbs,cgm_insulin_carbs',
              help='Comma-separated feature sets to ablate')
def main(data_path, device, epochs, batch_size, lr, patience, models, feature_sets):
    """Train all model/feature-set combinations for the feature ablation study."""
    print(f'Loading data from {data_path} ...')
    df = pd.read_parquet(data_path)
    print(f'  Loaded {len(df)} rows, '
          f'{df["SequenceID"].nunique()} sequences, '
          f'{df["PtID"].nunique()} patients')

    train_df, val_df = split_train_val(df)
    print(f'  Train: {train_df["SequenceID"].nunique()} sequences, '
          f'Val: {val_df["SequenceID"].nunique()} sequences')

    model_types = [m.strip() for m in models.split(',')]
    feat_sets = [f.strip() for f in feature_sets.split(',')]

    summaries = []
    for model_type in model_types:
        for feat_set in feat_sets:
            cfg = TrainConfig(
                model_type=model_type,
                feature_set=feat_set,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                patience=patience,
                device=device,
            )
            result = train_one(cfg, train_df, val_df)
            summaries.append(result)

    # Print summary table
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
