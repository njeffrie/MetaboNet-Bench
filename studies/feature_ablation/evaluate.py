"""
Feature ablation evaluation script.

Loads trained checkpoints, runs inference on the MetaboNet test split using
the same sliding-window approach as benchmark.py, and generates a comparison
table of overall RMSE and MARD.

Usage:
    python -m studies.feature_ablation.evaluate \
        --test_data data/metabonet_test.parquet --device cuda
"""

import os
import re
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
from tqdm import tqdm

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.lstm import LSTM
from models.UniTS import UniTS
from models.gluforecast import GluForecast

SEQ_LEN = 180
PRED_LEN = 12
MIN_SEQUENCE_LENGTH = SEQ_LEN + PRED_LEN
STEP_SIZE = 12


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: str):
    """Load a trained model from its checkpoint, auto-detecting type."""
    import torch
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model_type = ckpt['model_type']
    feature_set = ckpt['feature_set']

    if model_type == 'lstm':
        return LSTM(checkpoint_path, feature_set=feature_set, device=device)
    elif model_type == 'units':
        return UniTS(checkpoint_path, feature_set=feature_set, device=device)
    elif model_type == 'gluforecast':
        return GluForecast(
            checkpoint_path, feature_set=feature_set, device=device)
    else:
        raise ValueError(f'Unknown model_type in checkpoint: {model_type}')


# ---------------------------------------------------------------------------
# Inference (mirrors benchmark.py)
# ---------------------------------------------------------------------------

def run_batch(model_runner, input_batch):
    input_array = np.stack(input_batch, axis=0)
    ts, cgm, insulin, carbs = np.split(input_array, 4, axis=1)
    preds = model_runner.predict(
        ts.squeeze(1), cgm.squeeze(1),
        insulin.squeeze(1), carbs.squeeze(1),
    )
    return preds


def extract_patient_id(patient_id):
    return int(re.findall(r'\d+', str(patient_id))[-1])


def run_inference(model_runner, df: pd.DataFrame, run_name: str,
                  batch_size: int = 64):
    """Run sliding-window inference identical to benchmark.py."""
    all_results = []

    for ds_name, dataset_group in df.groupby('DatasetName'):
        input_batch = []
        batch_metadata = []
        batch_labels = []
        batch_pred_timestamps = []

        for patient_id, patient_data in tqdm(
                dataset_group.groupby('PtID'),
                desc=f'{run_name} / {ds_name}'):
            patient_id_num = extract_patient_id(patient_id)

            for _, sequence_data in patient_data.groupby('SequenceID'):
                seq_len = len(sequence_data)
                if seq_len < MIN_SEQUENCE_LENGTH:
                    continue

                cgm_values = sequence_data['CGM'].values
                timestamps = sequence_data['DataDtTm'].values.astype(np.int64)
                insulin_values = sequence_data['Insulin'].values
                carbs_values = sequence_data['Carbs'].values

                for i in range(0, seq_len - MIN_SEQUENCE_LENGTH + 1, STEP_SIZE):
                    model_input_cgm = cgm_values[i:i + SEQ_LEN]
                    label = cgm_values[i + SEQ_LEN:i + MIN_SEQUENCE_LENGTH]
                    ts_window = timestamps[i:i + SEQ_LEN]
                    insulin_window = insulin_values[i:i + SEQ_LEN]
                    carbs_window = carbs_values[i:i + SEQ_LEN]
                    pred_timestamps = timestamps[i + SEQ_LEN:i + MIN_SEQUENCE_LENGTH]

                    input_batch.append(np.stack([
                        ts_window, model_input_cgm,
                        insulin_window, carbs_window,
                    ], axis=0))
                    batch_metadata.append({
                        'dataset': ds_name,
                        'patient_id': patient_id_num,
                    })
                    batch_labels.append(label)
                    batch_pred_timestamps.append(pred_timestamps)

                    if len(input_batch) == batch_size:
                        preds = run_batch(model_runner, input_batch)
                        preds = np.clip(preds, 40, 600)
                        for pred, meta, lbl, pts in zip(
                                preds, batch_metadata, batch_labels,
                                batch_pred_timestamps):
                            for step in range(PRED_LEN):
                                all_results.append({
                                    'model': run_name,
                                    'dataset': meta['dataset'],
                                    'patient_id': meta['patient_id'],
                                    'timestamp': pts[step],
                                    'prediction': pred[step],
                                    'label': lbl[step],
                                    'horizon': step + 1,
                                })
                        input_batch, batch_metadata = [], []
                        batch_labels, batch_pred_timestamps = [], []

        if input_batch:
            preds = run_batch(model_runner, input_batch)
            preds = np.clip(preds, 40, 600)
            for pred, meta, lbl, pts in zip(
                    preds, batch_metadata, batch_labels,
                    batch_pred_timestamps):
                for step in range(PRED_LEN):
                    all_results.append({
                        'model': run_name,
                        'dataset': meta['dataset'],
                        'patient_id': meta['patient_id'],
                        'timestamp': pts[step],
                        'prediction': pred[step],
                        'label': lbl[step],
                        'horizon': step + 1,
                    })

    return pd.DataFrame(all_results)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(results_df: pd.DataFrame) -> pd.DataFrame:
    """Compute RMSE and MARD per model at each horizon and overall."""
    rows = []
    for model_name, mdf in results_df.groupby('model'):
        # Per-horizon
        for h in range(1, PRED_LEN + 1):
            hdf = mdf[mdf['horizon'] == h]
            if len(hdf) == 0:
                continue
            pred, label = hdf['prediction'].values, hdf['label'].values
            rmse = np.sqrt(np.mean((pred - label) ** 2))
            mard = np.mean(np.abs(pred - label) / np.abs(label)) * 100
            rows.append({
                'model': model_name,
                'horizon_minutes': h * 5,
                'rmse': rmse,
                'mard': mard,
                'n': len(hdf),
            })
        # Overall (all horizons)
        pred, label = mdf['prediction'].values, mdf['label'].values
        rmse = np.sqrt(np.mean((pred - label) ** 2))
        mard = np.mean(np.abs(pred - label) / np.abs(label)) * 100
        rows.append({
            'model': model_name,
            'horizon_minutes': 'overall',
            'rmse': rmse,
            'mard': mard,
            'n': len(mdf),
        })
    return pd.DataFrame(rows)


def print_comparison_table(metrics_df: pd.DataFrame):
    """Print RMSE and MARD tables to stdout."""
    per_horizon = metrics_df[metrics_df['horizon_minutes'] != 'overall'].copy()
    per_horizon['horizon_minutes'] = per_horizon['horizon_minutes'].astype(int)
    overall = metrics_df[metrics_df['horizon_minutes'] == 'overall']

    print(f'\n{"="*80}')
    print('RMSE by Model and Horizon (minutes)')
    print(f'{"="*80}')
    rmse_pivot = per_horizon.pivot_table(
        index='model', columns='horizon_minutes',
        values='rmse', aggfunc='first',
    ).round(2)
    print(rmse_pivot.to_string())

    print(f'\n{"="*80}')
    print('MARD (%) by Model and Horizon (minutes)')
    print(f'{"="*80}')
    mard_pivot = per_horizon.pivot_table(
        index='model', columns='horizon_minutes',
        values='mard', aggfunc='first',
    ).round(2)
    print(mard_pivot.to_string())

    print(f'\n{"="*80}')
    print('Overall Accuracy')
    print(f'{"="*80}')
    overall_table = overall[['model', 'rmse', 'mard', 'n']].copy()
    overall_table = overall_table.sort_values('rmse')
    overall_table['rmse'] = overall_table['rmse'].round(2)
    overall_table['mard'] = overall_table['mard'].round(2)
    print(overall_table.to_string(index=False))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option('--test_data', type=str, default='data/metabonet_test.parquet',
              help='Path to preprocessed test parquet')
@click.option('--checkpoint_dir', type=str,
              default='studies/feature_ablation/checkpoints',
              help='Directory containing trained checkpoints')
@click.option('--results_dir', type=str,
              default='studies/feature_ablation/results',
              help='Directory to save results')
@click.option('--device', type=str, default='cpu')
@click.option('--batch_size', type=int, default=64)
def main(test_data, checkpoint_dir, results_dir, device, batch_size):
    """Evaluate all trained ablation models and generate comparison tables."""
    checkpoint_dir = Path(checkpoint_dir)
    ckpt_files = sorted(checkpoint_dir.glob('*.pth'))
    if not ckpt_files:
        print(f'No checkpoints found in {checkpoint_dir}')
        return

    print(f'Found {len(ckpt_files)} checkpoints in {checkpoint_dir}')
    print(f'Loading test data from {test_data} ...')
    test_df = pd.read_parquet(test_data)
    print(f'  {len(test_df)} rows, '
          f'{test_df["SequenceID"].nunique()} sequences')

    all_results = []
    for ckpt_path in ckpt_files:
        run_name = ckpt_path.stem
        print(f'\nEvaluating {run_name} ...')
        model_runner = load_model(str(ckpt_path), device)
        result_df = run_inference(model_runner, test_df, run_name,
                                  batch_size=batch_size)
        all_results.append(result_df)
        print(f'  {len(result_df)} prediction rows')

    combined = pd.concat(all_results, ignore_index=True)

    os.makedirs(results_dir, exist_ok=True)
    parquet_path = os.path.join(results_dir, 'ablation_results.parquet')
    combined.to_parquet(parquet_path, index=False, engine='pyarrow')
    print(f'\nSaved combined results to {parquet_path}')

    metrics_df = compute_metrics(combined)

    csv_path = os.path.join(results_dir, 'ablation_metrics.csv')
    metrics_df.to_csv(csv_path, index=False)
    print(f'Saved metrics to {csv_path}')

    print_comparison_table(metrics_df)


if __name__ == '__main__':
    main()
