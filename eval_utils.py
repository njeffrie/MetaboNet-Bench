from pathlib import Path

import numpy as np
import pandas as pd
import torch


METADATA_COLS = {
    'dataset',
    'DatasetName',
    'Dataset',
    'source_file',
    'patient_id',
    'timestamp',
    'label',
    'horizon',
    'subject_split_across_traintest',
}


def prediction_columns(df):
    return [c for c in df.columns if c not in METADATA_COLS and c not in {'model', 'prediction'}]


def rmse(pred, label):
    return np.sqrt(np.mean((pred - label) ** 2))


def mard(pred, label):
    return np.mean(np.abs(pred - label) / np.abs(label)) * 100


def load_results(input_path):
    input_path = Path(input_path)
    if input_path.is_dir():
        frames = []
        for path in sorted(input_path.glob('*_results.parquet')):
            df = pd.read_parquet(path)
            if 'model' not in df.columns:
                df['model'] = path.stem.replace('_results', '')
            frames.append(df)
        if not frames:
            raise ValueError(f'No *_results.parquet files found in {input_path}')
        return pd.concat(frames, ignore_index=True)
    return pd.read_parquet(input_path)


def iter_model_frames(df):
    if {'model', 'prediction'} <= set(df.columns):
        for model, model_df in df.groupby('model', sort=True):
            yield model, model_df, 'prediction'
    else:
        cols = prediction_columns(df)
        if not cols:
            raise ValueError('Expected long-form model/prediction columns or wide model columns.')
        for model in sorted(cols):
            yield model, df, model


def resolve_device(device):
    if device == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('CUDA was requested, but torch.cuda.is_available() is false')
    return device


def bootstrap_mean_ci(values, n_bootstrap, ci, rng, seed, device, batch_size,
                      transform=lambda x: x):
    values = np.asarray(values, dtype=np.float64)
    samples = np.empty(n_bootstrap, dtype=np.float64)
    if device.startswith('cuda'):
        values_t = torch.as_tensor(values, dtype=torch.float32, device=device)
        gen = torch.Generator(device=device).manual_seed(seed)
        for start in range(0, n_bootstrap, batch_size):
            end = min(start + batch_size, n_bootstrap)
            idx = torch.randint(0, len(values), (end - start, len(values)),
                                device=device, generator=gen)
            samples[start:end] = values_t[idx].mean(dim=1).detach().cpu().numpy()
    else:
        for start in range(0, n_bootstrap, batch_size):
            end = min(start + batch_size, n_bootstrap)
            idx = rng.integers(0, len(values), size=(end - start, len(values)))
            samples[start:end] = values[idx].mean(axis=1)
    alpha = (100 - ci) / 2
    estimate = transform(values.mean())
    lower, upper = transform(np.percentile(samples, [alpha, 100 - alpha]))
    return float(estimate), float(lower), float(upper)


def dts_zones(labels, predictions, dts_grid_path, extent=(-62, 835, -47, 646)):
    import matplotlib.pyplot as plt

    zone_rgb = {
        'A': np.array([0.5647059, 0.72156864, 0.5019608], dtype=np.float32),
        'B': np.array([1.0039216, 1.0039216, 0.59607846], dtype=np.float32),
        'C': np.array([0.972549, 0.8156863, 0.5647059], dtype=np.float32),
        'D': np.array([0.9411765, 0.53333336, 0.5019608], dtype=np.float32),
        'E': np.array([0.78431374, 0.53333336, 0.65882355], dtype=np.float32),
    }
    r, p = map(lambda x: np.asarray(x).ravel(), (labels, predictions))
    img = plt.imread(dts_grid_path).astype(np.float32)
    h, w = img.shape[:2]
    xmin, xmax, ymin, ymax = extent
    x = np.round((r - xmin) / (xmax - xmin) * (w - 1)).astype(int)
    y = np.round((ymax - p) / (ymax - ymin) * (h - 1)).astype(int)
    keys = np.array(list(zone_rgb), dtype='<U1')
    colors = np.stack([zone_rgb[k] for k in keys])
    return keys[np.argmin(((img[y, x, :3][:, None] - colors) ** 2).sum(-1), axis=1)]
