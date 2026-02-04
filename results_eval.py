import re
from pathlib import Path

import click
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import os


def _find_column(df, candidates):
    """Return first column in df that is in candidates, else None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _normalize_patient_id(x):
    """Normalize patient ID for matching (same convention as benchmark)."""
    if pd.isna(x):
        return None
    if isinstance(x, (int, float)):
        return int(x)
    try:
        return int(re.findall(r"\d+", str(x))[-1])
    except (IndexError, ValueError, TypeError):
        return x


def _build_split_lookup(test_df, patient_col, dataset_col, split_col="subject_split_across_traintest"):
    """Build (patient_id_norm, dataset) -> subject_split_across_traintest lookup."""
    test_df = test_df.copy()
    test_df["_pid_norm"] = test_df[patient_col].apply(_normalize_patient_id)
    if dataset_col:
        lookup = test_df.groupby(["_pid_norm", dataset_col], dropna=False)[split_col].first()
        return {(pid, ds): val for (pid, ds), val in lookup.items()}
    lookup = test_df.groupby("_pid_norm", dropna=False)[split_col].first()
    return {(pid, None): val for pid, val in lookup.items()}


def load_results_with_split_column(results_dir="results", ds_path="data/metabonet_public_test.parquet", calculate_by_split=False):
    """Load *_results.parquet files and merge subject_split_across_traintest from test set."""
    results_dir = Path(results_dir)
    ds_path = Path(ds_path)
    if calculate_by_split:
        test_df = pd.read_parquet(ds_path)
        if "subject_split_across_traintest" not in test_df.columns:
            print(f"ERROR: 'subject_split_across_traintest' not in {ds_path}")
            return None

        patient_col = _find_column(test_df, ["PtID", "patient_id", "id", "PtID_normalized"])
        dataset_col = _find_column(test_df, ["DatasetName", "dataset", "source_file", "Dataset"])
        if not patient_col:
            print(f"ERROR: No patient ID column in test set. Columns: {list(test_df.columns)}")
            return None

        lookup = _build_split_lookup(test_df, patient_col, dataset_col)
    result_files = sorted(results_dir.glob("*_results.parquet"))
    if not result_files:
        print(f"No *_results.parquet found in {results_dir}")
        return None

    frames = []
    results_dataset_col = None

    for path in tqdm(result_files, desc="Loading results", unit="file"):
        df = pd.read_parquet(path)
        model_name = path.stem.replace("_results", "")
        df["model"] = model_name

        if results_dataset_col is None:
            results_dataset_col = _find_column(df, ["dataset", "DatasetName", "Dataset", "source_file"])

        df["_pid_norm"] = df["patient_id"].apply(_normalize_patient_id)
        if calculate_by_split:
            df["_ds"] = df.get(results_dataset_col, None) if results_dataset_col else None
            df["subject_split_across_traintest"] = df.apply(
                lambda r: lookup.get((r["_pid_norm"], r["_ds"] if dataset_col and results_dataset_col else None)),
                axis=1,
            )
            df = df.drop(columns=["_pid_norm", "_ds"])
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def _rmse(pred, label):
    return np.sqrt(np.mean((pred - label) ** 2))


def _mard(pred, label):
    return np.mean(np.abs(pred - label) / np.abs(label)) * 100


def _metrics_for_subset(subset_df, horizons):
    """Compute (rmse_list, mard_list, results_rows) for given horizons."""
    rmse_list, mard_list = [], []
    rows = []
    for h in horizons:
        h_df = subset_df[subset_df["horizon"] == h]
        if len(h_df) == 0:
            rmse_list.append(None)
            mard_list.append(None)
            continue
        pred, label = h_df["prediction"].values, h_df["label"].values
        rmse, mard = _rmse(pred, label), _mard(pred, label)
        rmse_list.append(rmse)
        mard_list.append(mard)
        rows.append({"horizon_minutes": h * 5, "horizon_step": h, "rmse": rmse, "mard": mard, "n_predictions": len(h_df)})
    return rmse_list, mard_list, rows


def compute_metrics(df, calculate_by_split):
    """Compute RMSE and MARD by horizon for each model, overall and by subject split."""
    if calculate_by_split:
        df["subject_split_across_traintest"] = df["subject_split_across_traintest"].fillna(False).astype(bool)

    expected_horizons = list(range(1, 13))
    models = sorted(df["model"].unique())
    results = []

    split_names = ["overall", "known_patients", "new_patients"] if calculate_by_split else ["overall"]
    split_values = [None, True, False] if calculate_by_split else [None]

    for model in tqdm(models, desc="Computing metrics", unit="model"):
        model_df = df[df["model"] == model]
        for split_name, split_val in zip(split_names, split_values):
            subset = model_df if split_val is None else model_df[model_df["subject_split_across_traintest"] == split_val]
            if len(subset) == 0:
                continue
            _, _, rows = _metrics_for_subset(subset, expected_horizons)
            for r in rows:
                r["model"] = model
                r["split_type"] = split_name
                results.append(r)

    results_df = pd.DataFrame(results)
    print(f"\n{'='*80}\nRESULTS\n{'='*80}")
    for split_type in split_names:
        part = results_df[results_df["split_type"] == split_type]
        if part.empty:
            continue
        title = split_type.upper().replace("_", "-")
        print(f"\n{title}:")
        print("RMSE by Model and Horizon:")
        print(part.pivot_table(index="model", columns="horizon_minutes", values="rmse", aggfunc="first").round(2).to_string())
        print("MARD (%) by Model and Horizon:")
        print(part.pivot_table(index="model", columns="horizon_minutes", values="mard", aggfunc="first").round(2).to_string())
    return results_df

def dts_zone_counts(labels, predictions, dts_grid_path, extent=(-62, 835, -47, 646)):
    """Calculate DTS zone counts for predictions and labels"""
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
    
    xi = np.round((r - xmin) / (xmax - xmin) * (w - 1)).astype(int)
    yi = np.round((ymax - p) / (ymax - ymin) * (h - 1)).astype(int)
    pix = img[yi, xi, :3]
    
    keys = np.array(list(zone_rgb), dtype='<U1')
    cols = np.stack([zone_rgb[k] for k in keys], axis=0)
    z = keys[np.argmin(((pix[:, None] - cols)**2).sum(-1), axis=1)]
    
    return {k: int((z == k).sum()) for k in 'ABCDE'}

def plot_dts_error_grid(df, horizon_min, subset_size = 2000):
    """Plot DTS error grid for a model-horizon combination"""
    dts_grid_path='data/dts_grid.png'
    df = df[df["horizon"] == horizon_min // 5]
    labels, predictions = df["label"].values, df["prediction"].values
    model_name = df["model"].values[0]
    zone_counts = dts_zone_counts(labels, predictions, dts_grid_path)
    zone_pct = {k: round(v / len(labels) * 100, 2) for k, v in zone_counts.items()}
    if subset_size is not None and subset_size < len(labels):
        indices = np.random.choice(len(labels), subset_size, replace=False)
        labels = labels[indices]
        predictions = predictions[indices]
    
    plt.figure(figsize=(10, 7.5), dpi=150)
    plt.imshow(plt.imread(dts_grid_path), extent=(-62, 835, -47, 646), origin='upper', aspect='auto')
    plt.scatter(labels, predictions, s=6, facecolors='white', edgecolors='black', linewidths=0.4)
    plt.axis('off')
    plt.text(0.05, -0.01, ('Zone A: $\\bf{{{A}\\%}}$, Zone B: $\\bf{{{B}\\%}}$, Zone C: $\\bf{{{C}\\%}}$, Zone D: $\\bf{{{D}\\%}}$, Zone E: $\\bf{{{E}\\%}}$').format(**zone_pct),
             transform=plt.gca().transAxes, fontsize=12, va='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/{model_name}_{horizon_min}min.png', dpi=150, bbox_inches='tight')
    plt.close()
    return zone_pct

def create_plots(df):
    for model in tqdm(df["model"].unique(), desc="Generating plots", unit="model"):
        model_df = df[df["model"] == model]
        plot_dts_error_grid(model_df, 30, subset_size=2000)

@click.command()
@click.option("--results_dir", type=str, default="results", help="Directory with *_results.parquet files")
@click.option("--ds_path", type=str, default="data/metabonet_public_test.parquet", help="Test parquet with subject_split_across_traintest")
@click.option("--output_path", type=str, default=None, help="Save combined results to this path")
@click.option("--generate_plots", is_flag=True, help="Generate plots for the results")
@click.option("--calculate_by_split", is_flag=True, help="Calculate metrics by subject split")
def main(results_dir="results", ds_path="data/metabonet_public_test.parquet", output_path=None, calculate_by_split=False, generate_plots=False):
    df = load_results_with_split_column(results_dir=results_dir, ds_path=ds_path, calculate_by_split=calculate_by_split)
    if df is None:
        return
    print(calculate_by_split)
    compute_metrics(df, calculate_by_split=calculate_by_split)
    if generate_plots:
        create_plots(df)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False, engine="pyarrow")
        print(f"Saved {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()
