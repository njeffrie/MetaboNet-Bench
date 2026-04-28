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


def _prediction_columns(df):
    metadata_cols = {
        "dataset",
        "DatasetName",
        "Dataset",
        "source_file",
        "patient_id",
        "timestamp",
        "label",
        "horizon",
        "subject_split_across_traintest",
    }
    return [c for c in df.columns if c not in metadata_cols]


def _compact_results_for_parquet(df):
    """Return a copy optimized for small parquet output."""
    compact = df.copy()

    for col in compact.columns:
        series = compact[col]
        if pd.api.types.is_bool_dtype(series):
            compact[col] = series.astype("bool")
        elif pd.api.types.is_integer_dtype(series):
            downcasted = pd.to_numeric(series, downcast="integer")
            if downcasted.dtype.itemsize < np.dtype("int32").itemsize:
                downcasted = downcasted.astype("int32")
            compact[col] = downcasted
        elif pd.api.types.is_float_dtype(series):
            compact[col] = pd.to_numeric(series, downcast="float")
        elif pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            compact[col] = series.astype("category")

    return compact


def _write_compact_parquet(df, output_path):
    compact = _compact_results_for_parquet(df)
    float_cols = [
        col for col in compact.columns
        if pd.api.types.is_float_dtype(compact[col])
    ]
    dictionary_cols = [
        col for col in compact.columns
        if col not in float_cols
    ]
    compact.to_parquet(
        output_path,
        index=False,
        engine="pyarrow",
        compression="zstd",
        compression_level=22,
        use_dictionary=dictionary_cols,
        use_byte_stream_split=float_cols,
        write_statistics=False,
        data_page_version="2.0",
    )


def load_results_with_split_column(results_dir="results", ds_path="data/metabonet_public_test.parquet", calculate_by_split=False):
    """Load *_results.parquet files into one wide table with a column per model."""
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

    combined = None
    results_dataset_col = None

    for path in tqdm(result_files, desc="Loading results", unit="file"):
        df = pd.read_parquet(path)
        if "prediction" not in df.columns:
            print(f"Skipping {path}: no prediction column")
            continue
        file_model_name = path.stem.replace("_results", "")

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
        else:
            df = df.drop(columns=["_pid_norm"])

        if "model" in df.columns and df["model"].nunique(dropna=True) > 1:
            model_frames = df.groupby("model", dropna=False)
        else:
            model_frames = [(file_model_name, df)]

        for model_name, model_part in model_frames:
            model_part = model_part.drop(columns=["model"], errors="ignore")
            metadata_cols = [c for c in model_part.columns if c != "prediction"]
            model_df = model_part[metadata_cols + ["prediction"]].rename(columns={"prediction": model_name})

            if combined is None:
                combined = model_df
            else:
                combined = combined.merge(model_df, on=metadata_cols, how="outer", validate="one_to_one")

    return combined


def _rmse(pred, label):
    return np.sqrt(np.mean((pred - label) ** 2))


def _mard(pred, label):
    return np.mean(np.abs(pred - label) / np.abs(label)) * 100


def _metrics_for_subset(subset_df, model_col, horizons):
    """Compute (rmse_list, mard_list, results_rows) for given horizons."""
    rmse_list, mard_list = [], []
    rows = []
    for h in horizons:
        h_df = subset_df[subset_df["horizon"] == h].dropna(subset=[model_col, "label"])
        if len(h_df) == 0:
            rmse_list.append(None)
            mard_list.append(None)
            continue
        pred, label = h_df[model_col].values, h_df["label"].values
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
    models = sorted(_prediction_columns(df))
    results = []

    split_names = ["overall", "known_patients", "new_patients"] if calculate_by_split else ["overall"]
    split_values = [None, True, False] if calculate_by_split else [None]

    for model in tqdm(models, desc="Computing metrics", unit="model"):
        for split_name, split_val in zip(split_names, split_values):
            subset = df if split_val is None else df[df["subject_split_across_traintest"] == split_val]
            if len(subset) == 0:
                continue
            _, _, rows = _metrics_for_subset(subset, model, expected_horizons)
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

def plot_dts_error_grid(df, model_name, horizon_min, subset_size = 2000):
    """Plot DTS error grid for a model-horizon combination"""
    dts_grid_path='data/dts_grid.png'
    df = df[df["horizon"] == horizon_min // 5].dropna(subset=[model_name, "label"])
    labels, predictions = df["label"].values, df[model_name].values
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
    for model in tqdm(_prediction_columns(df), desc="Generating plots", unit="model"):
        plot_dts_error_grid(df, model, 30, subset_size=2000)

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
        _write_compact_parquet(df, output_path)
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"Saved {len(df)} rows to {output_path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
