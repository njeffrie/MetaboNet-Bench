#!/usr/bin/env python3
"""
Combine model result files into a single parquet file merged with user demographics
from the MetaboNet public test split. The MetaboNet parquet location is resolved from
``--metabonet-test`` (CLI), then ``$METABONET_TEST_PARQUET`` (local path or s3:// URI).

Supports two input formats (auto-detected):

  npy (legacy): results_dir contains model subdirectories, each with <dataset>.npy files.
    Each .npy has shape (N, 4, 12):
      arr[:,0,:] timestamps (ns, prediction time; shifted +60min to align with label_t0)
      arr[:,1,:] user ID (repeated)
      arr[:,2,:] predictions (12 timesteps)
      arr[:,3,:] labels (12 timesteps)

  parquet (new): results_dir contains flat <model>_results.parquet files with columns:
      model, dataset, patient_id, timestamp (ns, = label time per horizon),
      prediction, label, horizon (1-12)

Output schema:
  user_id, timestamp_t0, dataset, model,
  label_t0..label_t11, pred_t0..pred_t11,
  Demographics from MetaboNet (age, gender, weight, height, ...)
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

METABONET_TEST_ENV = "METABONET_TEST_PARQUET"


def resolve_metabonet_test_path(local_path: Path | None = None) -> str:
    """Return a path/URI to the MetaboNet public test parquet.

    Resolution order:
      1. ``local_path`` (CLI arg) if given.
      2. ``$METABONET_TEST_PARQUET`` env var (local path or s3:// URI).
      3. Fail with a clear error.
    """
    if local_path is not None:
        if not Path(local_path).exists():
            raise FileNotFoundError(
                f"--metabonet-test path does not exist: {local_path}"
            )
        return str(local_path)

    env_value = os.environ.get(METABONET_TEST_ENV)
    if env_value:
        if not env_value.startswith("s3://") and not Path(env_value).exists():
            raise FileNotFoundError(
                f"${METABONET_TEST_ENV} points to a non-existent path: {env_value}"
            )
        return env_value

    raise RuntimeError(
        f"MetaboNet public test parquet location is not configured. "
        f"Set ${METABONET_TEST_ENV} (local path or s3:// URI) or pass "
        f"--metabonet-test PATH on the CLI."
    )

DEMOGRAPHIC_COLUMNS = [
    'source_file',
    'id',
    'age',
    'gender',
    'weight',
    'height',
    'age_of_diagnosis',
    'ethnicity',
    'insulin_delivery_modality',
    'cgm_device',
    'subject_split_across_traintest',
]

CGM_COLUMNS = [
    'source_file',
    'id',
    'CGM',
]

TIMESTAMP_MERGE_COLUMNS = [
    'source_file',
    'id',
    'date',
    'insulin',
    'carbs',
    'CGM',
]


# ---------------------------------------------------------------------------
# npy loading (legacy)
# ---------------------------------------------------------------------------

def load_npy_file(filepath: Path) -> np.ndarray:
    return np.load(filepath)


def process_npy_file(arr: np.ndarray, model: str, dataset: str) -> pd.DataFrame:
    """Convert (N, 4, 12) numpy array to wide-format DataFrame."""
    n_samples = arr.shape[0]
    timestamps = arr[:, 0, 0]
    # Raw timestamp is prediction time; labels start 60min later
    timestamp_t0 = pd.to_datetime(timestamps, unit='ns') + pd.Timedelta(minutes=60)
    user_ids = arr[:, 1, 0].astype(int)
    predictions = arr[:, 2, :]
    labels = arr[:, 3, :]

    data = {
        'user_id': user_ids,
        'timestamp_t0': timestamp_t0,
        'dataset': [dataset] * n_samples,
        'model': [model] * n_samples,
    }
    for t in range(12):
        data[f'label_t{t}'] = labels[:, t]
    for t in range(12):
        data[f'pred_t{t}'] = predictions[:, t]

    return pd.DataFrame(data)


def load_from_npy_dir(results_dir: Path) -> pd.DataFrame:
    """Walk results_dir/<model>/<dataset>.npy and return combined wide DataFrame."""
    all_dfs = []
    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith('.'):
            continue
        model_name = model_dir.name
        print(f"Processing model: {model_name}")
        for npy_file in sorted(model_dir.glob("*.npy")):
            dataset_name = npy_file.stem
            arr = load_npy_file(npy_file)
            df = process_npy_file(arr, model_name, dataset_name)
            all_dfs.append(df)
            print(f"  - {dataset_name}: {len(df):,} samples")
    return pd.concat(all_dfs, ignore_index=True)


# ---------------------------------------------------------------------------
# parquet loading (new)
# ---------------------------------------------------------------------------

def process_parquet_file(filepath: Path) -> pd.DataFrame:
    """
    Load a <model>_results.parquet file and pivot from long to wide format.

    timestamp encodes the label time at each horizon, so
    timestamp_t0 = timestamp - (horizon-1)*5min is constant within a prediction window.
    """
    df = pd.read_parquet(filepath)

    df['timestamp_t0'] = (
        pd.to_datetime(df['timestamp'], unit='ns')
        - (df['horizon'] - 1) * pd.Timedelta(minutes=5)
    )
    df = df.rename(columns={'patient_id': 'user_id'})

    label_wide = df.pivot_table(
        index=['model', 'dataset', 'user_id', 'timestamp_t0'],
        columns='horizon',
        values='label',
        aggfunc='first',
    )
    label_wide.columns = [f'label_t{h - 1}' for h in label_wide.columns]

    pred_wide = df.pivot_table(
        index=['model', 'dataset', 'user_id', 'timestamp_t0'],
        columns='horizon',
        values='prediction',
        aggfunc='first',
    )
    pred_wide.columns = [f'pred_t{h - 1}' for h in pred_wide.columns]

    return label_wide.join(pred_wide).reset_index()


def load_from_parquet_dir(results_dir: Path) -> pd.DataFrame:
    """Read flat *_results.parquet files and return combined wide DataFrame."""
    all_dfs = []
    for pq_file in sorted(results_dir.glob("*_results.parquet")):
        print(f"Processing: {pq_file.name}")
        df = process_parquet_file(pq_file)
        all_dfs.append(df)
        print(f"  - {df['model'].iloc[0]}: {len(df):,} windows")
    return pd.concat(all_dfs, ignore_index=True)


# ---------------------------------------------------------------------------
# S3 helpers (shared)
# ---------------------------------------------------------------------------

def load_demographics(metabonet_path: str) -> pd.DataFrame:
    print(f"Loading demographics from {metabonet_path}...")
    dataset = ds.dataset(metabonet_path, format='parquet')
    df = dataset.to_table(columns=DEMOGRAPHIC_COLUMNS).to_pandas()
    print(f"  Loaded {len(df):,} rows")
    demographics_df = df.groupby(['source_file', 'id']).first().reset_index()
    print(f"  Found {len(demographics_df):,} unique users with demographics")
    return demographics_df


def load_cgm_stats(metabonet_path: str) -> pd.DataFrame:
    print(f"Loading CGM data from {metabonet_path}...")
    dataset = ds.dataset(metabonet_path, format='parquet')
    df = dataset.to_table(columns=CGM_COLUMNS).to_pandas()
    print(f"  Loaded {len(df):,} CGM rows")
    cgm_stats = df.groupby(['source_file', 'id']).agg(
        cgm_mean=('CGM', 'mean'),
        cgm_std=('CGM', 'std'),
    ).reset_index()
    print(f"  Computed CGM stats for {len(cgm_stats):,} unique users")
    return cgm_stats


def load_timestamp_data(metabonet_path: str) -> pd.DataFrame:
    print(f"Loading timestamp data (insulin/carbs/CGM) from {metabonet_path}...")
    dataset = ds.dataset(metabonet_path, format='parquet')
    df = dataset.to_table(columns=TIMESTAMP_MERGE_COLUMNS).to_pandas()
    print(f"  Loaded {len(df):,} rows with timestamp data")
    return df


# ---------------------------------------------------------------------------
# Main combine logic
# ---------------------------------------------------------------------------

def enrich_with_demographics(combined_df: pd.DataFrame, metabonet_path: str) -> pd.DataFrame:
    """Merge MetaboNet demographics, CGM stats, and historical insulin/carbs into a wide-format df."""
    demographics_df = load_demographics(metabonet_path)
    cgm_stats_df = load_cgm_stats(metabonet_path)

    demographics_df = demographics_df.merge(
        cgm_stats_df, on=['source_file', 'id'], how='left'
    )

    combined_df['user_id_str'] = combined_df['user_id'].astype(str)
    combined_df = combined_df.merge(
        demographics_df,
        left_on=['dataset', 'user_id_str'],
        right_on=['source_file', 'id'],
        how='left',
    )
    combined_df = combined_df.drop(columns=['source_file', 'id', 'user_id_str'])

    demo_cols = [c for c in DEMOGRAPHIC_COLUMNS if c not in ['source_file', 'id']]
    demo_cols += ['cgm_mean', 'cgm_std']
    for col in demo_cols:
        non_null = combined_df[col].notna().sum()
        pct = 100 * non_null / len(combined_df)
        print(f"  {col}: {non_null:,} non-null ({pct:.1f}%)")

    if 'timestamp_t0' in combined_df.columns:
        timestamp_df = load_timestamp_data(metabonet_path)
        combined_df['user_id_str'] = combined_df['user_id'].astype(str)
        combined_df = combined_df.merge(
            timestamp_df,
            left_on=['dataset', 'user_id_str', 'timestamp_t0'],
            right_on=['source_file', 'id', 'date'],
            how='left',
        )
        combined_df = combined_df.rename(columns={'CGM': 'cgm_at_t0'})
        combined_df = combined_df.drop(columns=['source_file', 'id', 'date', 'user_id_str'])

        print("\n  Merging historical insulin/carbs values...")
        for i in range(1, 6):
            offset_minutes = i * 5
            combined_df[f'timestamp_tminus_{i}'] = combined_df['timestamp_t0'] - pd.Timedelta(minutes=offset_minutes)
            combined_df['user_id_str'] = combined_df['user_id'].astype(str)

            hist_df = combined_df.merge(
                timestamp_df[['source_file', 'id', 'date', 'insulin', 'carbs']],
                left_on=['dataset', 'user_id_str', f'timestamp_tminus_{i}'],
                right_on=['source_file', 'id', 'date'],
                how='left',
                suffixes=('', f'_tminus_{i}'),
            )
            combined_df[f'insulin_tminus_{i}'] = hist_df[f'insulin_tminus_{i}'] if f'insulin_tminus_{i}' in hist_df.columns else hist_df['insulin']
            combined_df[f'carbs_tminus_{i}'] = hist_df[f'carbs_tminus_{i}'] if f'carbs_tminus_{i}' in hist_df.columns else hist_df['carbs']
            combined_df = combined_df.drop(columns=[f'timestamp_tminus_{i}', 'user_id_str'])
            print(f"    t-{i} ({offset_minutes}min): insulin={combined_df[f'insulin_tminus_{i}'].notna().sum():,}, carbs={combined_df[f'carbs_tminus_{i}'].notna().sum():,}")

        print("\n--- Timestamp merge statistics ---")
        matched_rows = combined_df['cgm_at_t0'].notna().sum()
        total_rows = len(combined_df)
        print(f"  Rows matched by timestamp: {matched_rows:,} / {total_rows:,} ({100*matched_rows/total_rows:.1f}%)")

        print("\n--- Verifying timestamp alignment ---")
        valid_mask = combined_df['cgm_at_t0'].notna() & combined_df['label_t0'].notna()
        if valid_mask.sum() > 0:
            cgm_t0 = combined_df.loc[valid_mask, 'cgm_at_t0']
            label_t0 = combined_df.loc[valid_mask, 'label_t0']
            exact_matches = (cgm_t0 == label_t0).sum()
            close_matches = (abs(cgm_t0 - label_t0) < 0.1).sum()
            print(f"  Close matches (|diff| < 0.1): {close_matches:,} / {valid_mask.sum():,} ({100*close_matches/valid_mask.sum():.1f}%)")
            print(f"  Correlation: {cgm_t0.corr(label_t0):.4f}")
            if close_matches / valid_mask.sum() > 0.85:
                print("  ✓ Timestamp alignment verified")
            elif close_matches / valid_mask.sum() > 0.5:
                print("  ~ Timestamp alignment looks reasonable")
            else:
                print("  ✗ WARNING: Timestamp alignment may be incorrect!")
        else:
            print("  ✗ No rows with both cgm_at_t0 and label_t0")

    return combined_df


def load_one_parquet(filepath: Path) -> pd.DataFrame:
    """
    Load a single per-model parquet file. Auto-detects:
      - long format (model, dataset, patient_id, timestamp, prediction, label, horizon)
        → pivot to wide
      - wide format (model, dataset, user_id|patient_id, timestamp_t0, label_t*, pred_t*)
        → use as-is (rename patient_id → user_id if needed)
    """
    df = pd.read_parquet(filepath)
    if 'horizon' in df.columns and 'prediction' in df.columns and 'label' in df.columns:
        return process_parquet_file(filepath)
    if 'patient_id' in df.columns and 'user_id' not in df.columns:
        df = df.rename(columns={'patient_id': 'user_id'})
    return df


def combine_results(
    results_dir: Path,
    output_path: Path,
    merge_demographics: bool = True,
    metabonet_path: str | None = None,
) -> pd.DataFrame:
    # Auto-detect format
    has_parquets = bool(list(results_dir.glob("*_results.parquet")))
    has_npy_subdirs = any(
        p.is_dir() and not p.name.startswith('.')
        for p in results_dir.iterdir()
    )

    if has_parquets:
        print("Detected parquet format")
        combined_df = load_from_parquet_dir(results_dir)
    elif has_npy_subdirs:
        print("Detected npy format")
        combined_df = load_from_npy_dir(results_dir)
    else:
        raise ValueError(f"No *_results.parquet files or model subdirectories found in {results_dir}")

    print(f"\nCombined {len(combined_df):,} total rows")

    if merge_demographics:
        combined_df = enrich_with_demographics(combined_df, metabonet_path)

    combined_df.to_parquet(output_path, index=False)
    print(f"\nSaved combined results to: {output_path}")
    print(f"Total rows: {len(combined_df):,}")
    print(f"Columns: {list(combined_df.columns)}")

    return combined_df


def add_to_combined(
    combined_path: Path,
    new_files: list,
    output_path: Path,
    merge_demographics: bool = True,
    replace_existing_models: bool = False,
    metabonet_path: str | None = None,
) -> pd.DataFrame:
    """
    Append per-model parquet files to an existing combined parquet.

    Each new file should have a 'model' column (typically all the same value) in either
    long format (with horizon/prediction/label rows) or wide format (label_t*/pred_t*).
    Demographics are merged into the new rows only, then concatenated to the existing file.
    """
    print(f"Loading existing combined file: {combined_path}")
    existing_df = pd.read_parquet(combined_path)
    print(f"  {len(existing_df):,} rows, {existing_df['model'].nunique()} models: {sorted(existing_df['model'].unique())}")

    new_dfs = []
    for pq_file in new_files:
        print(f"\nProcessing: {pq_file}")
        df = load_one_parquet(pq_file)
        if 'model' in df.columns:
            model_values = df['model'].unique()
            label = ', '.join(map(str, sorted(model_values)))
            print(f"  model(s): {label}, {len(df):,} rows")
        else:
            raise ValueError(f"{pq_file} has no 'model' column")
        new_dfs.append(df)

    new_df = pd.concat(new_dfs, ignore_index=True)
    print(f"\nCombined new rows: {len(new_df):,}")

    if merge_demographics:
        # Only enrich if the demographic columns aren't already present
        demo_present = all(c in new_df.columns for c in ['age', 'gender', 'cgm_mean'])
        if demo_present:
            print("New files already contain demographics; skipping MetaboNet merge.")
        else:
            new_df = enrich_with_demographics(new_df, metabonet_path)

    new_models = set(new_df['model'].unique())
    overlapping = new_models & set(existing_df['model'].unique())
    if overlapping:
        if replace_existing_models:
            print(f"\nReplacing existing rows for models: {sorted(overlapping)}")
            existing_df = existing_df[~existing_df['model'].isin(overlapping)].copy()
        else:
            print(f"\nWARNING: models {sorted(overlapping)} already exist in {combined_path}.")
            print("  Both old and new rows will be kept. Pass --replace to drop the old rows first.")

    # Align columns: union, preserving existing order then appending any new columns
    all_cols = list(existing_df.columns) + [c for c in new_df.columns if c not in existing_df.columns]
    existing_df = existing_df.reindex(columns=all_cols)
    new_df = new_df.reindex(columns=all_cols)

    final_df = pd.concat([existing_df, new_df], ignore_index=True)
    final_df.to_parquet(output_path, index=False)
    print(f"\nSaved combined results to: {output_path}")
    print(f"Total rows: {len(final_df):,} ({len(existing_df):,} existing + {len(new_df):,} new)")
    print(f"Models: {final_df['model'].nunique()} ({', '.join(sorted(final_df['model'].unique()))})")

    return final_df


def _print_summary(df: pd.DataFrame) -> None:
    print("\n--- Summary ---")
    print(f"Models: {df['model'].nunique()} ({', '.join(sorted(df['model'].unique()))})")
    if 'dataset' in df.columns:
        print(f"Datasets: {df['dataset'].nunique()} ({', '.join(sorted(df['dataset'].dropna().unique()))})")
    print(f"Unique users: {df['user_id'].nunique()}")

    label_cols = [c for c in df.columns if c.startswith('label_')]
    pred_cols = [c for c in df.columns if c.startswith('pred_')]
    if label_cols and pred_cols:
        print("\n--- Sample glucose values ---")
        print(f"Label range: {df[label_cols].min().min():.1f} - {df[label_cols].max().max():.1f}")
        print(f"Prediction range: {df[pred_cols].min().min():.1f} - {df[pred_cols].max().max():.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="Combine model result files into a single parquet file"
    )
    subparsers = parser.add_subparsers(dest="command")

    # `combine` (existing behavior; default if no subcommand given)
    combine_p = subparsers.add_parser(
        "combine",
        help="Build a combined parquet from a results directory (npy subdirs or *_results.parquet files)",
    )
    combine_p.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results_with_timestamps"),
        help="Path to results directory (npy subdirs or flat *_results.parquet files)",
    )
    combine_p.add_argument(
        "--output",
        type=Path,
        default=Path("combined_results_new.parquet"),
        help="Output parquet file path (default: combined_results_new.parquet)",
    )
    combine_p.add_argument(
        "--no-demographics",
        action="store_true",
        help="Skip merging demographics from MetaboNet",
    )
    combine_p.add_argument(
        "--metabonet-test",
        type=Path,
        default=None,
        help=(
            "Path to a local copy of the MetaboNet public test parquet. "
            f"Falls back to ${METABONET_TEST_ENV} (local path or s3:// URI)."
        ),
    )

    # `add` (new behavior: append per-model parquet files to an existing combined file)
    add_p = subparsers.add_parser(
        "add",
        help="Append per-model parquet files to an existing combined parquet",
    )
    add_p.add_argument(
        "--combined",
        type=Path,
        required=True,
        help="Existing combined parquet to append to",
    )
    add_p.add_argument(
        "--add-files",
        type=Path,
        nargs='+',
        required=True,
        dest="files",
        help="Per-model parquet files to add (each typically has a single value in 'model')",
    )
    add_p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output parquet file path (default: overwrite --combined)",
    )
    add_p.add_argument(
        "--no-demographics",
        action="store_true",
        help="Skip merging demographics from MetaboNet for the new rows",
    )
    add_p.add_argument(
        "--metabonet-test",
        type=Path,
        default=None,
        help=(
            "Path to a local copy of the MetaboNet public test parquet. "
            f"Falls back to ${METABONET_TEST_ENV} (local path or s3:// URI)."
        ),
    )
    add_p.add_argument(
        "--replace",
        action="store_true",
        help="If a model in the new files already exists in --combined, drop the existing rows first",
    )

    # Backward compat: if user passes top-level flags with no subcommand, default to `combine`
    parser.add_argument(
        "--results-dir",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-demographics",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    args = parser.parse_args()
    command = args.command or "combine"

    if command == "combine":
        results_dir = args.results_dir or Path("results_with_timestamps")
        output = args.output or Path("combined_results_new.parquet")
        if not results_dir.exists():
            print(f"Error: Results directory not found: {results_dir}")
            return 1
        metabonet_path = None
        if not args.no_demographics:
            try:
                metabonet_path = resolve_metabonet_test_path(getattr(args, "metabonet_test", None))
            except (RuntimeError, FileNotFoundError) as e:
                print(f"Error: {e}")
                return 1
        df = combine_results(
            results_dir,
            output,
            merge_demographics=not args.no_demographics,
            metabonet_path=metabonet_path,
        )
        _print_summary(df)
        return 0

    if command == "add":
        if not args.combined.exists():
            print(f"Error: Combined file not found: {args.combined}")
            return 1
        missing = [f for f in args.files if not f.exists()]
        if missing:
            print(f"Error: Files not found: {missing}")
            return 1
        output = args.output or args.combined
        metabonet_path = None
        if not args.no_demographics:
            try:
                metabonet_path = resolve_metabonet_test_path(getattr(args, "metabonet_test", None))
            except (RuntimeError, FileNotFoundError) as e:
                print(f"Error: {e}")
                return 1
        df = add_to_combined(
            args.combined,
            args.files,
            output,
            merge_demographics=not args.no_demographics,
            replace_existing_models=args.replace,
            metabonet_path=metabonet_path,
        )
        _print_summary(df)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    exit(main())
