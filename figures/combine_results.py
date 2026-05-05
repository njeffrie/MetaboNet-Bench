#!/usr/bin/env python3
"""
Combine model result files into a single parquet file merged with user demographics
from the MetaboNet public test split. The MetaboNet parquet location is resolved from
``--metabonet-test`` (CLI), then ``$METABONET_TEST_PARQUET`` (local path or s3:// URI).

Supports four input formats (auto-detected):

  npy (legacy): results_dir contains model subdirectories, each with <dataset>.npy files.
    Each .npy has shape (N, 4, 12):
      arr[:,0,:] timestamps (ns, prediction time; shifted +60min to align with label_t0)
      arr[:,1,:] user ID (repeated)
      arr[:,2,:] predictions (12 timesteps)
      arr[:,3,:] labels (12 timesteps)

  parquet dir: results_dir contains flat <model>_results.parquet files with columns:
      model, dataset, patient_id, timestamp (ns, = label time per horizon),
      prediction, label, horizon (1-12)

  single file (multi-model): one parquet containing predictions for many models.
    Auto-detects three layouts and reads one model's columns at a time via
    pyarrow (so the full table is never resident in memory):
      - long-with-explicit-model-column: same schema as parquet dir but with
        many models concatenated into one file.
      - long-by-horizon, wide-by-model: row per (sample, horizon), with one
        prediction column per model name (and a shared `label`).
      - wide-by-horizon + wide-by-model: row per sample, columns
        `label_t0..t11` plus per-model `<model>_pred_t0..t11`.

Output schema:
  user_id, timestamp_t0, dataset, model,
  label_t0..label_t11, pred_t0..pred_t11,
  Demographics from MetaboNet (age, gender, weight, height, ...)
"""

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

METABONET_TEST_ENV = "METABONET_TEST_PARQUET"
DEFAULT_OUTPUT = "combined_results_with_aux.parquet"

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

def _pivot_long_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot a long-format DataFrame to the canonical wide schema.

    Required columns: ``model``, ``dataset``, one of (``patient_id``, ``user_id``),
    ``label``, ``prediction``, ``horizon``, plus a timestamp expressed as either
    ``timestamp`` (ns int or datetime64) or ``timestamp_t0``.

    timestamp encodes the label time at each horizon, so
    timestamp_t0 = timestamp - (horizon-1)*5min is constant within a prediction window.
    """
    df = df.copy()

    if 'timestamp_t0' not in df.columns:
        ts = df['timestamp']
        if not pd.api.types.is_datetime64_any_dtype(ts):
            ts = pd.to_datetime(ts, unit='ns')
        df['timestamp_t0'] = ts - (df['horizon'] - 1) * pd.Timedelta(minutes=5)

    if 'patient_id' in df.columns and 'user_id' not in df.columns:
        df = df.rename(columns={'patient_id': 'user_id'})

    # Single strict reshape (no aggregation): each (model, dataset, user_id,
    # timestamp_t0, horizon) tuple is unique, so pivot_table's aggregation pass
    # is wasted work. df.pivot raises if duplicates exist — that's the right
    # contract here.
    wide = df.pivot(
        index=['model', 'dataset', 'user_id', 'timestamp_t0'],
        columns='horizon',
        values=['label', 'prediction'],
    )
    # ('label', 1) -> 'label_t0' ; ('prediction', 1) -> 'pred_t0'
    rename_kind = {'label': 'label', 'prediction': 'pred'}
    wide.columns = [f"{rename_kind[k]}_t{h - 1}" for k, h in wide.columns]
    return wide.reset_index()


def process_parquet_file(filepath: Path) -> pd.DataFrame:
    """Load a <model>_results.parquet file and pivot from long to wide format."""
    return _pivot_long_to_wide(pd.read_parquet(filepath))


# ---------------------------------------------------------------------------
# Single-file (multi-model) loader — reads one model's columns at a time so
# we never materialize the full table in memory.
# ---------------------------------------------------------------------------

_PRED_T_RE = re.compile(r'(.+)_pred_t(\d+)$')
_LABEL_T_RE = re.compile(r'label_t(\d+)$')


def _reshape_long_horizon_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Cheap alternative to ``df.pivot`` when each window has 12 horizon rows in order.

    Sorts once by ``(model, dataset, user_id, timestamp_t0, horizon)`` and reshapes
    label/prediction arrays via numpy. Falls back to the pivot-based path if the
    horizon pattern isn't a clean 1..12 repeat.
    """
    df = df.copy()
    if 'timestamp_t0' not in df.columns:
        ts = df['timestamp']
        if not pd.api.types.is_datetime64_any_dtype(ts):
            ts = pd.to_datetime(ts, unit='ns')
        df['timestamp_t0'] = ts - (df['horizon'] - 1) * pd.Timedelta(minutes=5)
    if 'patient_id' in df.columns and 'user_id' not in df.columns:
        df = df.rename(columns={'patient_id': 'user_id'})

    df = df.sort_values(
        ['model', 'dataset', 'user_id', 'timestamp_t0', 'horizon'],
        kind='mergesort',
    ).reset_index(drop=True)

    n = len(df)
    if n == 0 or n % 12 != 0:
        return _pivot_long_to_wide(df)
    horizons = df['horizon'].to_numpy()
    expected = np.tile(np.arange(1, 13), n // 12)
    if not np.array_equal(horizons, expected):
        return _pivot_long_to_wide(df)

    n_windows = n // 12
    labels = df['label'].to_numpy().reshape(n_windows, 12)
    preds = df['prediction'].to_numpy().reshape(n_windows, 12)
    out = df.iloc[::12][['model', 'dataset', 'user_id', 'timestamp_t0']].reset_index(drop=True)
    for t in range(12):
        out[f'label_t{t}'] = labels[:, t]
    for t in range(12):
        out[f'pred_t{t}'] = preds[:, t]
    return out


def _iter_model_chunks_from_single_file(filepath: Path):
    """
    Generator yielding per-model wide DataFrames from a single multi-model parquet.

    Reads only the columns needed for the current model — never the whole table.
    For each yielded chunk: shape is (n_windows, ~28) and memory footprint is
    comparable to a single per-model results file.
    """
    import pyarrow as pa
    pf = pq.ParquetFile(str(filepath))
    schema = pf.schema_arrow
    cols = list(schema.names)
    cols_set = set(cols)
    fields_by_name = {f.name: f for f in schema}

    user_col = next((c for c in ('user_id', 'patient_id') if c in cols_set), None)
    time_col = next((c for c in ('timestamp_t0', 'timestamp') if c in cols_set), None)

    # ---- 1. existing long format with explicit model column ---------------
    if {'model', 'prediction', 'label', 'horizon'}.issubset(cols_set):
        print(f"Detected long-format file with explicit 'model' column: {filepath.name}")
        df = process_parquet_file(filepath)
        for model, sub in df.groupby('model', sort=False):
            yield sub.reset_index(drop=True)
        return

    if user_col is None:
        raise ValueError(f"{filepath}: missing user_id/patient_id column")

    # ---- 3. wide-by-horizon + wide-by-model -------------------------------
    has_label_t = any(_LABEL_T_RE.fullmatch(c) for c in cols)
    pred_t_models = sorted({
        _PRED_T_RE.fullmatch(c).group(1) for c in cols if _PRED_T_RE.fullmatch(c)
    })
    if has_label_t and pred_t_models:
        print(
            f"Detected wide-by-horizon + wide-by-model file "
            f"({len(pred_t_models)} models, columns like '<model>_pred_t<h>')"
        )
        # Read minimal index + label_t* columns once, reuse per model.
        label_cols = [c for c in cols if _LABEL_T_RE.fullmatch(c)]
        index_cols = [user_col]
        if 'timestamp_t0' in cols_set:
            index_cols.append('timestamp_t0')
        elif 'timestamp' in cols_set:
            index_cols.append('timestamp')
        if 'dataset' in cols_set:
            index_cols.append('dataset')
        shared_cols = index_cols + label_cols
        print(f"  reading {len(shared_cols)} minimal shared columns once...")
        shared_df = pq.read_table(
            str(filepath), columns=shared_cols, use_threads=True,
        ).to_pandas()
        if 'patient_id' in shared_df.columns and 'user_id' not in shared_df.columns:
            shared_df = shared_df.rename(columns={'patient_id': 'user_id'})
        if 'dataset' not in shared_df.columns:
            shared_df['dataset'] = ''

        for model in pred_t_models:
            model_t_cols = sorted(
                [c for c in cols if _PRED_T_RE.fullmatch(c) and _PRED_T_RE.fullmatch(c).group(1) == model],
                key=lambda c: int(_PRED_T_RE.fullmatch(c).group(2)),
            )
            pred_df = pq.read_table(
                str(filepath), columns=model_t_cols, use_threads=True,
            ).to_pandas()
            pred_df.columns = [f"pred_t{_PRED_T_RE.fullmatch(c).group(2)}" for c in model_t_cols]
            chunk = pd.concat([shared_df, pred_df], axis=1)
            chunk['model'] = model
            yield chunk
            del chunk, pred_df
        return

    # ---- 2. long-by-horizon + wide-by-model -------------------------------
    if {'label', 'horizon'}.issubset(cols_set) and time_col is not None:
        # Treat any column outside the known schema as a *candidate* model column,
        # then narrow to numeric float types so demographic / categorical columns
        # (e.g. subject_split_across_traintest, label-encoded ids) don't leak in.
        known = {
            'label', 'horizon', 'model', 'prediction', 'dataset',
            'source_file', 'id', user_col, time_col,
        }
        candidates = [c for c in cols if c not in known]
        model_cols = [
            c for c in candidates
            if pa.types.is_floating(fields_by_name[c].type)
        ]
        skipped = [c for c in candidates if c not in model_cols]
        if not model_cols:
            raise ValueError(
                f"No floating-point prediction columns detected in {filepath}. "
                f"Candidates that were skipped due to non-float dtype: {skipped}"
            )
        print(
            f"Detected long-by-horizon + wide-by-model file "
            f"({len(model_cols)} models)"
        )
        if skipped:
            head = ', '.join(skipped[:6])
            tail = ', ...' if len(skipped) > 6 else ''
            print(f"  Skipped {len(skipped)} non-float columns (treated as auxiliary): {head}{tail}")

        # Per model: read only (user_col, time_col, horizon, label, dataset?,
        # this_model_col). That mirrors the column set of a single per-model
        # results file. ``dataset`` is included when present so the downstream
        # demographics merge (which keys on ``dataset``) can succeed.
        minimal_shared = [user_col, time_col, 'horizon', 'label']
        if 'dataset' in cols_set:
            minimal_shared.append('dataset')
        for model in model_cols:
            tbl = pq.read_table(
                str(filepath), columns=minimal_shared + [model], use_threads=True,
            )
            sub = tbl.to_pandas()
            del tbl
            sub = sub.rename(columns={model: 'prediction'})
            sub['model'] = model
            if 'dataset' not in sub.columns:
                sub['dataset'] = ''
            wide = _reshape_long_horizon_to_wide(sub)
            del sub
            yield wide
        return

    raise ValueError(
        f"Could not detect a supported schema for {filepath}.\n"
        f"  columns: {cols}"
    )


def _stream_combine_single_file_to_disk(
    filepath: Path,
    output_path: Path,
    merge_demographics: bool,
    metabonet_path: str | None,
) -> int:
    """Stream per-model wide chunks straight to ``output_path`` via ParquetWriter.

    Returns the total number of rows written.
    """
    import pyarrow as pa

    if merge_demographics:
        # Lookup tables are small; load them once and reuse for every chunk.
        # (timestamp_df can be large but is bounded — same size as the MetaboNet test split.)
        demographics_df, timestamp_df = _load_demographic_helpers(
            metabonet_path, want_timestamps=True,
        )
    else:
        demographics_df, timestamp_df = None, None

    writer: pq.ParquetWriter | None = None
    total = 0
    models_written: list[str] = []
    try:
        for chunk in _iter_model_chunks_from_single_file(filepath):
            model = str(chunk['model'].iloc[0]) if 'model' in chunk.columns else '?'
            print(f"  writing chunk: {model} ({len(chunk):,} windows)")
            if demographics_df is not None:
                chunk = _enrich_chunk(chunk, demographics_df, timestamp_df)

            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(str(output_path), table.schema)
            else:
                # Cast each subsequent chunk to the writer's schema so columns
                # appear in the same order with the same types.
                table = table.cast(writer.schema, safe=False)
            writer.write_table(table)
            total += len(chunk)
            models_written.append(model)
            del chunk, table
    finally:
        if writer is not None:
            writer.close()

    print(f"\nSaved combined results to: {output_path}")
    print(f"Total rows: {total:,}")
    print(f"Models written ({len(models_written)}): {', '.join(models_written)}")
    return total


def load_from_single_multi_model_file(filepath: Path) -> pd.DataFrame:
    """
    Load a single parquet that contains predictions for multiple models.

    Auto-detects three layouts and reads each model's columns separately via
    pyarrow so the full table is never resident in memory at once:

      1. Long-with-explicit-model-column
         (existing: ``model, prediction, label, horizon, ...``).
      2. Long-by-horizon, wide-by-model
         (``label, horizon, user_id, timestamp`` + one prediction column per model).
      3. Wide-by-horizon, wide-by-model
         (``label_t0..t11`` + per-model ``<model>_pred_t0..t11`` columns).
    """
    pf = pq.ParquetFile(str(filepath))
    cols = list(pf.schema_arrow.names)
    cols_set = set(cols)

    # ---- 1. existing long format with explicit model column ---------------
    if {'model', 'prediction', 'label', 'horizon'}.issubset(cols_set):
        print(f"Detected long-format file with explicit 'model' column: {filepath.name}")
        return process_parquet_file(filepath)

    user_col = next((c for c in ('user_id', 'patient_id') if c in cols_set), None)
    time_col = next((c for c in ('timestamp_t0', 'timestamp') if c in cols_set), None)

    # ---- 3. wide-by-horizon + wide-by-model -------------------------------
    has_label_t = any(_LABEL_T_RE.fullmatch(c) for c in cols)
    pred_t_models = sorted({
        _PRED_T_RE.fullmatch(c).group(1) for c in cols if _PRED_T_RE.fullmatch(c)
    })
    if has_label_t and pred_t_models and user_col is not None:
        print(
            f"Detected wide-by-horizon + wide-by-model file "
            f"({len(pred_t_models)} models, columns like '<model>_pred_t<h>')"
        )

        # Single pass over column names: bucket model→cols and build the rename map.
        model_to_cols: dict[str, list[str]] = {m: [] for m in pred_t_models}
        pred_renames: dict[str, str] = {}
        for c in cols:
            m = _PRED_T_RE.fullmatch(c)
            if m is not None:
                model_to_cols[m.group(1)].append(c)
                pred_renames[c] = f"pred_t{m.group(2)}"
        shared = [c for c in cols if c not in pred_renames]

        # Read the shared columns ONCE — they're identical across models, so
        # re-decompressing them per iteration is the real cost, not the regex.
        print(f"  reading {len(shared)} shared columns once...")
        shared_df = pq.read_table(
            str(filepath), columns=shared, use_threads=True,
        ).to_pandas()
        if 'patient_id' in shared_df.columns and 'user_id' not in shared_df.columns:
            shared_df = shared_df.rename(columns={'patient_id': 'user_id'})
        if 'dataset' not in shared_df.columns:
            shared_df['dataset'] = ''

        out = []
        for model, model_cols in model_to_cols.items():
            pred_df = pq.read_table(
                str(filepath), columns=model_cols, use_threads=True,
            ).to_pandas()
            pred_df.columns = [pred_renames[c] for c in model_cols]
            sub = pd.concat([shared_df, pred_df], axis=1)
            sub['model'] = model
            out.append(sub)
            del pred_df
            print(f"  - {model}: {len(sub):,} rows")
        return pd.concat(out, ignore_index=True)

    # ---- 2. long-by-horizon + wide-by-model -------------------------------
    if {'label', 'horizon'}.issubset(cols_set) and user_col is not None and time_col is not None:
        # Treat any column outside the known schema as a model's prediction column.
        known = {
            'label', 'horizon', 'model', 'prediction', 'dataset',
            'source_file', 'id', user_col, time_col,
        }
        model_cols = [c for c in cols if c not in known]
        if model_cols:
            print(
                f"Detected long-by-horizon + wide-by-model file "
                f"({len(model_cols)} models, columns: {', '.join(model_cols)})"
            )
            shared = [c for c in cols if c not in model_cols]
            out = []
            for model in model_cols:
                tbl = pq.read_table(str(filepath), columns=shared + [model])
                sub = tbl.to_pandas()
                sub = sub.rename(columns={model: 'prediction'})
                sub['model'] = model
                if 'dataset' not in sub.columns:
                    sub['dataset'] = ''
                if 'patient_id' not in sub.columns and user_col != 'patient_id':
                    sub = sub.rename(columns={user_col: 'patient_id'})
                wide = _pivot_long_to_wide(sub)
                out.append(wide)
                print(f"  - {model}: {len(wide):,} windows")
            return pd.concat(out, ignore_index=True)

    raise ValueError(
        f"Could not detect a supported schema for {filepath}.\n"
        f"  columns: {cols}\n"
        f"Expected one of:\n"
        f"  - long with explicit 'model' column: model, prediction, label, horizon, "
        f"user_id|patient_id, timestamp\n"
        f"  - long-by-horizon, wide-by-model: label, horizon, user_id|patient_id, "
        f"timestamp + one prediction column per model\n"
        f"  - wide-by-horizon + wide-by-model: label_t0..t11, user_id|patient_id, "
        f"timestamp_t0 + <model>_pred_t0..t11 columns"
    )


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

def _enrich_chunk(
    chunk_df: pd.DataFrame,
    demographics_df: pd.DataFrame,
    timestamp_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Apply demographics + CGM stats + (optional) historical insulin/carbs to a chunk.

    ``demographics_df`` must already have ``cgm_mean`` / ``cgm_std`` merged in.
    """
    chunk_df = chunk_df.copy()
    chunk_df['user_id_str'] = chunk_df['user_id'].astype(str)
    chunk_df = chunk_df.merge(
        demographics_df,
        left_on=['dataset', 'user_id_str'],
        right_on=['source_file', 'id'],
        how='left',
    )
    chunk_df = chunk_df.drop(columns=['source_file', 'id', 'user_id_str'])

    if timestamp_df is not None and 'timestamp_t0' in chunk_df.columns:
        chunk_df['user_id_str'] = chunk_df['user_id'].astype(str)
        chunk_df = chunk_df.merge(
            timestamp_df,
            left_on=['dataset', 'user_id_str', 'timestamp_t0'],
            right_on=['source_file', 'id', 'date'],
            how='left',
        )
        chunk_df = chunk_df.rename(columns={'CGM': 'cgm_at_t0'})
        chunk_df = chunk_df.drop(columns=['source_file', 'id', 'date', 'user_id_str'])

        for i in range(1, 6):
            offset_minutes = i * 5
            chunk_df[f'timestamp_tminus_{i}'] = (
                chunk_df['timestamp_t0'] - pd.Timedelta(minutes=offset_minutes)
            )
            chunk_df['user_id_str'] = chunk_df['user_id'].astype(str)
            hist_df = chunk_df.merge(
                timestamp_df[['source_file', 'id', 'date', 'insulin', 'carbs']],
                left_on=['dataset', 'user_id_str', f'timestamp_tminus_{i}'],
                right_on=['source_file', 'id', 'date'],
                how='left',
                suffixes=('', f'_tminus_{i}'),
            )
            chunk_df[f'insulin_tminus_{i}'] = (
                hist_df[f'insulin_tminus_{i}'] if f'insulin_tminus_{i}' in hist_df.columns
                else hist_df['insulin']
            )
            chunk_df[f'carbs_tminus_{i}'] = (
                hist_df[f'carbs_tminus_{i}'] if f'carbs_tminus_{i}' in hist_df.columns
                else hist_df['carbs']
            )
            chunk_df = chunk_df.drop(columns=[f'timestamp_tminus_{i}', 'user_id_str'])

    return chunk_df


def _load_demographic_helpers(metabonet_path: str, want_timestamps: bool):
    """Pre-load and pre-merge the small lookup tables used by ``_enrich_chunk``."""
    demographics_df = load_demographics(metabonet_path)
    cgm_stats_df = load_cgm_stats(metabonet_path)
    demographics_df = demographics_df.merge(
        cgm_stats_df, on=['source_file', 'id'], how='left'
    )
    timestamp_df = load_timestamp_data(metabonet_path) if want_timestamps else None
    return demographics_df, timestamp_df


def enrich_with_demographics(combined_df: pd.DataFrame, metabonet_path: str) -> pd.DataFrame:
    """Merge MetaboNet demographics, CGM stats, and historical insulin/carbs into a wide-format df."""
    demographics_df, timestamp_df = _load_demographic_helpers(
        metabonet_path,
        want_timestamps='timestamp_t0' in combined_df.columns,
    )
    combined_df = _enrich_chunk(combined_df, demographics_df, timestamp_df)

    demo_cols = [c for c in DEMOGRAPHIC_COLUMNS if c not in ['source_file', 'id']]
    demo_cols += ['cgm_mean', 'cgm_std']
    for col in demo_cols:
        non_null = combined_df[col].notna().sum()
        pct = 100 * non_null / len(combined_df)
        print(f"  {col}: {non_null:,} non-null ({pct:.1f}%)")

    if 'cgm_at_t0' in combined_df.columns:
        print("\n--- Timestamp merge statistics ---")
        matched_rows = combined_df['cgm_at_t0'].notna().sum()
        total_rows = len(combined_df)
        print(f"  Rows matched by timestamp: {matched_rows:,} / {total_rows:,} ({100*matched_rows/total_rows:.1f}%)")

        valid_mask = combined_df['cgm_at_t0'].notna() & combined_df['label_t0'].notna()
        if valid_mask.sum() > 0:
            cgm_t0 = combined_df.loc[valid_mask, 'cgm_at_t0']
            label_t0 = combined_df.loc[valid_mask, 'label_t0']
            close_matches = (abs(cgm_t0 - label_t0) < 0.1).sum()
            print(f"  Close matches (|diff| < 0.1): {close_matches:,} / {valid_mask.sum():,} ({100*close_matches/valid_mask.sum():.1f}%)")

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
    results_path: Path,
    output_path: Path,
    merge_demographics: bool = True,
    metabonet_path: str | None = None,
) -> pd.DataFrame | None:
    # Auto-detect: single multi-model file vs. directory of per-model files.
    if results_path.is_file():
        # Streaming path: never materialize all models in memory at once.
        print(f"Loading from single multi-model file: {results_path}")
        _stream_combine_single_file_to_disk(
            results_path, output_path, merge_demographics, metabonet_path,
        )
        return None
    if results_path.is_dir():
        has_parquets = bool(list(results_path.glob("*_results.parquet")))
        has_npy_subdirs = any(
            p.is_dir() and not p.name.startswith('.')
            for p in results_path.iterdir()
        )

        if has_parquets:
            print("Detected parquet format")
            combined_df = load_from_parquet_dir(results_path)
        elif has_npy_subdirs:
            print("Detected npy format")
            combined_df = load_from_npy_dir(results_path)
        else:
            raise ValueError(
                f"No *_results.parquet files or model subdirectories found in {results_path}"
            )
    else:
        raise ValueError(f"Input path does not exist: {results_path}")

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
        help=(
            "Build a combined parquet from either (a) a directory of per-model "
            "results (npy subdirs or flat *_results.parquet files) or (b) a "
            "single multi-model parquet file."
        ),
    )
    combine_p.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results_with_timestamps"),
        help=(
            "Path to a results directory (npy subdirs or flat *_results.parquet "
            "files) OR to a single parquet file containing predictions for "
            "multiple models. Single-file layouts are auto-detected and read "
            "one model's columns at a time via pyarrow."
        ),
    )
    combine_p.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help="Alias for --results-dir when pointing at a single multi-model parquet.",
    )
    combine_p.add_argument(
        "--output",
        type=Path,
        default=Path(DEFAULT_OUTPUT),
        help=f"Output parquet file path (default: {DEFAULT_OUTPUT})",
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
        results_path = (
            getattr(args, "input_file", None)
            or args.results_dir
            or Path("results_with_timestamps")
        )
        output = args.output or Path(DEFAULT_OUTPUT)
        if not results_path.exists():
            print(f"Error: Input path not found: {results_path}")
            return 1
        metabonet_path = None
        if not args.no_demographics:
            try:
                metabonet_path = resolve_metabonet_test_path(getattr(args, "metabonet_test", None))
            except (RuntimeError, FileNotFoundError) as e:
                print(f"Error: {e}")
                return 1
        df = combine_results(
            results_path,
            output,
            merge_demographics=not args.no_demographics,
            metabonet_path=metabonet_path,
        )
        if df is not None:
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
