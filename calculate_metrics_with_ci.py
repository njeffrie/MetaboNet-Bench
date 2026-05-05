from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda iterable, **_: iterable

try:
    import torch
except ImportError:
    torch = None


METADATA_COLS = {
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


def _prediction_columns(df):
    return [
        c for c in df.columns
        if c not in METADATA_COLS and c not in {"model", "prediction"}
    ]


def _load_results(input_path):
    input_path = Path(input_path)
    if input_path.is_dir():
        frames = []
        for path in sorted(input_path.glob("*_results.parquet")):
            df = pd.read_parquet(path)
            if "model" not in df.columns:
                df["model"] = path.stem.replace("_results", "")
            frames.append(df)
        if not frames:
            raise click.ClickException(f"No *_results.parquet files found in {input_path}")
        return pd.concat(frames, ignore_index=True)
    return pd.read_parquet(input_path)


def _iter_model_frames(df):
    if {"model", "prediction"} <= set(df.columns):
        for model, model_df in df.groupby("model", sort=True):
            yield model, model_df, "prediction"
    else:
        pred_cols = _prediction_columns(df)
        if not pred_cols:
            raise click.ClickException(
                "No model prediction columns found. Expected long-form "
                "`model`/`prediction` columns or a wide table with model columns."
            )
        for model in sorted(pred_cols):
            yield model, df, model


def _parse_horizons(horizons):
    if not horizons:
        return None
    return [int(h.strip()) for h in horizons.split(",") if h.strip()]


def _split_subsets(df, by_split):
    yield "overall", df
    if not by_split:
        return
    if "subject_split_across_traintest" not in df.columns:
        raise click.ClickException(
            "--by_split requires a subject_split_across_traintest column"
        )
    split_col = df["subject_split_across_traintest"].fillna(False).astype(bool)
    yield "known_patients", df[split_col]
    yield "new_patients", df[~split_col]


def _resolve_device(device):
    if device == "auto":
        return "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        if torch is None:
            raise click.ClickException("CUDA bootstrap requires torch to be installed")
        if not torch.cuda.is_available():
            raise click.ClickException("CUDA was requested, but torch.cuda.is_available() is false")
    return device


def _bootstrap_mean_ci(
    values, n_bootstrap, ci, rng, seed, device, batch_size, transform=lambda x: x,
):
    values = np.asarray(values, dtype=np.float64)
    samples = np.empty(n_bootstrap, dtype=np.float64)
    if device.startswith("cuda"):
        values_t = torch.as_tensor(values, dtype=torch.float32, device=device)
        gen = torch.Generator(device=device).manual_seed(seed)
        for start in range(0, n_bootstrap, batch_size):
            end = min(start + batch_size, n_bootstrap)
            idx = torch.randint(0, len(values), (end - start, len(values)), device=device, generator=gen)
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


def _bootstrap_metrics(pred, label, n_bootstrap, ci, rng, seed, device, bootstrap_batch_size):
    rmse, rmse_lo, rmse_hi = _bootstrap_mean_ci(
        (pred - label) ** 2, n_bootstrap, ci, rng, seed,
        device, bootstrap_batch_size, np.sqrt)
    mard, mard_lo, mard_hi = _bootstrap_mean_ci(
        np.abs(pred - label) / np.abs(label) * 100, n_bootstrap, ci, rng,
        seed + 1, device, bootstrap_batch_size)
    return {
        "rmse": rmse,
        "rmse_ci_lower": rmse_lo,
        "rmse_ci_upper": rmse_hi,
        "mard": mard,
        "mard_ci_lower": mard_lo,
        "mard_ci_upper": mard_hi,
    }


def _dts_zones(labels, predictions, dts_grid_path, extent=(-62, 835, -47, 646)):
    zone_rgb = {
        "A": np.array([0.5647059, 0.72156864, 0.5019608], dtype=np.float32),
        "B": np.array([1.0039216, 1.0039216, 0.59607846], dtype=np.float32),
        "C": np.array([0.972549, 0.8156863, 0.5647059], dtype=np.float32),
        "D": np.array([0.9411765, 0.53333336, 0.5019608], dtype=np.float32),
        "E": np.array([0.78431374, 0.53333336, 0.65882355], dtype=np.float32),
    }
    r, p = map(lambda x: np.asarray(x).ravel(), (labels, predictions))
    img = plt.imread(dts_grid_path).astype(np.float32)
    h, w = img.shape[:2]
    xmin, xmax, ymin, ymax = extent
    x = np.round((r - xmin) / (xmax - xmin) * (w - 1)).astype(int)
    y = np.round((ymax - p) / (ymax - ymin) * (h - 1)).astype(int)
    keys = np.array(list(zone_rgb), dtype="<U1")
    colors = np.stack([zone_rgb[k] for k in keys])
    return keys[np.argmin(((img[y, x, :3][:, None] - colors) ** 2).sum(-1), axis=1)]


def dts_confidence_intervals(
    df,
    dts_grid_path,
    horizon_minutes=30,
    by_split=False,
    n_bootstrap=1000,
    ci=95.0,
    seed=42,
    device="auto",
    bootstrap_batch_size=256,
):
    device = _resolve_device(device)
    rng = np.random.default_rng(seed)
    rows = []
    for model, model_df, prediction_col in tqdm(
        list(_iter_model_frames(df)), desc="Bootstrapping DTS CIs", unit="model"
    ):
        for split_type, split_df in _split_subsets(model_df, by_split):
            hdf = split_df[split_df["horizon"] == horizon_minutes // 5]
            hdf = hdf.dropna(subset=[prediction_col, "label"])
            if hdf.empty:
                continue
            zones = _dts_zones(
                hdf["label"].to_numpy(), hdf[prediction_col].to_numpy(), dts_grid_path)
            row = {
                "model": model,
                "split_type": split_type,
                "horizon_minutes": horizon_minutes,
                "n_predictions": len(zones),
            }
            for name, mask in {**{z: zones == z for z in "ABCDE"}, "CDE": np.isin(zones, list("CDE"))}.items():
                count = int(mask.sum())
                pct, lower, upper = _bootstrap_mean_ci(
                    mask.astype(np.float32) * 100, n_bootstrap, ci, rng,
                    seed + len(rows), device, bootstrap_batch_size)
                row.update({
                    f"zone_{name}_count": count,
                    f"zone_{name}_pct": pct,
                    f"zone_{name}_ci_lower": lower,
                    f"zone_{name}_ci_upper": upper,
                })
            rows.append(row)
    return pd.DataFrame(rows)


def _rows_for_subset(
    model,
    subset,
    prediction_col,
    split_type,
    horizons,
    include_overall,
    n_bootstrap,
    ci,
    rng,
    seed,
    device,
    bootstrap_batch_size,
):
    rows = []
    available_horizons = sorted(subset["horizon"].dropna().unique())
    selected_horizons = horizons or available_horizons

    groups = []
    for horizon in selected_horizons:
        h_df = subset[subset["horizon"] == horizon]
        groups.append((int(horizon), int(horizon) * 5, h_df))
    if include_overall:
        groups.append(("overall", "overall", subset))

    for group_idx, (horizon_step, horizon_minutes, group_df) in enumerate(groups):
        group_df = group_df.dropna(subset=[prediction_col, "label"])
        if group_df.empty:
            continue

        pred = group_df[prediction_col].to_numpy(dtype=np.float64)
        label = group_df["label"].to_numpy(dtype=np.float64)
        metrics = _bootstrap_metrics(
            pred,
            label,
            n_bootstrap,
            ci,
            rng,
            seed + group_idx,
            device,
            bootstrap_batch_size,
        )
        rows.append({
            "model": model,
            "split_type": split_type,
            "horizon_step": horizon_step,
            "horizon_minutes": horizon_minutes,
            "n_predictions": len(group_df),
            "n_bootstrap": n_bootstrap,
            "ci": ci,
            **metrics,
        })

    return rows


def bootstrap_confidence_intervals(
    df,
    horizons=None,
    include_overall=True,
    by_split=False,
    n_bootstrap=1000,
    ci=95.0,
    seed=42,
    device="auto",
    bootstrap_batch_size=256,
):
    if n_bootstrap <= 0:
        raise click.ClickException("--n_bootstrap must be positive")
    if bootstrap_batch_size <= 0:
        raise click.ClickException("--bootstrap_batch_size must be positive")
    device = _resolve_device(device)
    rng = np.random.default_rng(seed)
    rows = []

    model_frames = list(_iter_model_frames(df))
    for model, model_df, prediction_col in tqdm(
        model_frames, desc="Bootstrapping CIs", unit="model"
    ):
        for split_type, split_df in _split_subsets(model_df, by_split):
            if split_df.empty:
                continue
            rows.extend(_rows_for_subset(
                model=model,
                subset=split_df,
                prediction_col=prediction_col,
                split_type=split_type,
                horizons=horizons,
                include_overall=include_overall,
                n_bootstrap=n_bootstrap,
                ci=ci,
                rng=rng,
                seed=seed + len(rows),
                device=device,
                bootstrap_batch_size=bootstrap_batch_size,
            ))

    return pd.DataFrame(rows)


@click.command()
@click.option(
    "--input_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Combined results parquet or directory containing *_results.parquet files.",
)
@click.option(
    "--output_path",
    type=click.Path(path_type=Path),
    default=Path("results/metric_confidence_intervals.csv"),
    show_default=True,
    help="Where to save bootstrap confidence intervals.",
)
@click.option(
    "--dts_output_path",
    type=click.Path(path_type=Path),
    default=Path("results/dts_zone_confidence_intervals.csv"),
    show_default=True,
    help="Where to save DTS zone confidence intervals.",
)
@click.option(
    "--dts_grid_path",
    type=click.Path(exists=True, path_type=Path),
    default=Path("data/dts_grid.png"),
    show_default=True,
)
@click.option("--dts_horizon_minutes", type=int, default=30, show_default=True)
@click.option("--skip_dts", is_flag=True, help="Only calculate RMSE/MARD CIs.")
@click.option("--n_bootstrap", type=int, default=1000, show_default=True)
@click.option("--ci", type=float, default=95.0, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option(
    "--device",
    type=str,
    default="auto",
    show_default=True,
    help="Bootstrap backend device: auto, cpu, cuda, or cuda:<index>.",
)
@click.option(
    "--bootstrap_batch_size",
    type=int,
    default=256,
    show_default=True,
    help="Number of bootstrap resamples to compute per vectorized batch.",
)
@click.option(
    "--horizons",
    type=str,
    default=None,
    help="Comma-separated horizon steps to include. Defaults to all present.",
)
@click.option("--include_overall/--no_include_overall", default=True, show_default=True)
@click.option("--by_split", is_flag=True, help="Also compute CIs by known/new patient split.")
def main(
    input_path,
    output_path,
    dts_output_path,
    dts_grid_path,
    dts_horizon_minutes,
    skip_dts,
    n_bootstrap,
    ci,
    seed,
    device,
    bootstrap_batch_size,
    horizons,
    include_overall,
    by_split,
):
    """Bootstrap confidence intervals for RMSE and MARD by model and horizon."""
    df = _load_results(input_path)
    ci_df = bootstrap_confidence_intervals(
        df,
        horizons=_parse_horizons(horizons),
        include_overall=include_overall,
        by_split=by_split,
        n_bootstrap=n_bootstrap,
        ci=ci,
        seed=seed,
        device=device,
        bootstrap_batch_size=bootstrap_batch_size,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".parquet":
        ci_df.to_parquet(output_path, index=False, engine="pyarrow", compression="zstd")
    else:
        ci_df.to_csv(output_path, index=False)

    print(f"Saved {len(ci_df)} confidence interval rows to {output_path}")

    if not skip_dts:
        dts_df = dts_confidence_intervals(
            df,
            dts_grid_path=dts_grid_path,
            horizon_minutes=dts_horizon_minutes,
            by_split=by_split,
            n_bootstrap=n_bootstrap,
            ci=ci,
            seed=seed,
            device=device,
            bootstrap_batch_size=bootstrap_batch_size,
        )
        dts_output_path.parent.mkdir(parents=True, exist_ok=True)
        if dts_output_path.suffix == ".parquet":
            dts_df.to_parquet(
                dts_output_path, index=False, engine="pyarrow", compression="zstd")
        else:
            dts_df.to_csv(dts_output_path, index=False)
        print(f"Saved {len(dts_df)} DTS confidence interval rows to {dts_output_path}")


if __name__ == "__main__":
    main()
