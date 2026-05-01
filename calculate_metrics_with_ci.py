from pathlib import Path

import click
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


def _rmse(pred, label):
    return np.sqrt(np.mean((pred - label) ** 2))


def _mard(pred, label):
    return np.mean(np.abs(pred - label) / np.abs(label)) * 100


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


def _bootstrap_metrics_numpy(pred, label, n_bootstrap, ci, rng, bootstrap_batch_size):
    n = len(label)
    rmse_samples = np.empty(n_bootstrap, dtype=np.float64)
    mard_samples = np.empty(n_bootstrap, dtype=np.float64)

    for start in range(0, n_bootstrap, bootstrap_batch_size):
        end = min(start + bootstrap_batch_size, n_bootstrap)
        idx = rng.integers(0, n, size=(end - start, n))
        sample_pred = pred[idx]
        sample_label = label[idx]
        err = sample_pred - sample_label
        rmse_samples[start:end] = np.sqrt(np.mean(err ** 2, axis=1))
        mard_samples[start:end] = np.mean(
            np.abs(err) / np.abs(sample_label), axis=1
        ) * 100

    alpha = (100 - ci) / 2
    return {
        "rmse": _rmse(pred, label),
        "rmse_ci_lower": np.percentile(rmse_samples, alpha),
        "rmse_ci_upper": np.percentile(rmse_samples, 100 - alpha),
        "mard": _mard(pred, label),
        "mard_ci_lower": np.percentile(mard_samples, alpha),
        "mard_ci_upper": np.percentile(mard_samples, 100 - alpha),
    }


def _bootstrap_metrics_torch(pred, label, n_bootstrap, ci, seed, device, bootstrap_batch_size):
    n = len(label)
    pred_t = torch.as_tensor(pred, dtype=torch.float32, device=device)
    label_t = torch.as_tensor(label, dtype=torch.float32, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    rmse_samples = torch.empty(n_bootstrap, dtype=torch.float32, device=device)
    mard_samples = torch.empty(n_bootstrap, dtype=torch.float32, device=device)

    for start in range(0, n_bootstrap, bootstrap_batch_size):
        end = min(start + bootstrap_batch_size, n_bootstrap)
        idx = torch.randint(
            0, n, (end - start, n),
            device=device,
            generator=generator,
        )
        sample_pred = pred_t[idx]
        sample_label = label_t[idx]
        err = sample_pred - sample_label
        rmse_samples[start:end] = torch.sqrt(torch.mean(err.square(), dim=1))
        mard_samples[start:end] = torch.mean(
            torch.abs(err) / torch.abs(sample_label), dim=1
        ) * 100

    alpha = (100 - ci) / 200
    quantiles = torch.tensor([alpha, 1 - alpha], dtype=torch.float32, device=device)
    rmse_ci = torch.quantile(rmse_samples, quantiles).cpu().numpy()
    mard_ci = torch.quantile(mard_samples, quantiles).cpu().numpy()

    err = pred_t - label_t
    rmse = torch.sqrt(torch.mean(err.square())).item()
    mard = (torch.mean(torch.abs(err) / torch.abs(label_t)) * 100).item()

    return {
        "rmse": rmse,
        "rmse_ci_lower": float(rmse_ci[0]),
        "rmse_ci_upper": float(rmse_ci[1]),
        "mard": mard,
        "mard_ci_lower": float(mard_ci[0]),
        "mard_ci_upper": float(mard_ci[1]),
    }


def _bootstrap_metrics(
    pred,
    label,
    n_bootstrap,
    ci,
    rng,
    seed,
    device,
    bootstrap_batch_size,
):
    if device.startswith("cuda"):
        return _bootstrap_metrics_torch(
            pred, label, n_bootstrap, ci, seed, device, bootstrap_batch_size
        )
    return _bootstrap_metrics_numpy(
        pred, label, n_bootstrap, ci, rng, bootstrap_batch_size
    )


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


if __name__ == "__main__":
    main()
