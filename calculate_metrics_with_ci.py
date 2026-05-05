from pathlib import Path

import click
import numpy as np
import pandas as pd

from eval_utils import (
    bootstrap_mean_ci,
    dts_zones,
    iter_model_frames,
    load_results,
    resolve_device,
)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda iterable, **_: iterable


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


def _bootstrap_metrics(pred, label, n_bootstrap, ci, rng, seed, device, bootstrap_batch_size):
    rmse, rmse_lo, rmse_hi = bootstrap_mean_ci(
        (pred - label) ** 2, n_bootstrap, ci, rng, seed,
        device, bootstrap_batch_size, np.sqrt)
    mard, mard_lo, mard_hi = bootstrap_mean_ci(
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
    device = resolve_device(device)
    rng = np.random.default_rng(seed)
    rows = []
    for model, model_df, prediction_col in tqdm(
        list(iter_model_frames(df)), desc="Bootstrapping DTS CIs", unit="model"
    ):
        for split_type, split_df in _split_subsets(model_df, by_split):
            hdf = split_df[split_df["horizon"] == horizon_minutes // 5]
            hdf = hdf.dropna(subset=[prediction_col, "label"])
            if hdf.empty:
                continue
            zones = dts_zones(
                hdf["label"].to_numpy(), hdf[prediction_col].to_numpy(), dts_grid_path)
            row = {
                "model": model,
                "split_type": split_type,
                "horizon_minutes": horizon_minutes,
                "n_predictions": len(zones),
            }
            for name, mask in {**{z: zones == z for z in "ABCDE"}, "CDE": np.isin(zones, list("CDE"))}.items():
                count = int(mask.sum())
                pct, lower, upper = bootstrap_mean_ci(
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
    device = resolve_device(device)
    rng = np.random.default_rng(seed)
    rows = []

    model_frames = list(iter_model_frames(df))
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
    df = load_results(input_path)
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
