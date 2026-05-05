#!/usr/bin/env python3
"""
Plot RMSE vs reference CGM label value with binned means and confidence bounds.
Shows how prediction error varies with actual glucose level.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model_styles import (
    add_legends_below,
    get_marker_edge_kwargs,
    get_model_linestyle,
    get_model_marker,
    add_figsize_arg,
    add_model_filter_args,
    add_model_legend_below,
    apply_model_filter,
    get_model_color_for,
    get_model_label,
    sort_models_for_render,
)


def compute_error_by_label(df: pd.DataFrame, timestep: int) -> pd.DataFrame:
    """
    Compute prediction error for each sample at a given timestep.

    Args:
        df: DataFrame with model, label_t*, pred_t* columns
        timestep: Timestep index (0-11)

    Returns:
        DataFrame with model, label, squared_error columns
    """
    label_col = f'label_t{timestep}'
    pred_col = f'pred_t{timestep}'

    result = pd.DataFrame({
        'model': df['model'],
        'label': df[label_col],
        'pred': df[pred_col],
        'squared_error': (df[pred_col] - df[label_col]) ** 2,
    })

    return result


def _label_xrange(error_df: pd.DataFrame, bg_min: float = 39.0, bg_max: float | None = None) -> tuple:
    """Shared x-range used by every BG-axis chart in this script.

    Lower bound defaults to 39 mg/dL (CGM physical floor). Upper bound defaults to the
    99th percentile of label values when not given, to avoid a long sparse tail.
    """
    labels = error_df['label'].dropna().to_numpy()
    if bg_max is None:
        bg_max = float(np.nanpercentile(labels, 99))
    return float(bg_min), float(bg_max)


def _nice_xticks(xmin: float, xmax: float, max_ticks: int = 10) -> np.ndarray:
    """Round-number ticks in [xmin, xmax], at most max_ticks of them."""
    span = xmax - xmin
    if span <= 0 or not np.isfinite(span):
        return np.array([xmin])
    for step in (5, 10, 20, 25, 50, 100, 200, 250, 500, 1000):
        start = int(np.ceil(xmin / step)) * step
        ticks = np.arange(start, xmax + step / 2, step)
        if 0 < len(ticks) <= max_ticks:
            return ticks
    return np.linspace(xmin, xmax, max_ticks)


def compute_binned_rmse(x, squared_errors, n_bins=30, min_count=10, bg_min=39.0, bg_max=None):
    """
    Compute binned RMSE with standard error.

    Args:
        x: x values (label/CGM values)
        squared_errors: squared prediction errors
        n_bins: Number of bins
        min_count: Minimum samples per bin to include
        bg_min: Lower bin edge (mg/dL). Defaults to 39 (CGM physical floor).
        bg_max: Upper bin edge (mg/dL). Defaults to the 99th percentile of x.

    Returns:
        bin_centers, rmse_values, rmse_sems, counts
    """
    x = np.array(x)
    squared_errors = np.array(squared_errors)

    if bg_max is None:
        bg_max = float(np.nanpercentile(x, 99))

    bins = np.linspace(float(bg_min), float(bg_max), n_bins + 1)
    bin_indices = np.digitize(x, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_centers = (bins[:-1] + bins[1:]) / 2
    rmse_values = np.full(n_bins, np.nan)
    rmse_sems = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins)

    for i in range(n_bins):
        mask = bin_indices == i
        count = mask.sum()
        counts[i] = count
        if count >= min_count:
            mse = np.mean(squared_errors[mask])
            rmse_values[i] = np.sqrt(mse)
            # SEM for RMSE: approximate using delta method
            # SE(RMSE) ≈ std(squared_errors) / (2 * RMSE * sqrt(n))
            std_se = np.std(squared_errors[mask])
            rmse_sems[i] = std_se / (2 * rmse_values[i] * np.sqrt(count))

    # Filter out bins with insufficient data
    valid = ~np.isnan(rmse_values)
    return bin_centers[valid], rmse_values[valid], rmse_sems[valid], counts[valid]


def plot_rmse_vs_label(
    error_df: pd.DataFrame,
    horizon_minutes: int,
    output_path: Path,
    figsize: tuple = (14, 10),
    dpi: int = 150,
    n_bins: int = 30,
    show_scatter: bool = True,
    bg_min: float = 39.0,
    bg_max: float | None = None,
) -> None:
    """
    Create plot of RMSE vs label value with binned RMSE and CI.

    Args:
        error_df: DataFrame with model, label, squared_error columns
        horizon_minutes: Prediction horizon in minutes (for title)
        output_path: Path to save the plot
        figsize: Figure size in inches
        dpi: Resolution for saved figure
        n_bins: Number of bins for computing stats
        show_scatter: Whether to show scatter points (shows sqrt of squared error)
    """
    models = sorted(error_df['model'].unique())
    n_models = len(models)
    cols = 3
    rows = (n_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize, sharex=True, sharey=True)
    axes = axes.flatten()

    xmin, xmax = _label_xrange(error_df, bg_min=bg_min, bg_max=bg_max)

    for idx, model in enumerate(models):
        ax = axes[idx]
        model_data = error_df[error_df['model'] == model].dropna(subset=['label', 'squared_error'])

        if len(model_data) < 50:
            ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes,
                    ha='center', va='center')
            ax.set_title(model, fontsize=11, fontweight='bold')
            continue

        x = model_data['label'].values
        sq_err = model_data['squared_error'].values

        # Scatter plot of absolute errors (subsample for performance)
        if show_scatter:
            abs_err = np.sqrt(sq_err)
            if len(x) > 5000:
                sample_idx = np.random.choice(len(x), 5000, replace=False)
                ax.scatter(x[sample_idx], abs_err[sample_idx], alpha=0.1, s=5, c='steelblue', edgecolors='none')
            else:
                ax.scatter(x, abs_err, alpha=0.1, s=5, c='steelblue', edgecolors='none')

        # Binned RMSE
        bin_centers, rmse_values, rmse_sems, counts = compute_binned_rmse(
            x, sq_err, n_bins=n_bins, bg_min=bg_min, bg_max=bg_max,
        )

        ci_mult = 1.96  # 95% CI

        ax.plot(bin_centers, rmse_values, 'r-', linewidth=2, label='RMSE')
        ax.fill_between(
            bin_centers,
            rmse_values - ci_mult * rmse_sems,
            rmse_values + ci_mult * rmse_sems,
            color='red', alpha=0.2, label='95% CI'
        )

        # Stats annotation
        overall_rmse = np.sqrt(np.mean(sq_err))
        ax.text(
            0.05, 0.95,
            f'RMSE = {overall_rmse:.1f}\nn = {len(model_data):,}',
            transform=ax.transAxes,
            fontsize=8,
            va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        )

        ax.set_title(model, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(xmin, xmax)
        ax.set_xticks(_nice_xticks(xmin, xmax))

        if idx >= (rows - 1) * cols:
            ax.set_xlabel('CGM at Prediction (mg/dL)', fontsize=10)
        if idx % cols == 0:
            ax.set_ylabel('RMSE (mg/dL)', fontsize=10)

    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    title = f'RMSE vs Reference CGM at {horizon_minutes}min Horizon'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved plot to {output_path}")


def plot_rmse_vs_label_combined(
    error_df: pd.DataFrame,
    horizon_minutes: int,
    output_path: Path,
    figsize: tuple = (10, 6),
    dpi: int = 150,
    n_bins: int = 30,
    color_by: str = 'arch',
    bg_min: float = 39.0,
    bg_max: float | None = None,
) -> None:
    """
    Create a single plot with all models' binned RMSE curves and CIs.

    Args:
        error_df: DataFrame with model, label, squared_error columns
        horizon_minutes: Prediction horizon in minutes (for title)
        output_path: Path to save the plot
        figsize: Figure size in inches
        dpi: Resolution for saved figure
        n_bins: Number of bins for computing stats
    """
    fig, ax = plt.subplots(figsize=figsize)

    models = sort_models_for_render(error_df['model'].unique())

    ci_mult = 1.96  # 95% CI

    for model in models:
        model_data = error_df[error_df['model'] == model].dropna(subset=['label', 'squared_error'])

        if len(model_data) < 50:
            continue

        x = model_data['label'].values
        sq_err = model_data['squared_error'].values

        # Binned RMSE
        bin_centers, rmse_values, rmse_sems, counts = compute_binned_rmse(
            x, sq_err, n_bins=n_bins, bg_min=bg_min, bg_max=bg_max,
        )

        color = get_model_color_for(model, color_by=color_by)
        ax.plot(bin_centers, rmse_values, color=color, linewidth=1.0,
                label=get_model_label(model, color_by=color_by),
                linestyle=get_model_linestyle(model),
                marker=get_model_marker(model),
                markersize=5, markevery=5,
                **get_marker_edge_kwargs())
        ax.fill_between(bin_centers, rmse_values - ci_mult * rmse_sems, rmse_values + ci_mult * rmse_sems, color=color, alpha=0.15)

    ax.set_xlabel('CGM at Prediction (mg/dL)', fontsize=12)
    ax.set_ylabel('RMSE (mg/dL)', fontsize=12)
    ax.set_title(f'RMSE vs CGM at Prediction for {horizon_minutes}min Horizon', fontsize=14, fontweight='bold')
    xmin, xmax = _label_xrange(error_df, bg_min=bg_min, bg_max=bg_max)
    ax.set_xlim(xmin, xmax)
    ax.set_xticks(_nice_xticks(xmin, xmax))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if color_by == 'feature':
        add_legends_below(fig, ax)
    else:
        add_model_legend_below(fig, ax)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved combined plot to {output_path}")


def plot_label_support(
    error_df: pd.DataFrame,
    horizon_minutes: int,
    output_path: Path,
    figsize: tuple = (10, 3.5),
    dpi: int = 150,
    n_bins: int = 30,
    bg_min: float = 39.0,
    bg_max: float | None = None,
) -> None:
    """Bar chart of sample counts per Reference CGM bin (matches plot_rmse_by_bg x-axis)."""
    # Sample support is identical across models (one row per model per sample),
    # so use any single model's labels to bin once.
    one_model = error_df['model'].iloc[0]
    labels = error_df.loc[error_df['model'] == one_model, 'label'].dropna().to_numpy()
    if len(labels) == 0:
        print(f"No labels available for support plot; skipping {output_path}")
        return

    # Use the same bin edges that compute_binned_rmse uses (bg_min .. bg_max, n_bins).
    xmin, xmax = _label_xrange(error_df, bg_min=bg_min, bg_max=bg_max)
    bins = np.linspace(xmin, xmax, n_bins + 1)
    counts, _ = np.histogram(labels, bins=bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(bin_centers, counts, width=bin_width * 0.9, color='steelblue', edgecolor='none')
    ax.set_xlabel('Reference CGM (mg/dL)', fontsize=12)
    ax.set_ylabel('Samples', fontsize=12)
    ax.set_title(
        f'Reference CGM Distribution (n = {len(labels):,})',
        fontsize=13, fontweight='bold',
    )
    ax.set_xlim(xmin, xmax)
    ax.set_xticks(_nice_xticks(xmin, xmax))
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved support plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot prediction error vs reference CGM label"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("combined_results_new.parquet"),
        help="Input parquet file (default: combined_results_new.parquet)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        choices=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
        default=60,
        help="Prediction horizon in minutes (default: 60)",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=30,
        help="Number of bins for computing stats (default: 30)",
    )
    parser.add_argument(
        "--bg-min",
        type=float,
        default=39.0,
        help="Lower x-axis / bin edge in mg/dL (default: 39, the CGM physical floor)",
    )
    parser.add_argument(
        "--bg-max",
        type=float,
        default=None,
        help="Upper x-axis / bin edge in mg/dL (default: 99th percentile of label values)",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Create a single plot with all models instead of grid",
    )
    parser.add_argument(
        "--no-scatter",
        action="store_true",
        help="Hide scatter points (grid mode only)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output plot file (auto-generated if not specified)",
    )
    add_model_filter_args(parser)
    add_figsize_arg(parser, default=(10.0, 6.0), help_suffix=' [combined mode]')
    add_figsize_arg(parser, name='--figsize-grid', default=(14.0, 10.0), help_suffix=' [grid mode]')
    add_figsize_arg(parser, name='--figsize-support', default=(10.0, 3.5), help_suffix=' [support bar chart]')

    args = parser.parse_args()

    if args.output is None:
        suffix = "_combined" if args.combined else "_grid"
        args.output = Path(f"error_vs_label_{args.horizon}min{suffix}.png")

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        return 1

    # Load data
    print(f"Loading data from {args.input}...")
    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df):,} rows")

    keep_models, color_by = apply_model_filter(df['model'].unique().tolist(), args)
    df = df[df['model'].isin(keep_models)]
    print(f"Models: {', '.join(keep_models)}")

    # Convert horizon to timestep
    timestep = (args.horizon // 5) - 1
    print(f"Using {args.horizon}min horizon (timestep {timestep})")

    # Compute error by label
    print("Computing prediction errors...")
    error_df = compute_error_by_label(df, timestep)
    print(f"Computed errors for {len(error_df):,} predictions")

    # Plot
    print(f"Generating plot with {args.n_bins} bins...")
    if args.combined:
        plot_rmse_vs_label_combined(
            error_df, args.horizon, args.output,
            figsize=tuple(args.figsize),
            n_bins=args.n_bins, color_by=color_by,
            bg_min=args.bg_min, bg_max=args.bg_max,
        )
    else:
        plot_rmse_vs_label(
            error_df, args.horizon, args.output,
            figsize=tuple(args.figsize_grid),
            n_bins=args.n_bins,
            show_scatter=not args.no_scatter,
            bg_min=args.bg_min, bg_max=args.bg_max,
        )

    # Always emit a companion support bar chart alongside the main figure.
    support_output = args.output.with_stem(args.output.stem + '_support')
    plot_label_support(
        error_df, args.horizon, support_output,
        figsize=tuple(args.figsize_support),
        n_bins=args.n_bins,
        bg_min=args.bg_min, bg_max=args.bg_max,
    )

    return 0


if __name__ == "__main__":
    exit(main())
