#!/usr/bin/env python3
"""
Plot RMSE as a heatmap with reference BG bins on x-axis.
Supports two modes:
  - bg-horizon: Y-axis is time horizon (5-60 min)
  - bg-cgm-std: Y-axis is CGM variability (std) bins, for a single time horizon

Generates one heatmap per model.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

from model_styles import add_model_filter_args, apply_model_filter


# Default BG bins
DEFAULT_BG_BINS = list(range(40, 401, 20))  # 40-60, 60-80, ..., 380-400

# Clinical BG bins (mg/dL) - fine granularity
CLINICAL_BG_BINS = [0, 40, 54, 70, 90, 110, 130, 150, 180, 210, 250, 300, 350, 500]
CLINICAL_BG_LABELS = ['<40', '40-54', '54-70', '70-90', '90-110', '110-130', '130-150', '150-180', '180-210', '210-250', '250-300', '300-350', '>350']

# CGM std bins (mg/dL) - fine granularity
CGM_STD_BINS = [0, 10, 18, 26, 34, 42, 50, 60, 75, 95, 200]
CGM_STD_LABELS = ['<10', '10-18', '18-26', '26-34', '34-42', '42-50', '50-60', '60-75', '75-95', '>95']


def compute_rmse_grid(
    df: pd.DataFrame,
    bg_bins: list,
    min_count: int = 10,
) -> dict:
    """
    Compute RMSE for each model, BG bin, and timestep.

    Args:
        df: DataFrame with model, label_t0-t11, pred_t0-t11 columns
        bg_bins: Bin edges for reference BG values
        min_count: Minimum samples per cell to include

    Returns:
        Dict with 'models', 'bg_bin_centers', 'time_horizons', 'rmse_grids'
    """
    models = sorted(df['model'].unique())
    n_bins = len(bg_bins) - 1
    n_timesteps = 12
    time_horizons = [(t + 1) * 5 for t in range(n_timesteps)]
    bg_bin_centers = [(bg_bins[i] + bg_bins[i + 1]) / 2 for i in range(n_bins)]
    bg_bin_labels = [f'{bg_bins[i]}-{bg_bins[i+1]}' for i in range(n_bins)]

    rmse_grids = {}

    for model in models:
        model_df = df[df['model'] == model]
        grid = np.full((n_timesteps, n_bins), np.nan)

        for t in range(n_timesteps):
            label_col = f'label_t{t}'
            pred_col = f'pred_t{t}'

            # Get label values and squared errors
            labels = model_df[label_col].values
            squared_errors = (model_df[pred_col] - model_df[label_col]) ** 2

            # Bin by label value
            bin_indices = np.digitize(labels, bg_bins) - 1
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)

            for b in range(n_bins):
                mask = bin_indices == b
                count = mask.sum()
                if count >= min_count:
                    mse = squared_errors[mask].mean()
                    grid[t, b] = np.sqrt(mse)

        rmse_grids[model] = grid

    return {
        'models': models,
        'bg_bin_centers': bg_bin_centers,
        'bg_bin_labels': bg_bin_labels,
        'time_horizons': time_horizons,
        'rmse_grids': rmse_grids,
    }


def compute_rmse_grid_bg_cgm_std(
    df: pd.DataFrame,
    timestep: int,
    bg_bins: list,
    bg_labels: list,
    cgm_std_bins: list,
    cgm_std_labels: list,
    min_count: int = 10,
) -> dict:
    """
    Compute RMSE for each model, BG bin, and CGM std bin at a single timestep.

    Args:
        df: DataFrame with model, cgm_std, label_t*, pred_t* columns
        timestep: Timestep index (0-11)
        bg_bins: Bin edges for reference BG values
        bg_labels: Labels for BG bins
        cgm_std_bins: Bin edges for CGM std values
        cgm_std_labels: Labels for CGM std bins
        min_count: Minimum samples per cell to include

    Returns:
        Dict with 'models', 'bg_labels', 'cgm_std_labels', 'rmse_grids', 'count_grids'
    """
    models = sorted(df['model'].unique())
    label_col = f'label_t{timestep}'
    pred_col = f'pred_t{timestep}'
    n_bg_bins = len(bg_bins) - 1
    n_std_bins = len(cgm_std_bins) - 1

    rmse_grids = {}
    count_grids = {}

    for model in models:
        model_df = df[df['model'] == model].copy()
        grid = np.full((n_std_bins, n_bg_bins), np.nan)
        count_grid = np.zeros((n_std_bins, n_bg_bins), dtype=int)

        # Get values
        bg_values = model_df[label_col].values
        std_values = model_df['cgm_std'].values
        squared_errors = (model_df[pred_col] - model_df[label_col]) ** 2

        # Bin both dimensions
        bg_bin_indices = np.digitize(bg_values, bg_bins) - 1
        bg_bin_indices = np.clip(bg_bin_indices, 0, n_bg_bins - 1)

        std_bin_indices = np.digitize(std_values, cgm_std_bins) - 1
        std_bin_indices = np.clip(std_bin_indices, 0, n_std_bins - 1)

        for std_idx in range(n_std_bins):
            for bg_idx in range(n_bg_bins):
                mask = (bg_bin_indices == bg_idx) & (std_bin_indices == std_idx)
                count = mask.sum()
                count_grid[std_idx, bg_idx] = count
                if count >= min_count:
                    mse = squared_errors[mask].mean()
                    grid[std_idx, bg_idx] = np.sqrt(mse)

        rmse_grids[model] = grid
        count_grids[model] = count_grid

    return {
        'models': models,
        'bg_labels': bg_labels,
        'cgm_std_labels': cgm_std_labels,
        'rmse_grids': rmse_grids,
        'count_grids': count_grids,
    }


def plot_rmse_heatmaps_bg_cgm_std(
    results: dict,
    horizon_minutes: int,
    output_path: Path,
    figsize: tuple = (18, 14),
    dpi: int = 150,
    cmap: str = 'YlOrRd',
    annotate: bool = True,
) -> None:
    """
    Create heatmap grid showing RMSE by BG bin and CGM std bin.

    Args:
        results: Dict from compute_rmse_grid_bg_cgm_std
        horizon_minutes: Prediction horizon in minutes (for title)
        output_path: Path to save the plot
        figsize: Figure size in inches
        dpi: Resolution for saved figure
        cmap: Colormap name
        annotate: Whether to annotate cells with RMSE values
    """
    models = results['models']
    bg_labels = results['bg_labels']
    cgm_std_labels = results['cgm_std_labels']
    rmse_grids = results['rmse_grids']
    count_grids = results['count_grids']

    n_models = len(models)
    cols = 3
    rows = (n_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()

    # Get global min/max for shared colorbar
    all_values = np.concatenate([g.flatten() for g in rmse_grids.values()])
    all_values = all_values[~np.isnan(all_values)]
    if len(all_values) > 0:
        vmin = np.percentile(all_values, 5)
        vmax = np.percentile(all_values, 95)
    else:
        vmin, vmax = 0, 1

    for idx, model in enumerate(models):
        ax = axes[idx]
        grid = rmse_grids[model]
        counts = count_grids[model]

        im = ax.imshow(
            grid,
            aspect='auto',
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            origin='lower',
        )

        # Annotate cells with RMSE values
        if annotate:
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    val = grid[i, j]
                    count = counts[i, j]
                    if not np.isnan(val):
                        # Choose text color based on background
                        text_color = 'white' if val > (vmin + vmax) / 2 else 'black'
                        ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                                fontsize=7, color=text_color, fontweight='bold')

        # Set tick labels
        ax.set_xticks(range(len(bg_labels)))
        ax.set_xticklabels(bg_labels, rotation=0, ha='center', fontsize=8)

        ax.set_yticks(range(len(cgm_std_labels)))
        ax.set_yticklabels(cgm_std_labels, fontsize=8)

        ax.set_title(model, fontsize=11, fontweight='bold')
        ax.set_xlabel('Reference BG (mg/dL)', fontsize=10)
        ax.set_ylabel('CGM Std Dev (mg/dL)', fontsize=10)

    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    # Add shared colorbar
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('RMSE (mg/dL)', fontsize=11)

    fig.suptitle(
        f'RMSE by Reference BG and CGM Variability at {horizon_minutes}min Horizon',
        fontsize=14,
        fontweight='bold',
    )
    plt.tight_layout(rect=[0, 0, 0.88, 0.96])
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved heatmap to {output_path}")


def plot_rmse_heatmaps(
    results: dict,
    output_path: Path,
    figsize: tuple = (16, 12),
    dpi: int = 150,
    cmap: str = 'viridis',
    shared_colorbar: bool = True,
) -> None:
    """
    Create heatmap grid with one subplot per model.

    Args:
        results: Dict from compute_rmse_grid
        output_path: Path to save the plot
        figsize: Figure size in inches
        dpi: Resolution for saved figure
        cmap: Colormap name
        shared_colorbar: If True, use same color scale across all models
    """
    models = results['models']
    bg_bin_labels = results['bg_bin_labels']
    time_horizons = results['time_horizons']
    rmse_grids = results['rmse_grids']

    n_models = len(models)
    cols = 3
    rows = (n_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()

    # Get global min/max for shared colorbar
    if shared_colorbar:
        all_values = np.concatenate([g.flatten() for g in rmse_grids.values()])
        all_values = all_values[~np.isnan(all_values)]
        vmin = np.percentile(all_values, 5)
        vmax = np.percentile(all_values, 95)
    else:
        vmin, vmax = None, None

    for idx, model in enumerate(models):
        ax = axes[idx]
        grid = rmse_grids[model]

        im = ax.imshow(
            grid,
            aspect='auto',
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            origin='lower',
        )

        # Set tick labels
        # X-axis: BG bins (show every 2nd or 3rd label to avoid crowding)
        n_bg_bins = len(bg_bin_labels)
        step = max(1, n_bg_bins // 10)
        ax.set_xticks(range(0, n_bg_bins, step))
        ax.set_xticklabels([bg_bin_labels[i] for i in range(0, n_bg_bins, step)], rotation=45, ha='right', fontsize=8)

        # Y-axis: time horizons
        ax.set_yticks(range(len(time_horizons)))
        ax.set_yticklabels([f'{t}' for t in time_horizons], fontsize=9)

        ax.set_title(model, fontsize=11, fontweight='bold')

        if idx >= (rows - 1) * cols:
            ax.set_xlabel('Reference BG (mg/dL)', fontsize=10)
        if idx % cols == 0:
            ax.set_ylabel('Horizon (min)', fontsize=10)

        if not shared_colorbar:
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('RMSE', fontsize=9)

    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    # Add shared colorbar
    if shared_colorbar:
        fig.subplots_adjust(right=0.88)
        cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label('RMSE (mg/dL)', fontsize=11)

    fig.suptitle('RMSE by Reference BG and Prediction Horizon', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.88 if shared_colorbar else 1, 0.96])
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot RMSE heatmap by reference BG and time horizon or CGM variability"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("combined_results_new.parquet"),
        help="Input parquet file (default: combined_results_new.parquet)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=['bg-horizon', 'bg-cgm-std'],
        default='bg-horizon',
        help="Heatmap mode: bg-horizon (BG x time) or bg-cgm-std (BG x CGM std)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        choices=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
        default=None,
        help="Prediction horizon in minutes (required for bg-cgm-std mode)",
    )
    parser.add_argument(
        "--bin-size",
        type=int,
        default=20,
        help="Size of BG bins in mg/dL for bg-horizon mode (default: 20)",
    )
    parser.add_argument(
        "--bg-min",
        type=int,
        default=40,
        help="Minimum BG value for bins (default: 40)",
    )
    parser.add_argument(
        "--bg-max",
        type=int,
        default=400,
        help="Maximum BG value for bins (default: 400)",
    )
    parser.add_argument(
        "--clinical-bins",
        action="store_true",
        help="Use clinical BG bins (hypo/in-range/hyper) instead of uniform bins",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="viridis",
        help="Colormap (default: viridis, try YlOrRd for bg-cgm-std)",
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Enable RMSE value annotations on cells (bg-cgm-std mode only)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output plot file (auto-generated if not specified)",
    )
    add_model_filter_args(parser)

    args = parser.parse_args()

    # Validate arguments
    if args.mode == 'bg-cgm-std' and args.horizon is None:
        parser.error("--horizon is required for bg-cgm-std mode")

    # Set default output path
    if args.output is None:
        if args.mode == 'bg-horizon':
            args.output = Path("rmse_heatmap_bg_horizon.png")
        else:
            args.output = Path(f"rmse_heatmap_bg_cgm_std_{args.horizon}min.png")

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        return 1

    # Load data
    print(f"Loading data from {args.input}...")
    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df):,} rows")

    keep, _ = apply_model_filter(df['model'].unique().tolist(), args)
    df = df[df['model'].isin(keep)]
    print(f"Models: {', '.join(keep)}")

    if args.mode == 'bg-horizon':
        # Original mode: BG x time horizon
        bg_bins = list(range(args.bg_min, args.bg_max + 1, args.bin_size))
        print(f"BG bins: {args.bg_min} to {args.bg_max} by {args.bin_size} ({len(bg_bins) - 1} bins)")

        print("Computing RMSE grid...")
        results = compute_rmse_grid(df, bg_bins)

        plot_rmse_heatmaps(results, args.output, cmap=args.cmap)

    else:  # bg-cgm-std mode
        timestep = (args.horizon // 5) - 1

        # Check for cgm_std column
        if 'cgm_std' not in df.columns:
            print("Error: Column 'cgm_std' not found in data.")
            return 1

        # Use clinical bins or uniform bins
        if args.clinical_bins:
            bg_bins = CLINICAL_BG_BINS
            bg_labels = CLINICAL_BG_LABELS
        else:
            bg_bins = list(range(args.bg_min, args.bg_max + 1, args.bin_size))
            bg_labels = [f'{bg_bins[i]}-{bg_bins[i+1]}' for i in range(len(bg_bins) - 1)]

        print(f"Mode: BG x CGM Std at {args.horizon}min horizon (timestep {timestep})")
        print(f"BG bins: {len(bg_bins) - 1} bins")
        print(f"CGM Std bins: {CGM_STD_LABELS}")

        # Print data stats
        label_col = f'label_t{timestep}'
        print(f"\nReference BG range: {df[label_col].min():.1f} - {df[label_col].max():.1f} mg/dL")
        print(f"CGM Std range: {df['cgm_std'].min():.1f} - {df['cgm_std'].max():.1f} mg/dL")

        print("\nComputing RMSE grid...")
        results = compute_rmse_grid_bg_cgm_std(
            df,
            timestep,
            bg_bins,
            bg_labels,
            CGM_STD_BINS,
            CGM_STD_LABELS,
        )

        # Print sample counts
        print("\nSample counts per bin (summed across models):")
        total_counts = sum(results['count_grids'].values())
        for i, std_label in enumerate(CGM_STD_LABELS):
            row_counts = [str(total_counts[i, j]) for j in range(len(bg_labels))]
            print(f"  {std_label}: {', '.join(row_counts)}")

        # Use YlOrRd as default for this mode if not specified
        cmap = args.cmap if args.cmap != 'viridis' else 'YlOrRd'

        plot_rmse_heatmaps_bg_cgm_std(
            results,
            args.horizon,
            args.output,
            cmap=cmap,
            annotate=args.annotate,
        )

    return 0


if __name__ == "__main__":
    exit(main())
