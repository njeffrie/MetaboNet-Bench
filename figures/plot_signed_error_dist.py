#!/usr/bin/env python3
"""
Plot distribution of signed errors (pred - label) for each model at each timescale.
Shows bias and error spread over prediction horizons.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from model_styles import add_model_filter_args, apply_model_filter


def compute_signed_errors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute signed errors for all timesteps.

    Args:
        df: DataFrame with model, label_t0-t11, pred_t0-t11 columns

    Returns:
        Long-format DataFrame with model, horizon, signed_error columns
    """
    records = []

    for t in range(12):
        label_col = f'label_t{t}'
        pred_col = f'pred_t{t}'
        horizon = (t + 1) * 5

        errors = df[pred_col] - df[label_col]

        for model in df['model'].unique():
            model_mask = df['model'] == model
            model_errors = errors[model_mask].values

            for err in model_errors:
                records.append({
                    'model': model,
                    'horizon': horizon,
                    'signed_error': err,
                })

    return pd.DataFrame(records)


def compute_error_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute error statistics for each model and timestep.

    Args:
        df: DataFrame with model, label_t0-t11, pred_t0-t11 columns

    Returns:
        DataFrame with model, horizon, mean, std, median, q25, q75
    """
    records = []

    for model in sorted(df['model'].unique()):
        model_df = df[df['model'] == model]

        for t in range(12):
            label_col = f'label_t{t}'
            pred_col = f'pred_t{t}'
            horizon = (t + 1) * 5

            errors = (model_df[pred_col] - model_df[label_col]).values

            records.append({
                'model': model,
                'horizon': horizon,
                'mean': np.mean(errors),
                'std': np.std(errors),
                'median': np.median(errors),
                'q25': np.percentile(errors, 25),
                'q75': np.percentile(errors, 75),
                'q05': np.percentile(errors, 5),
                'q95': np.percentile(errors, 95),
            })

    return pd.DataFrame(records)


def plot_error_ribbons(
    stats_df: pd.DataFrame,
    output_path: Path,
    figsize: tuple = (16, 12),
    dpi: int = 150,
) -> None:
    """
    Plot signed error distribution as ribbons (mean ± std, with IQR).
    One subplot per model in a grid layout.

    Args:
        stats_df: DataFrame with model, horizon, mean, std, q25, q75, etc.
        output_path: Path to save the plot
        figsize: Figure size in inches
        dpi: Resolution for saved figure
    """
    models = sorted(stats_df['model'].unique())
    n_models = len(models)
    cols = 3
    rows = (n_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize, sharex=True, sharey=True)
    axes = axes.flatten()

    # Get global y-axis limits for consistent scaling
    y_min = stats_df['q05'].min()
    y_max = stats_df['q95'].max()
    y_margin = (y_max - y_min) * 0.1
    y_min -= y_margin
    y_max += y_margin

    for idx, model in enumerate(models):
        ax = axes[idx]
        model_data = stats_df[stats_df['model'] == model].sort_values('horizon')
        horizons = model_data['horizon'].values
        means = model_data['mean'].values
        q25 = model_data['q25'].values
        q75 = model_data['q75'].values
        q05 = model_data['q05'].values
        q95 = model_data['q95'].values

        # Plot 5-95 percentile range (light)
        ax.fill_between(horizons, q05, q95, color='steelblue', alpha=0.2, label='5-95%')

        # Plot IQR (medium)
        ax.fill_between(horizons, q25, q75, color='steelblue', alpha=0.4, label='IQR')

        # Plot mean line
        ax.plot(horizons, means, color='steelblue', linewidth=2, marker='o', markersize=4, label='Mean',
                markeredgecolor='black', markeredgewidth=0.5)

        # Add zero line
        ax.axhline(y=0, color='red', linestyle='-', linewidth=1.5, alpha=0.7)

        ax.set_title(model, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(y_min, y_max)

        if idx >= (rows - 1) * cols:
            ax.set_xlabel('Horizon (min)', fontsize=10)
        if idx % cols == 0:
            ax.set_ylabel('Signed Error (mg/dL)', fontsize=10)

        ax.set_xticks([(t + 1) * 5 for t in range(12)])

    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    # Add legend to first subplot
    axes[0].legend(loc='upper left', fontsize=8)

    fig.suptitle(
        'Signed Error Distribution by Model and Horizon\n(line=mean, dark band=IQR, light band=5-95%, red=zero)',
        fontsize=14,
        fontweight='bold',
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved ribbon plot to {output_path}")


def plot_error_boxplots(
    df: pd.DataFrame,
    output_path: Path,
    figsize: tuple = (16, 10),
    dpi: int = 150,
    sample_size: int = 50000,
) -> None:
    """
    Plot signed error distributions as box plots, one row per model.

    Args:
        df: DataFrame with model, label_t0-t11, pred_t0-t11 columns
        output_path: Path to save the plot
        figsize: Figure size in inches
        dpi: Resolution for saved figure
        sample_size: Number of samples per model (for performance)
    """
    models = sorted(df['model'].unique())
    n_models = len(models)
    n_timesteps = 12

    fig, axes = plt.subplots(n_models, 1, figsize=figsize, sharex=True, sharey=True)
    if n_models == 1:
        axes = [axes]

    horizons = [(t + 1) * 5 for t in range(n_timesteps)]

    for idx, model in enumerate(models):
        ax = axes[idx]
        model_df = df[df['model'] == model]

        # Subsample for performance
        if len(model_df) > sample_size:
            model_df = model_df.sample(n=sample_size, random_state=42)

        # Collect errors for each timestep
        error_data = []
        for t in range(n_timesteps):
            label_col = f'label_t{t}'
            pred_col = f'pred_t{t}'
            errors = (model_df[pred_col] - model_df[label_col]).values
            error_data.append(errors)

        # Box plot
        bp = ax.boxplot(
            error_data,
            positions=horizons,
            widths=3,
            patch_artist=True,
            showfliers=False,  # Hide outliers for cleaner plot
        )

        # Style boxes
        for patch in bp['boxes']:
            patch.set_facecolor('steelblue')
            patch.set_alpha(0.6)

        ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
        ax.set_ylabel(model, fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

    axes[-1].set_xlabel('Prediction Horizon (minutes)', fontsize=12)
    axes[-1].set_xticks(horizons)

    fig.suptitle('Signed Error Distribution by Model and Horizon', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved box plot to {output_path}")


def plot_error_histograms(
    df: pd.DataFrame,
    output_path: Path,
    figsize: tuple = (16, 14),
    dpi: int = 150,
    sample_size: int = 100000,
    n_bins: int = 50,
) -> None:
    """
    Plot signed error distributions as histograms, grid of model x timestep.

    Args:
        df: DataFrame with model, label_t0-t11, pred_t0-t11 columns
        output_path: Path to save the plot
        figsize: Figure size in inches
        dpi: Resolution for saved figure
        sample_size: Number of samples per model (for performance)
        n_bins: Number of histogram bins
    """
    models = sorted(df['model'].unique())
    n_models = len(models)
    timesteps = [0, 2, 5, 8, 11]  # 5min, 15min, 30min, 45min, 60min
    n_timesteps = len(timesteps)

    fig, axes = plt.subplots(n_models, n_timesteps, figsize=figsize, sharex=True, sharey=True)

    # Determine global x-axis limits
    all_errors = []
    for model in models:
        model_df = df[df['model'] == model]
        if len(model_df) > sample_size:
            model_df = model_df.sample(n=sample_size, random_state=42)
        for t in timesteps:
            errors = (model_df[f'pred_t{t}'] - model_df[f'label_t{t}']).values
            all_errors.extend(errors)

    x_limit = np.percentile(np.abs(all_errors), 99)
    bin_edges = np.linspace(-x_limit, x_limit, n_bins + 1)

    for row, model in enumerate(models):
        model_df = df[df['model'] == model]
        if len(model_df) > sample_size:
            model_df = model_df.sample(n=sample_size, random_state=42)

        for col, t in enumerate(timesteps):
            ax = axes[row, col]
            horizon = (t + 1) * 5

            errors = (model_df[f'pred_t{t}'] - model_df[f'label_t{t}']).values

            # Plot histogram
            ax.hist(errors, bins=bin_edges, color='steelblue', alpha=0.7, edgecolor='white', linewidth=0.5)

            # Mark zero with prominent vertical line
            ax.axvline(x=0, color='red', linestyle='-', linewidth=2, label='Zero')

            # Add mean line
            mean_err = np.mean(errors)
            ax.axvline(x=mean_err, color='orange', linestyle='--', linewidth=1.5, label=f'Mean: {mean_err:.1f}')

            # Labels
            if row == 0:
                ax.set_title(f'{horizon} min', fontsize=11, fontweight='bold')
            if col == 0:
                ax.set_ylabel(model, fontsize=10, fontweight='bold')
            if row == n_models - 1:
                ax.set_xlabel('Signed Error (mg/dL)', fontsize=9)

            # Stats annotation
            ax.text(
                0.95, 0.95,
                f'μ={mean_err:.1f}\nσ={np.std(errors):.1f}',
                transform=ax.transAxes,
                fontsize=8,
                va='top', ha='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            )

    # Add legend to first subplot
    axes[0, 0].legend(loc='upper left', fontsize=8)

    fig.suptitle('Signed Error Distribution (pred - label) by Model and Horizon\nRed line = 0, Orange dashed = Mean',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved histogram plot to {output_path}")


def plot_error_violins(
    df: pd.DataFrame,
    output_path: Path,
    figsize: tuple = (14, 10),
    dpi: int = 150,
    sample_size: int = 50000,
) -> None:
    """
    Plot signed error distributions as violin plots, one subplot per model.

    Args:
        df: DataFrame with model, label_t0-t11, pred_t0-t11 columns
        output_path: Path to save the plot
        figsize: Figure size in inches
        dpi: Resolution for saved figure
        sample_size: Number of samples per model (for performance)
    """
    models = sorted(df['model'].unique())
    n_models = len(models)
    cols = 3
    rows = (n_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize, sharex=True, sharey=True)
    axes = axes.flatten()

    horizons = [(t + 1) * 5 for t in range(12)]

    for idx, model in enumerate(models):
        ax = axes[idx]
        model_df = df[df['model'] == model]

        # Subsample for performance
        if len(model_df) > sample_size:
            model_df = model_df.sample(n=sample_size, random_state=42)

        # Collect errors for each timestep
        error_data = []
        for t in range(12):
            label_col = f'label_t{t}'
            pred_col = f'pred_t{t}'
            errors = (model_df[pred_col] - model_df[label_col]).values
            error_data.append(errors)

        # Violin plot
        vp = ax.violinplot(
            error_data,
            positions=horizons,
            widths=4,
            showmeans=True,
            showmedians=True,
        )

        # Style violins
        for pc in vp['bodies']:
            pc.set_facecolor('steelblue')
            pc.set_alpha(0.6)

        ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
        ax.set_title(model, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        if idx >= (rows - 1) * cols:
            ax.set_xlabel('Horizon (min)', fontsize=10)
        if idx % cols == 0:
            ax.set_ylabel('Signed Error (mg/dL)', fontsize=10)

    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('Signed Error Distribution by Model and Horizon', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved violin plot to {output_path}")


def plot_error_ridges(
    df: pd.DataFrame,
    output_path: Path,
    figsize: tuple = (16, 14),
    dpi: int = 150,
    sample_size: int = 50000,
) -> None:
    """
    Plot signed error distributions as ridge/joy plots (2.5D stacked KDE curves).

    Args:
        df: DataFrame with model, label_t0-t11, pred_t0-t11 columns
        output_path: Path to save the plot
        figsize: Figure size in inches
        dpi: Resolution for saved figure
        sample_size: Number of samples per model (for performance)
    """
    models = sorted(df['model'].unique())
    n_models = len(models)
    cols = 3
    rows = (n_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()

    horizons = [(t + 1) * 5 for t in range(12)]

    # Determine global x-axis limits across all models
    all_errors = []
    for model in models:
        model_df = df[df['model'] == model]
        if len(model_df) > sample_size:
            model_df = model_df.sample(n=sample_size, random_state=42)
        for t in range(12):
            errors = (model_df[f'pred_t{t}'] - model_df[f'label_t{t}']).values
            all_errors.extend(errors)

    x_limit = np.percentile(np.abs(all_errors), 99)
    x_grid = np.linspace(-x_limit, x_limit, 200)

    # Color map for timesteps
    colors = plt.cm.viridis(np.linspace(0, 1, 12))

    for idx, model in enumerate(models):
        ax = axes[idx]
        model_df = df[df['model'] == model]

        # Subsample for performance
        if len(model_df) > sample_size:
            model_df = model_df.sample(n=sample_size, random_state=42)

        # Scale factor for vertical offset between ridges
        scale = 0.015

        for t in range(12):
            horizon = horizons[t]
            errors = (model_df[f'pred_t{t}'] - model_df[f'label_t{t}']).values

            # Compute KDE
            try:
                kde = gaussian_kde(errors)
                y_kde = kde(x_grid)
            except Exception:
                # Fallback if KDE fails (e.g., singular matrix)
                y_kde = np.zeros_like(x_grid)

            # Normalize KDE to have similar visual heights
            if y_kde.max() > 0:
                y_kde = y_kde / y_kde.max()

            offset = t * scale * 4

            # Fill under the curve
            ax.fill_between(
                x_grid,
                offset,
                offset + y_kde * scale * 3,
                alpha=0.7,
                color=colors[t],
                label=f'{horizon} min' if idx == 0 else None,
            )

            # Draw outline
            ax.plot(x_grid, offset + y_kde * scale * 3, color='black', lw=0.5)

        # Red vertical line at x=0
        ax.axvline(x=0, color='red', linewidth=2, linestyle='-', zorder=10)

        ax.set_title(model, fontsize=11, fontweight='bold')
        ax.set_xlabel('Signed Error (mg/dL)', fontsize=10)
        ax.set_xlim(-x_limit, x_limit)

        # Hide y-axis ticks (offsets are arbitrary)
        ax.set_yticks([])
        ax.set_ylabel('Prediction Horizon →', fontsize=10)

    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    # Add legend to first subplot
    axes[0].legend(loc='upper right', fontsize=7, title='Horizon')

    fig.suptitle(
        'Signed Error Ridge Plot (KDE per Horizon)\nRed line = 0 error',
        fontsize=14,
        fontweight='bold',
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved ridge plot to {output_path}")


def plot_error_surface3d(
    df: pd.DataFrame,
    output_path: Path,
    figsize: tuple = (18, 14),
    dpi: int = 150,
    sample_size: int = 50000,
) -> None:
    """
    Plot signed error distributions as true 3D surface plots.

    Args:
        df: DataFrame with model, label_t0-t11, pred_t0-t11 columns
        output_path: Path to save the plot
        figsize: Figure size in inches
        dpi: Resolution for saved figure
        sample_size: Number of samples per model (for performance)
    """
    models = sorted(df['model'].unique())
    n_models = len(models)
    cols = 3
    rows = (n_models + cols - 1) // cols

    fig = plt.figure(figsize=figsize)

    horizons = np.array([(t + 1) * 5 for t in range(12)])

    # Determine global x-axis limits across all models
    all_errors = []
    for model in models:
        model_df = df[df['model'] == model]
        if len(model_df) > sample_size:
            model_df = model_df.sample(n=sample_size, random_state=42)
        for t in range(12):
            errors = (model_df[f'pred_t{t}'] - model_df[f'label_t{t}']).values
            all_errors.extend(errors)

    x_limit = np.percentile(np.abs(all_errors), 99)
    x_grid = np.linspace(-x_limit, x_limit, 100)

    for idx, model in enumerate(models):
        ax = fig.add_subplot(rows, cols, idx + 1, projection='3d')
        model_df = df[df['model'] == model]

        # Subsample for performance
        if len(model_df) > sample_size:
            model_df = model_df.sample(n=sample_size, random_state=42)

        # Build density matrix: rows=timesteps, cols=error values
        density_matrix = np.zeros((12, len(x_grid)))

        for t in range(12):
            errors = (model_df[f'pred_t{t}'] - model_df[f'label_t{t}']).values

            try:
                kde = gaussian_kde(errors)
                density_matrix[t, :] = kde(x_grid)
            except Exception:
                density_matrix[t, :] = 0

        # Create meshgrid
        X, Y = np.meshgrid(x_grid, horizons)
        Z = density_matrix

        # Plot surface
        surf = ax.plot_surface(
            X, Y, Z,
            cmap='viridis',
            alpha=0.8,
            edgecolor='none',
            antialiased=True,
        )

        # Red plane/line at x=0
        z_max = Z.max() if Z.max() > 0 else 1
        ax.plot([0, 0], [horizons[0], horizons[-1]], [0, 0], color='red', lw=3, zorder=10)
        ax.plot([0, 0], [horizons[0], horizons[-1]], [z_max, z_max], color='red', lw=2, linestyle='--', alpha=0.7)

        ax.set_xlabel('Error (mg/dL)', fontsize=9, labelpad=5)
        ax.set_ylabel('Horizon (min)', fontsize=9, labelpad=5)
        ax.set_zlabel('Density', fontsize=9, labelpad=5)
        ax.set_title(model, fontsize=11, fontweight='bold')

        # Adjust view angle for better visualization
        ax.view_init(elev=25, azim=-60)

    # Hide unused subplots
    for idx in range(n_models, rows * cols):
        fig.add_subplot(rows, cols, idx + 1).set_visible(False)

    fig.suptitle(
        'Signed Error 3D Surface (KDE Density)\nRed line marks x=0',
        fontsize=14,
        fontweight='bold',
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved 3D surface plot to {output_path}")


def plot_error_waterfall(
    df: pd.DataFrame,
    output_path: Path,
    figsize: tuple = (18, 14),
    dpi: int = 150,
    sample_size: int = 50000,
) -> None:
    """
    Plot signed error distributions as 3D waterfall (polygon slices).

    Args:
        df: DataFrame with model, label_t0-t11, pred_t0-t11 columns
        output_path: Path to save the plot
        figsize: Figure size in inches
        dpi: Resolution for saved figure
        sample_size: Number of samples per model (for performance)
    """
    models = sorted(df['model'].unique())
    n_models = len(models)
    cols = 3
    rows = (n_models + cols - 1) // cols

    fig = plt.figure(figsize=figsize)

    horizons = [(t + 1) * 5 for t in range(12)]

    # Determine global x-axis limits across all models
    all_errors = []
    for model in models:
        model_df = df[df['model'] == model]
        if len(model_df) > sample_size:
            model_df = model_df.sample(n=sample_size, random_state=42)
        for t in range(12):
            errors = (model_df[f'pred_t{t}'] - model_df[f'label_t{t}']).values
            all_errors.extend(errors)

    x_limit = np.percentile(np.abs(all_errors), 99)
    x_grid = np.linspace(-x_limit, x_limit, 100)

    # Color map for timesteps
    colors = plt.cm.viridis(np.linspace(0, 1, 12))

    for idx, model in enumerate(models):
        ax = fig.add_subplot(rows, cols, idx + 1, projection='3d')
        model_df = df[df['model'] == model]

        # Subsample for performance
        if len(model_df) > sample_size:
            model_df = model_df.sample(n=sample_size, random_state=42)

        # Find global max for normalization
        all_kdes = []
        for t in range(12):
            errors = (model_df[f'pred_t{t}'] - model_df[f'label_t{t}']).values
            try:
                kde = gaussian_kde(errors)
                all_kdes.append(kde(x_grid))
            except Exception:
                all_kdes.append(np.zeros_like(x_grid))

        z_max = max(kde_vals.max() for kde_vals in all_kdes) if all_kdes else 1

        # Plot each timestep as a polygon slice
        for t in range(12):
            horizon = horizons[t]
            kde_vals = all_kdes[t]

            # Build polygon vertices: (x, y=horizon, z=density)
            # Start from bottom-left, trace the curve, end at bottom-right
            verts = []

            # Bottom-left corner
            verts.append((x_grid[0], horizon, 0))

            # Trace the KDE curve
            for i, x in enumerate(x_grid):
                verts.append((x, horizon, kde_vals[i]))

            # Bottom-right corner
            verts.append((x_grid[-1], horizon, 0))

            # Create polygon collection
            poly = Poly3DCollection(
                [verts],
                alpha=0.7,
                facecolor=colors[t],
                edgecolor='black',
                linewidth=0.3,
            )
            ax.add_collection3d(poly)

        # Red line at x=0 along the horizon axis
        ax.plot(
            [0, 0],
            [horizons[0], horizons[-1]],
            [0, 0],
            color='red',
            lw=3,
            zorder=10,
        )

        ax.set_xlabel('Error (mg/dL)', fontsize=9, labelpad=5)
        ax.set_ylabel('Horizon (min)', fontsize=9, labelpad=5)
        ax.set_zlabel('Density', fontsize=9, labelpad=5)
        ax.set_title(model, fontsize=11, fontweight='bold')

        # Set axis limits
        ax.set_xlim(-x_limit, x_limit)
        ax.set_ylim(horizons[0], horizons[-1])
        ax.set_zlim(0, z_max * 1.1)

        # Adjust view angle
        ax.view_init(elev=20, azim=-50)

    # Hide unused subplots
    for idx in range(n_models, rows * cols):
        fig.add_subplot(rows, cols, idx + 1).set_visible(False)

    fig.suptitle(
        'Signed Error 3D Waterfall (KDE Slices per Horizon)\nRed line marks x=0',
        fontsize=14,
        fontweight='bold',
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved 3D waterfall plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot signed error distribution for each model at each timescale"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("combined_results_new.parquet"),
        help="Input parquet file (default: combined_results_new.parquet)",
    )
    parser.add_argument(
        "--plot-type",
        type=str,
        choices=['ribbon', 'boxplot', 'violin', 'histogram', 'ridge', 'surface3d', 'waterfall'],
        default='histogram',
        help="Type of plot: ribbon, boxplot, violin, histogram, ridge (2.5D KDE), surface3d (3D density surface), or waterfall (3D slices)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output plot file (auto-generated if not specified)",
    )
    add_model_filter_args(parser)

    args = parser.parse_args()

    if args.output is None:
        args.output = Path(f"signed_error_{args.plot_type}.png")

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

    # Generate plot based on type
    if args.plot_type == 'ribbon':
        print("Computing error statistics...")
        stats_df = compute_error_stats(df)

        # Print summary
        print("\nMean signed error by model and horizon:")
        pivot = stats_df.pivot(index='model', columns='horizon', values='mean')
        print(pivot.round(2).to_string())

        plot_error_ribbons(stats_df, args.output)

    elif args.plot_type == 'boxplot':
        plot_error_boxplots(df, args.output)

    elif args.plot_type == 'violin':
        plot_error_violins(df, args.output)

    elif args.plot_type == 'histogram':
        plot_error_histograms(df, args.output)

    elif args.plot_type == 'ridge':
        plot_error_ridges(df, args.output)

    elif args.plot_type == 'surface3d':
        plot_error_surface3d(df, args.output)

    elif args.plot_type == 'waterfall':
        plot_error_waterfall(df, args.output)

    return 0


if __name__ == "__main__":
    exit(main())
