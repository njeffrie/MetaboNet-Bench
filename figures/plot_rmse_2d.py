#!/usr/bin/env python3
"""
Plot RMSE as contour plot with two variables as axes.
Creates a grid with one subplot per model.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.interpolate import griddata

from model_styles import add_model_filter_args, apply_model_filter


# Display names for variables
VARIABLE_DISPLAY_NAMES = {
    'age': 'Age (years)',
    'cgm_mean': 'Mean CGM (mg/dL)',
    'cgm_std': 'CGM Std Dev (mg/dL)',
    'weight': 'Weight (lbs)',
    'height': 'Height (ft)',
    'age_of_diagnosis': 'Age of Diagnosis (years)',
}


def compute_patient_rmse(df: pd.DataFrame, variables: list, timestep: int = None) -> pd.DataFrame:
    """
    Compute RMSE per patient.

    Args:
        df: DataFrame with user_id, dataset, model, variables, label_t*, pred_t* columns
        variables: List of variables to include in output
        timestep: If specified, compute RMSE at this timestep only (0-11).
                  If None, compute RMSE across all timesteps.

    Returns:
        DataFrame with user_id, dataset, model, variables, rmse columns
    """
    if timestep is not None:
        # Single timestep
        label_col = f'label_t{timestep}'
        pred_col = f'pred_t{timestep}'
        df['squared_error'] = (df[label_col] - df[pred_col]) ** 2
    else:
        # All timesteps - compute mean squared error across all 12 timesteps
        label_cols = [f'label_t{t}' for t in range(12)]
        pred_cols = [f'pred_t{t}' for t in range(12)]

        # Compute squared error for each timestep and average
        squared_errors = []
        for label_col, pred_col in zip(label_cols, pred_cols):
            squared_errors.append((df[label_col] - df[pred_col]) ** 2)
        df['squared_error'] = np.mean(squared_errors, axis=0)

    # Build aggregation dict
    agg_dict = {'squared_error': 'mean'}
    for var in variables:
        agg_dict[var] = 'first'

    # Group by patient (user_id, dataset, model) and compute mean squared error, then RMSE
    patient_rmse = df.groupby(['user_id', 'dataset', 'model']).agg(agg_dict).reset_index()

    patient_rmse['rmse'] = np.sqrt(patient_rmse['squared_error'])
    patient_rmse = patient_rmse.drop(columns=['squared_error'])

    return patient_rmse


def plot_rmse_2d(
    patient_df: pd.DataFrame,
    x_var: str,
    y_var: str,
    output_path: Path,
    horizon_minutes: int = None,
    figsize: tuple = (14, 10),
    dpi: int = 150,
    cmap: str = 'viridis',
    n_bins: int = 50,
    n_levels: int = 15,
) -> None:
    """
    Create 2D contour plots with RMSE as color, one per model.

    Args:
        patient_df: DataFrame with user_id, model, x_var, y_var, rmse columns
        x_var: Name of the x-axis variable
        y_var: Name of the y-axis variable
        output_path: Path to save the plot
        horizon_minutes: Prediction horizon in minutes (for title), None if all timesteps
        figsize: Figure size in inches
        dpi: Resolution for saved figure
        cmap: Colormap name
        n_bins: Number of bins for grid interpolation
        n_levels: Number of contour levels
    """
    models = sorted(patient_df['model'].unique())
    n_models = len(models)
    cols = 3
    rows = (n_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()

    x_label = VARIABLE_DISPLAY_NAMES.get(x_var, x_var)
    y_label = VARIABLE_DISPLAY_NAMES.get(y_var, y_var)

    # Get global RMSE range for consistent colorbar
    valid_data = patient_df.dropna(subset=[x_var, y_var, 'rmse'])
    vmin = valid_data['rmse'].quantile(0.05)
    vmax = valid_data['rmse'].quantile(0.95)

    # Get global x/y range
    x_min, x_max = valid_data[x_var].quantile(0.01), valid_data[x_var].quantile(0.99)
    y_min, y_max = valid_data[y_var].quantile(0.01), valid_data[y_var].quantile(0.99)

    # Create grid for interpolation
    xi = np.linspace(x_min, x_max, n_bins)
    yi = np.linspace(y_min, y_max, n_bins)
    Xi, Yi = np.meshgrid(xi, yi)

    contour_plots = []

    for idx, model in enumerate(models):
        ax = axes[idx]
        model_data = patient_df[patient_df['model'] == model].dropna(subset=[x_var, y_var, 'rmse'])

        if len(model_data) > 10:
            # Get data points
            x = model_data[x_var].values
            y = model_data[y_var].values
            z = model_data['rmse'].values

            # Interpolate to grid
            Zi = griddata((x, y), z, (Xi, Yi), method='linear')

            # Create filled contour plot
            contour = ax.contourf(
                Xi, Yi, Zi,
                levels=n_levels,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                extend='both',
            )
            contour_plots.append(contour)

            # Add contour lines
            ax.contour(Xi, Yi, Zi, levels=n_levels, colors='white', linewidths=0.3, alpha=0.5)

            # Add correlation annotation
            corr_x = model_data[x_var].corr(model_data['rmse'])
            corr_y = model_data[y_var].corr(model_data['rmse'])
            ax.text(
                0.05, 0.95,
                f'r(x,rmse)={corr_x:.2f}\nr(y,rmse)={corr_y:.2f}\nn={len(model_data):,}',
                transform=ax.transAxes,
                fontsize=8,
                va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            )

        ax.set_title(model, fontsize=11, fontweight='bold')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        if idx >= (rows - 1) * cols:
            ax.set_xlabel(x_label, fontsize=10)
        if idx % cols == 0:
            ax.set_ylabel(y_label, fontsize=10)

    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    # Add colorbar
    if contour_plots:
        sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.02, pad=0.02)
        cbar.set_label('RMSE (mg/dL)', fontsize=11)

    if horizon_minutes:
        title = f'RMSE by {x_label} and {y_label} ({horizon_minutes}min Horizon)'
    else:
        title = f'RMSE by {x_label} and {y_label} (all horizons averaged)'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.95, 0.96])
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot RMSE as color on a 2D scatter plot"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("combined_results_new.parquet"),
        help="Input parquet file (default: combined_results_new.parquet)",
    )
    parser.add_argument(
        "--x-var", "-x",
        type=str,
        default="cgm_std",
        choices=['age', 'cgm_mean', 'cgm_std', 'weight', 'height', 'age_of_diagnosis'],
        help="Variable for x-axis (default: cgm_std)",
    )
    parser.add_argument(
        "--y-var", "-y",
        type=str,
        default="cgm_mean",
        choices=['age', 'cgm_mean', 'cgm_std', 'weight', 'height', 'age_of_diagnosis'],
        help="Variable for y-axis (default: cgm_mean)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        choices=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
        default=None,
        help="Prediction horizon in minutes. If not specified, averages across all horizons.",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="viridis",
        help="Colormap for RMSE (default: viridis)",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=50,
        help="Number of bins for grid interpolation (default: 50)",
    )
    parser.add_argument(
        "--n-levels",
        type=int,
        default=15,
        help="Number of contour levels (default: 15)",
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
        horizon_str = f"_{args.horizon}min" if args.horizon else "_all"
        args.output = Path(f"rmse_2d_{args.x_var}_vs_{args.y_var}{horizon_str}.png")

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

    # Check if variables exist
    for var in [args.x_var, args.y_var]:
        if var not in df.columns:
            print(f"Error: Variable '{var}' not found in data.")
            print(f"Available columns: {list(df.columns)}")
            return 1

    # Convert horizon to timestep
    timestep = None
    if args.horizon:
        timestep = (args.horizon // 5) - 1
        print(f"Using {args.horizon}min horizon (timestep {timestep})")
    else:
        print("Averaging RMSE across all 12 timesteps (5-60min)")

    # Compute patient-level RMSE
    print("Computing per-patient RMSE...")
    patient_df = compute_patient_rmse(df, [args.x_var, args.y_var], timestep)
    print(f"Computed RMSE for {len(patient_df):,} patient-model combinations")

    # Filter to patients with both variables
    n_with_data = patient_df[[args.x_var, args.y_var]].notna().all(axis=1).sum()
    print(f"Patients with both {args.x_var} and {args.y_var}: {n_with_data:,} ({100 * n_with_data / len(patient_df):.1f}%)")

    if n_with_data == 0:
        print(f"Error: No data available. Cannot create plot.")
        return 1

    # Plot
    plot_rmse_2d(
        patient_df, args.x_var, args.y_var, args.output, args.horizon,
        cmap=args.cmap, n_bins=args.n_bins, n_levels=args.n_levels
    )

    return 0


if __name__ == "__main__":
    exit(main())
