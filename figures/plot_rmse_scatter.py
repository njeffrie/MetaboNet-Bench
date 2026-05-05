#!/usr/bin/env python3
"""
Plot RMSE vs a continuous variable (age, cgm_mean, cgm_std, etc.) as a scatter plot.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

from model_styles import add_model_filter_args, apply_model_filter


# Display names for x-axis variables
VARIABLE_DISPLAY_NAMES = {
    'age': 'Age (years)',
    'cgm_mean': 'Mean CGM (mg/dL)',
    'cgm_std': 'CGM Std Dev (mg/dL)',
    'weight': 'Weight (lbs)',
    'height': 'Height (ft)',
    'age_of_diagnosis': 'Age of Diagnosis (years)',
}


def compute_patient_rmse(df: pd.DataFrame, x_var: str, timestep: int = None) -> pd.DataFrame:
    """
    Compute RMSE per patient.

    Args:
        df: DataFrame with user_id, dataset, model, x_var, label_t*, pred_t* columns
        x_var: The x-axis variable to include in output
        timestep: If specified, compute RMSE at this timestep only (0-11).
                  If None, compute RMSE across all timesteps.

    Returns:
        DataFrame with user_id, dataset, model, x_var, rmse columns (one row per patient per model)
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

    # Group by patient (user_id, dataset, model) and compute mean squared error, then RMSE
    patient_rmse = df.groupby(['user_id', 'dataset', 'model']).agg({
        'squared_error': 'mean',
        x_var: 'first',
    }).reset_index()

    patient_rmse['rmse'] = np.sqrt(patient_rmse['squared_error'])
    patient_rmse = patient_rmse.drop(columns=['squared_error'])

    return patient_rmse


def plot_rmse_vs_variable(
    patient_df: pd.DataFrame,
    x_var: str,
    output_path: Path,
    horizon_minutes: int = None,
    figsize: tuple = (12, 8),
    dpi: int = 150,
    alpha: float = 0.3,
) -> None:
    """
    Create scatter plot of RMSE vs a variable, with each patient as a dot.
    Creates a grid with one subplot per model.

    Args:
        patient_df: DataFrame with user_id, model, x_var, rmse columns
        x_var: Name of the x-axis variable
        output_path: Path to save the plot
        horizon_minutes: Prediction horizon in minutes (for title), None if all timesteps
        figsize: Figure size in inches
        dpi: Resolution for saved figure
        alpha: Transparency of dots
    """
    models = sorted(patient_df['model'].unique())
    n_models = len(models)
    cols = 3
    rows = (n_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize, sharex=True, sharey=True)
    axes = axes.flatten()

    x_label = VARIABLE_DISPLAY_NAMES.get(x_var, x_var)

    for idx, model in enumerate(models):
        ax = axes[idx]
        model_data = patient_df[patient_df['model'] == model].dropna(subset=[x_var, 'rmse'])

        ax.scatter(
            model_data[x_var],
            model_data['rmse'],
            alpha=alpha,
            s=10,
            c='steelblue',
            edgecolors='none',
        )

        # Add LOESS trend line
        if len(model_data) > 10:
            # LOESS smoothing (frac controls smoothness, 0.3 is moderately smooth)
            loess_result = lowess(
                model_data['rmse'].values,
                model_data[x_var].values,
                frac=0.3,
                return_sorted=True,
            )
            ax.plot(loess_result[:, 0], loess_result[:, 1], 'r-', linewidth=2, label='LOESS')

            # Compute correlation
            corr = model_data[x_var].corr(model_data['rmse'])
            ax.text(0.05, 0.95, f'r = {corr:.3f}\nn = {len(model_data):,}',
                    transform=ax.transAxes, fontsize=9, va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_title(model, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)

        if idx >= (rows - 1) * cols:
            ax.set_xlabel(x_label, fontsize=10)
        if idx % cols == 0:
            ax.set_ylabel('RMSE (mg/dL)', fontsize=10)

    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    if horizon_minutes:
        title = f'Patient RMSE vs {x_label} at {horizon_minutes}min Horizon'
    else:
        title = f'Patient RMSE vs {x_label} (averaged across all horizons)'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot RMSE vs a continuous variable as a scatter plot"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("combined_results_new.parquet"),
        help="Input parquet file (default: combined_results_new.parquet)",
    )
    parser.add_argument(
        "--variable", "-v",
        type=str,
        default="age",
        choices=['age', 'cgm_mean', 'cgm_std', 'weight', 'height', 'age_of_diagnosis'],
        help="Variable for x-axis (default: age)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        choices=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
        default=None,
        help="Prediction horizon in minutes. If not specified, averages across all horizons.",
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
        args.output = Path(f"rmse_vs_{args.variable}{horizon_str}.png")

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

    # Check if variable exists
    if args.variable not in df.columns:
        print(f"Error: Variable '{args.variable}' not found in data.")
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
    patient_df = compute_patient_rmse(df, args.variable, timestep)
    print(f"Computed RMSE for {len(patient_df):,} patient-model combinations")

    # Filter to patients with variable data
    n_with_data = patient_df[args.variable].notna().sum()
    print(f"Patients with {args.variable} data: {n_with_data:,} ({100 * n_with_data / len(patient_df):.1f}%)")

    if n_with_data == 0:
        print(f"Error: No data available for {args.variable}. Cannot create plot.")
        return 1

    # Plot
    plot_rmse_vs_variable(patient_df, args.variable, args.output, args.horizon)

    return 0


if __name__ == "__main__":
    exit(main())
