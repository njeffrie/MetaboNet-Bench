#!/usr/bin/env python3
"""
Plot RMSE vs a continuous variable with all models on the same plot.
Groups data into bins and shows mean RMSE with standard error bars.
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


# Display names for x-axis variables
VARIABLE_DISPLAY_NAMES = {
    'age': 'Age (years)',
    'cgm_mean': 'Mean CGM (mg/dL)',
    'cgm_std': 'CGM Std Dev (mg/dL)',
    'weight': 'Weight (lbs)',
    'height': 'Height (ft)',
    'age_of_diagnosis': 'Age of Diagnosis (years)',
    'years_since_diagnosis': 'Years Since Diagnosis',
}

# Default bin configurations for each variable
VARIABLE_BINS = {
    'age': {'start': 0, 'end': 80, 'step': 5},
    'cgm_mean': {'start': 80, 'end': 220, 'step': 10},
    'cgm_std': {'start': 20, 'end': 80, 'step': 5},
    'weight': {'start': 80, 'end': 280, 'step': 20},
    'height': {'start': 4, 'end': 7, 'step': 0.25},
    'age_of_diagnosis': {'start': 0, 'end': 60, 'step': 5},
    'years_since_diagnosis': {'start': 0, 'end': 60, 'step': 5},
}


def compute_binned_rmse(df: pd.DataFrame, x_var: str, timestep: int = None, bins: list = None) -> pd.DataFrame:
    """
    Compute RMSE per model and bin.

    Args:
        df: DataFrame with model, x_var, label_t*, pred_t* columns
        x_var: The x-axis variable to bin
        timestep: If specified, compute RMSE at this timestep only (0-11).
                  If None, compute RMSE across all timesteps.
        bins: List of bin edges

    Returns:
        DataFrame with model, bin_center, rmse, rmse_std, n_samples columns
    """
    df = df.copy()

    # Compute squared error
    if timestep is not None:
        label_col = f'label_t{timestep}'
        pred_col = f'pred_t{timestep}'
        df['squared_error'] = (df[label_col] - df[pred_col]) ** 2
    else:
        # All timesteps - average per-sample squared error across the 12 horizons,
        # ignoring NaN cells (different models / windows cover different horizons).
        squared_errors = []
        for t in range(12):
            squared_errors.append((df[f'label_t{t}'] - df[f'pred_t{t}']) ** 2)
        df['squared_error'] = np.nanmean(np.stack(squared_errors, axis=0), axis=0)

    # Create bins
    bin_labels = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins) - 1)]
    df['bin'] = pd.cut(df[x_var], bins=bins, labels=bin_labels, include_lowest=True)

    # Group by model and bin, compute RMSE and std error
    results = []
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        for bin_center in bin_labels:
            bin_df = model_df[model_df['bin'] == bin_center]
            if len(bin_df) > 0:
                # NaN-tolerant: a sample whose squared_error is NaN at all 12
                # horizons stays NaN here and is dropped by nanmean/nanstd.
                sq_err = bin_df['squared_error'].to_numpy(dtype=float)
                finite = sq_err[np.isfinite(sq_err)]
                if finite.size == 0:
                    continue
                mse = float(np.mean(finite))
                rmse = float(np.sqrt(mse))
                # Standard error of RMSE (approximation)
                rmse_values = np.sqrt(finite)
                rmse_std = float(rmse_values.std(ddof=0) / np.sqrt(finite.size))
                results.append({
                    'model': model,
                    'bin_center': bin_center,
                    'rmse': rmse,
                    'rmse_std': rmse_std,
                    'n_samples': len(bin_df),
                })

    return pd.DataFrame(results)


def plot_rmse_all_models(
    binned_df: pd.DataFrame,
    x_var: str,
    output_path: Path,
    horizon_minutes: int = None,
    figsize: tuple = (12, 7),
    dpi: int = 150,
    raw_df: pd.DataFrame = None,
    show_scatter: bool = False,
    timestep: int = None,
    color_by: str = 'arch',
) -> None:
    """
    Create a single plot with binned RMSE and error bars for all models.

    Args:
        binned_df: DataFrame with model, bin_center, rmse, rmse_std, n_samples columns
        x_var: Name of the x-axis variable
        output_path: Path to save the plot
        horizon_minutes: Prediction horizon in minutes (for title), None if all timesteps
        figsize: Figure size in inches
        dpi: Resolution for saved figure
        exclude_models: List of models to exclude from plot
        raw_df: Raw DataFrame for scatter plot overlay
        show_scatter: Whether to show scatter plot of individual points
        timestep: Timestep used for computing error (for scatter plot)
    """
    fig, ax = plt.subplots(figsize=figsize)

    models = sort_models_for_render(binned_df['model'].unique())

    # Plot scatter points first (behind lines)
    if show_scatter and raw_df is not None:
        for model in models:
            model_df = raw_df[raw_df['model'] == model].copy()
            if len(model_df) == 0:
                continue
            if timestep is not None:
                error = np.abs(model_df[f'label_t{timestep}'] - model_df[f'pred_t{timestep}'])
            else:
                errors = []
                for t in range(12):
                    errors.append((model_df[f'label_t{t}'] - model_df[f'pred_t{t}']) ** 2)
                error = np.sqrt(np.mean(errors, axis=0))
            ax.scatter(
                model_df[x_var],
                error,
                c=[get_model_color_for(model, color_by=color_by)],
                s=1,
                alpha=0.1,
                rasterized=True,
            )

    x_label = VARIABLE_DISPLAY_NAMES.get(x_var, x_var)

    for model in models:
        model_data = binned_df[binned_df['model'] == model].sort_values('bin_center')

        if len(model_data) > 0:
            ax.errorbar(
                model_data['bin_center'],
                model_data['rmse'],
                yerr=1.96 * model_data['rmse_std'],
                color=get_model_color_for(model, color_by=color_by),
                linewidth=1.0,
                label=get_model_label(model, color_by=color_by),
                marker=get_model_marker(model),
                markersize=5,
                linestyle=get_model_linestyle(model),
                capsize=3,
                elinewidth=0.8,
                **get_marker_edge_kwargs(),
            )

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel('RMSE (mg/dL)', fontsize=12)

    if horizon_minutes:
        title = f'RMSE vs {x_label} by Model ({horizon_minutes}min Horizon)'
    else:
        title = f'RMSE vs {x_label} by Model (all horizons averaged)'
    ax.set_title(title, fontsize=14, fontweight='bold')

    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if color_by == 'feature':
        add_legends_below(fig, ax)
    else:
        add_model_legend_below(fig, ax)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot RMSE vs a variable with all models on the same plot"
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
        default="cgm_std",
        choices=['age', 'cgm_mean', 'cgm_std', 'weight', 'height', 'age_of_diagnosis', 'years_since_diagnosis'],
        help="Variable for x-axis (default: cgm_std)",
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
    add_figsize_arg(parser, default=(12.0, 7.0))
    parser.add_argument(
        "--show-scatter",
        action="store_true",
        help="Overlay scatter plot of individual points behind the binned lines",
    )

    args = parser.parse_args()

    if args.output is None:
        horizon_str = f"_{args.horizon}min" if args.horizon else "_all"
        args.output = Path(f"rmse_by_{args.variable}_all_models{horizon_str}.png")

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

    # Compute derived variables
    if args.variable == 'years_since_diagnosis':
        if 'age' in df.columns and 'age_of_diagnosis' in df.columns:
            df['years_since_diagnosis'] = df['age'] - df['age_of_diagnosis']
            print(f"Computed years_since_diagnosis from age - age_of_diagnosis")
        else:
            print("Error: Cannot compute years_since_diagnosis - missing 'age' or 'age_of_diagnosis'")
            return 1

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

    # Filter to rows with variable data
    n_with_data = df[args.variable].notna().sum()
    print(f"Rows with {args.variable} data: {n_with_data:,} ({100 * n_with_data / len(df):.1f}%)")

    if n_with_data == 0:
        print(f"Error: No data available for {args.variable}. Cannot create plot.")
        return 1

    # Get bin configuration for this variable
    bin_config = VARIABLE_BINS.get(args.variable, {'start': 0, 'end': 100, 'step': 10})
    bins = list(np.arange(bin_config['start'], bin_config['end'] + bin_config['step'], bin_config['step']))
    print(f"Using bins: {bins[0]} to {bins[-1]} with step {bin_config['step']}")

    # Compute binned RMSE
    print("Computing binned RMSE...")
    binned_df = compute_binned_rmse(df, args.variable, timestep, bins)
    print(f"Computed RMSE for {len(binned_df):,} model-bin combinations")

    # Plot
    plot_rmse_all_models(binned_df, args.variable, args.output, args.horizon,
                         figsize=tuple(args.figsize),
                         raw_df=df if args.show_scatter else None,
                         show_scatter=args.show_scatter,
                         timestep=timestep,
                         color_by=color_by)

    return 0


if __name__ == "__main__":
    exit(main())
