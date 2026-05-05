#!/usr/bin/env python3
"""
Plot RMSE by prediction horizon for each model.

Each timestep represents 5 minutes, so we plot RMSE at 5m, 10m, 15m, ..., 60m.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model_styles import (
    add_figsize_arg,
    add_model_filter_args,
    add_model_legend_below,
    apply_model_filter,
    get_model_color_for,
    get_model_label,
    sort_models_for_render,
)


def compute_rmse_by_timestep(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute RMSE for each model at each timestep.

    Args:
        df: DataFrame with model, label_t0-t11, pred_t0-t11 columns

    Returns:
        DataFrame with model, timestep, minutes, rmse columns
    """
    timesteps = list(range(12))
    results = []

    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        for t in timesteps:
            label_col = f'label_t{t}'
            pred_col = f'pred_t{t}'
            mse = ((model_df[label_col] - model_df[pred_col]) ** 2).mean()
            rmse = np.sqrt(mse)
            results.append({
                'model': model,
                'timestep': t,
                'minutes': (t + 1) * 5,
                'rmse': rmse,
            })

    return pd.DataFrame(results)


def plot_rmse_by_timestep(
    rmse_df: pd.DataFrame,
    output_path: Path,
    figsize: tuple = (7, 4.5),
    dpi: int = 150,
    color_by: str = 'arch',
) -> None:
    """
    Create line plot of RMSE by timestep for each model.

    Args:
        rmse_df: DataFrame with model, minutes, rmse columns
        output_path: Path to save the plot
        figsize: Figure size in inches
        dpi: Resolution for saved figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Within each render-order group (non-gluforecast first, gluforecast last on top),
    # order worst -> best by mean RMSE so better models within the group draw on top.
    mean_rmse = rmse_df.groupby('model')['rmse'].mean()
    models = sort_models_for_render(
        rmse_df['model'].unique(),
        secondary_key=lambda m: -mean_rmse[m],
    )

    for i, model in enumerate(models):
        model_data = rmse_df[rmse_df['model'] == model].sort_values('minutes')
        ax.plot(
            model_data['minutes'],
            model_data['rmse'],
            marker='o',
            label=get_model_label(model, color_by=color_by),
            color=get_model_color_for(model, color_by=color_by),
            linewidth=1.0,
            markersize=4,
            linestyle='-',
            zorder=2 + i,
        )

    ax.set_xlabel('Prediction Horizon (minutes)', fontsize=14)
    ax.set_ylabel('RMSE (mg/dL)', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)

    # Mirror y-axis in mmol/L on the right (1 mmol/L = 18.0182 mg/dL)
    MGDL_PER_MMOL = 18.0182
    ax_right = ax.secondary_yaxis(
        'right',
        functions=(lambda y: y / MGDL_PER_MMOL, lambda y: y * MGDL_PER_MMOL),
    )
    ax_right.set_ylabel('RMSE (mmol/L)', fontsize=14)
    ax_right.tick_params(axis='y', labelsize=12)
    ax.set_title(
        'Glucose Prediction RMSE by Model and Time Horizon',
        fontsize=15,
        fontweight='bold',
    )

    # Set x-axis ticks at each 5-minute interval
    time_minutes = [(t + 1) * 5 for t in range(12)]
    ax.set_xticks(time_minutes)
    ax.set_xlim(0, 65)

    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    add_model_legend_below(fig, ax)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot RMSE by prediction horizon for each model"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("combined_results_new.parquet"),
        help="Input parquet file (default: combined_results_new.parquet)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("model_rmse_by_timestep.png"),
        help="Output plot file (default: model_rmse_by_timestep.png)",
    )
    add_model_filter_args(parser)
    add_figsize_arg(parser, default=(7.0, 4.5))

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        return 1

    print(f"Loading data from {args.input}...")
    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df):,} rows")

    keep, color_by = apply_model_filter(df['model'].unique().tolist(), args)
    df = df[df['model'].isin(keep)]
    print(f"Models: {', '.join(keep)}")

    print("\nComputing RMSE by model and timestep...")
    rmse_df = compute_rmse_by_timestep(df)

    pivot_table = rmse_df.pivot(index='minutes', columns='model', values='rmse')
    print("\nRMSE (mg/dL) by model and prediction horizon:")
    print(pivot_table.round(2).to_string())

    plot_rmse_by_timestep(rmse_df, args.output, figsize=tuple(args.figsize), color_by=color_by)

    return 0


if __name__ == "__main__":
    exit(main())
