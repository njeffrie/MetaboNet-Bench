#!/usr/bin/env python3
"""
Plot RMSE at 30-minute horizon by recency of carb intake.

Shows model performance across 6 conditions:
- carbs > 0 at t0
- carbs > 0 at t-5min
- carbs > 0 at t-10min
- carbs > 0 at t-15min
- carbs > 0 at t-20min
- carbs > 0 at t-25min
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


def compute_rmse(df: pd.DataFrame, pred_col: str, label_col: str) -> float:
    """Compute RMSE between prediction and label columns."""
    valid = df[pred_col].notna() & df[label_col].notna()
    if valid.sum() == 0:
        return np.nan
    diff = df.loc[valid, pred_col] - df.loc[valid, label_col]
    return np.sqrt((diff ** 2).mean())


def main():
    parser = argparse.ArgumentParser(description="Plot RMSE by recent carb intake")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("combined_results_new.parquet"),
        help="Input parquet file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("rmse_by_recent_carbs.png"),
        help="Output plot file",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=6,
        help="Prediction horizon index (default: 6 = 30 minutes)",
    )
    add_model_filter_args(parser)
    add_figsize_arg(parser, default=(12.0, 7.0))
    args = parser.parse_args()

    print(f"Loading data from {args.input}...")
    df = pd.read_parquet(args.input)
    print(f"  Loaded {len(df):,} rows")

    models, color_by = apply_model_filter(df['model'].unique().tolist(), args)
    models = sort_models_for_render(models)
    df = df[df['model'].isin(models)]
    print(f"  Models: {models}")

    # Define carb columns to check (t0 to t-25min)
    carb_columns = ['carbs', 'carbs_tminus_1', 'carbs_tminus_2',
                    'carbs_tminus_3', 'carbs_tminus_4', 'carbs_tminus_5']
    x_labels = ['t0', 't-5min', 't-10min', 't-15min', 't-20min', 't-25min']

    # Get prediction and label columns for the specified horizon
    pred_col = f'pred_t{args.horizon}'
    label_col = f'label_t{args.horizon}'
    horizon_minutes = args.horizon * 5

    print(f"\nComputing RMSE at {horizon_minutes}-minute horizon ({label_col})...")

    # Compute RMSE for each model and carb condition
    results = {model: [] for model in models}
    sample_counts = {model: [] for model in models}

    for carb_col in carb_columns:
        # Filter to rows where carbs > 0 at this time point
        mask = df[carb_col].notna() & (df[carb_col] > 0)
        df_filtered = df[mask]
        print(f"\n  {carb_col} > 0: {len(df_filtered):,} rows")

        for model in models:
            model_df = df_filtered[df_filtered['model'] == model]
            rmse = compute_rmse(model_df, pred_col, label_col)
            results[model].append(rmse)
            sample_counts[model].append(len(model_df))
            print(f"    {model}: RMSE={rmse:.2f}, n={len(model_df):,}")

    # Create the plot
    fig, ax = plt.subplots(figsize=tuple(args.figsize))

    x = np.arange(len(x_labels))
    width = 0.8 / len(models)

    for i, model in enumerate(models):
        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(x + offset, results[model], width,
                      label=get_model_label(model, color_by=color_by),
                      color=get_model_color_for(model, color_by=color_by))

    ax.set_xlabel('Time of Carb Intake (relative to prediction time)', fontsize=12)
    ax.set_ylabel(f'RMSE at {horizon_minutes}-minute horizon (mg/dL)', fontsize=12)
    ax.set_title(f'Model RMSE at {horizon_minutes}min Horizon\nby Recency of Carb Intake (carbs > 0)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    add_model_legend_below(fig, ax)
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {args.output}")

    # Also create a line chart version
    fig2, ax2 = plt.subplots(figsize=tuple(args.figsize))

    for model in models:
        ax2.plot(x, results[model], marker='o',
                 label=get_model_label(model, color_by=color_by),
                 color=get_model_color_for(model, color_by=color_by),
                 linewidth=1.0, markersize=4, linestyle='-')

    ax2.set_xlabel('Time of Carb Intake (relative to prediction time)', fontsize=12)
    ax2.set_ylabel(f'RMSE at {horizon_minutes}-minute horizon (mg/dL)', fontsize=12)
    ax2.set_title(f'Model RMSE at {horizon_minutes}min Horizon\nby Recency of Carb Intake (carbs > 0)', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels)
    ax2.grid(alpha=0.3)

    line_output = args.output.with_stem(args.output.stem + '_line')
    plt.tight_layout()
    add_model_legend_below(fig2, ax2)
    plt.savefig(line_output, dpi=150, bbox_inches='tight')
    print(f"Saved line plot to {line_output}")

    return 0


if __name__ == "__main__":
    exit(main())
