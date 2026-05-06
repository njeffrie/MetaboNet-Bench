#!/usr/bin/env python3
"""
Plot RMSE across all prediction horizons comparing:
1) No recent meal (carbs = 0 for all t_minus_n)
2) Recent meal present (carbs > 0 for at least one t_minus_n)

One line per model on each subplot.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model_styles import (
    add_legends_below,
    add_model_filter_args,
    add_model_legend_below,
    apply_model_filter,
    get_marker_edge_kwargs,
    get_model_color_for,
    get_model_label,
    get_model_linestyle,
    get_model_marker,
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
    parser = argparse.ArgumentParser(description="Plot RMSE by meal presence")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("combined_results_new.parquet"),
        help="Input parquet file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("rmse_by_meal_presence.png"),
        help="Output plot file",
    )
    add_model_filter_args(parser)
    parser.add_argument(
        "--ylim",
        type=float,
        nargs=2,
        default=None,
        metavar=('YMIN', 'YMAX'),
        help="Y-axis limits for the main side-by-side plot (e.g. --ylim 0 80)",
    )
    parser.add_argument(
        "--ylim-diff",
        type=float,
        nargs=2,
        default=None,
        metavar=('YMIN', 'YMAX'),
        help="Y-axis limits for the difference plot (e.g. --ylim-diff -5 30)",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs=2,
        default=[5, 60],
        metavar=('MIN_MIN', 'MAX_MIN'),
        help="Horizon range in minutes, multiples of 5 (default: 5 60)",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=[16.0, 7.0],
        metavar=('W', 'H'),
        help="Figure size (inches) for the main side-by-side plot (default: 16 7)",
    )
    parser.add_argument(
        "--figsize-diff",
        type=float,
        nargs=2,
        default=[12.0, 7.0],
        metavar=('W', 'H'),
        help="Figure size (inches) for the difference plot (default: 12 7)",
    )
    parser.add_argument(
        "--no-legend",
        action="store_true",
        help="Suppress the legend on both the side-by-side and difference plots",
    )
    args = parser.parse_args()

    h_min_min, h_max_min = args.horizons
    if h_min_min < 5 or h_max_min > 60 or h_min_min % 5 or h_max_min % 5 or h_min_min > h_max_min:
        raise SystemExit(f"--horizons must be multiples of 5 in [5, 60] with MIN_MIN <= MAX_MIN; got {args.horizons}")

    print(f"Loading data from {args.input}...")
    df = pd.read_parquet(args.input)
    print(f"  Loaded {len(df):,} rows")

    # Apply standard ablation / variant filtering up front
    models, color_by = apply_model_filter(df['model'].unique().tolist(), args)
    models = sort_models_for_render(models)
    df = df[df['model'].isin(models)]
    print(f"  Models: {models}")

    # Define carb columns to check (t-5min to t-25min)
    carb_columns = ['carbs_tminus_1', 'carbs_tminus_2', 'carbs_tminus_3',
                    'carbs_tminus_4', 'carbs_tminus_5']

    # Create meal presence masks
    # First, filter to rows that have all carb columns available
    has_all_carbs = df[carb_columns].notna().all(axis=1)
    df_with_carbs = df[has_all_carbs].copy()
    print(f"  Rows with all carb columns: {len(df_with_carbs):,}")

    # No meal: all carb columns are 0 (or very close to 0)
    no_meal_mask = (df_with_carbs[carb_columns] <= 0).all(axis=1)
    df_no_meal = df_with_carbs[no_meal_mask]
    print(f"  No recent meal: {len(df_no_meal):,} rows")

    # Has meal: at least one carb column > 0
    has_meal_mask = (df_with_carbs[carb_columns] > 0).any(axis=1)
    df_has_meal = df_with_carbs[has_meal_mask]
    print(f"  Recent meal present: {len(df_has_meal):,} rows")

    # Prediction horizons (indices 0..11 correspond to 5..60 min in 5-min steps)
    horizons = list(range((h_min_min // 5) - 1, (h_max_min // 5)))
    horizon_minutes = [(h + 1) * 5 for h in horizons]

    # Compute RMSE for each model and horizon for both conditions
    print("\nComputing RMSE across horizons...")

    label_cols = [f'label_t{h}' for h in horizons]
    pred_cols = [f'pred_t{h}' for h in horizons]

    def rmse_per_model_per_horizon(cond_df: pd.DataFrame) -> pd.DataFrame:
        """One group-by pass: returns DataFrame indexed by model, columns = horizons."""
        sq = (cond_df[label_cols].to_numpy() - cond_df[pred_cols].to_numpy()) ** 2
        sq_df = pd.DataFrame(sq, index=cond_df.index, columns=horizons)
        sq_df['model'] = cond_df['model'].values
        mse = sq_df.groupby('model', observed=True)[horizons].mean()
        return np.sqrt(mse)

    rmse_no_meal_df = rmse_per_model_per_horizon(df_no_meal)
    rmse_has_meal_df = rmse_per_model_per_horizon(df_has_meal)

    results_no_meal = {
        model: rmse_no_meal_df.loc[model, horizons].tolist() if model in rmse_no_meal_df.index
        else [np.nan] * len(horizons)
        for model in models
    }
    results_has_meal = {
        model: rmse_has_meal_df.loc[model, horizons].tolist() if model in rmse_has_meal_df.index
        else [np.nan] * len(horizons)
        for model in models
    }

    # Print summary at the largest horizon in the requested range
    summary_idx = len(horizons) - 1
    summary_min = horizon_minutes[summary_idx]
    print(f"\n--- RMSE at {summary_min}min horizon (t{horizons[summary_idx]}) ---")
    print("Model                      No Meal    Has Meal    Diff")
    print("-" * 55)
    for model in models:
        no_meal = results_no_meal[model][summary_idx]
        has_meal = results_has_meal[model][summary_idx]
        diff = has_meal - no_meal
        print(f"{model:25s} {no_meal:8.2f}   {has_meal:8.2f}   {diff:+6.2f}")

    # Create the plot with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=tuple(args.figsize), sharey=True)

    # Subplot 1: No recent meal
    for model in models:
        ax1.plot(horizon_minutes, results_no_meal[model], label=get_model_label(model, color_by=color_by),
                 color=get_model_color_for(model, color_by=color_by),
                 marker=get_model_marker(model),
                 linestyle=get_model_linestyle(model),
                 linewidth=1.0, markersize=5,
                 **get_marker_edge_kwargs())

    ax1.set_xlabel('Prediction Horizon (minutes)', fontsize=12)
    ax1.set_ylabel('RMSE (mg/dL)', fontsize=12)
    ax1.set_title(f'No Recent Meal\n(carbs = 0 for t-5 to t-25min)\nn = {len(df_no_meal)//len(models):,} per model', fontsize=12)
    ax1.set_xticks(horizon_minutes)
    ax1.grid(alpha=0.3)

    # Subplot 2: Recent meal present
    for model in models:
        ax2.plot(horizon_minutes, results_has_meal[model], label=get_model_label(model, color_by=color_by),
                 color=get_model_color_for(model, color_by=color_by),
                 marker=get_model_marker(model),
                 linestyle=get_model_linestyle(model),
                 linewidth=1.0, markersize=5,
                 **get_marker_edge_kwargs())

    ax2.set_xlabel('Prediction Horizon (minutes)', fontsize=12)
    ax2.set_title(f'Recent Meal Present\n(carbs > 0 for at least one of t-5 to t-25min)\nn = {len(df_has_meal)//len(models):,} per model', fontsize=12)
    ax2.set_xticks(horizon_minutes)
    ax2.grid(alpha=0.3)

    if args.ylim is not None:
        ax1.set_ylim(args.ylim)
        ax2.set_ylim(args.ylim)

    plt.suptitle('Model RMSE by Prediction Horizon: Meal vs No Meal', fontsize=14, y=1.02)
    plt.tight_layout()
    if not args.no_legend:
        if color_by == 'feature':
            add_legends_below(fig, ax1)
        else:
            add_model_legend_below(fig, ax1)
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {args.output}")

    # Also create a difference plot
    fig2, ax3 = plt.subplots(figsize=tuple(args.figsize_diff))

    for model in models:
        diff = [results_has_meal[model][h] - results_no_meal[model][h] for h in range(len(horizons))]
        ax3.plot(horizon_minutes, diff, label=get_model_label(model, color_by=color_by),
                 color=get_model_color_for(model, color_by=color_by),
                 marker=get_model_marker(model),
                 linestyle=get_model_linestyle(model),
                 linewidth=1.0, markersize=5,
                 **get_marker_edge_kwargs())

    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Prediction Horizon (minutes)', fontsize=12)
    ax3.set_ylabel('RMSE Difference (Has Meal - No Meal) (mg/dL)', fontsize=12)
    ax3.set_title('Impact of Recent Meal on Model RMSE\n(Positive = worse performance with meal)', fontsize=14)
    ax3.set_xticks(horizon_minutes)
    ax3.grid(alpha=0.3)

    if args.ylim_diff is not None:
        ax3.set_ylim(args.ylim_diff)

    diff_output = args.output.with_stem(args.output.stem + '_diff')
    plt.tight_layout()
    if not args.no_legend:
        if color_by == 'feature':
            add_legends_below(fig2, ax3)
        else:
            add_model_legend_below(fig2, ax3)
    plt.savefig(diff_output, dpi=150, bbox_inches='tight')
    print(f"Saved difference plot to {diff_output}")

    return 0


if __name__ == "__main__":
    exit(main())
