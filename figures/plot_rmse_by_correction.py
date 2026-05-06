#!/usr/bin/env python3
"""
Plot RMSE across all prediction horizons comparing:
1) Hyperglycemia with correction (bg > 250 AND insulin > 2u)
2) Hyperglycemia without correction (bg > 250 AND insulin <= 2u)

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
    line_style_for,
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
    parser = argparse.ArgumentParser(description="Plot RMSE by corrective bolus")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("combined_results_new.parquet"),
        help="Input parquet file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("rmse_by_correction.png"),
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

    # Define insulin columns to check (only historical, not current)
    insulin_columns = ['insulin_tminus_1', 'insulin_tminus_2',
                       'insulin_tminus_3', 'insulin_tminus_4', 'insulin_tminus_5']

    # Filter to rows with required data
    has_bg = df['cgm_at_t0'].notna()
    has_insulin = df[insulin_columns].notna().all(axis=1)
    df_valid = df[has_bg & has_insulin].copy()
    print(f"  Rows with bg and insulin data: {len(df_valid):,}")

    # Hyperglycemia: bg > 250
    hyperglycemia_mask = df_valid['cgm_at_t0'] > 250
    df_hyper = df_valid[hyperglycemia_mask]
    print(f"  Hyperglycemia (bg > 250): {len(df_hyper):,} rows")

    # Correction: insulin > 2u at any timepoint
    has_correction = (df_hyper[insulin_columns] > 2).any(axis=1)

    df_hyper_with_correction = df_hyper[has_correction]
    df_hyper_without_correction = df_hyper[~has_correction]

    print(f"  Hyperglycemia with correction (insulin > 2u): {len(df_hyper_with_correction):,} rows")
    print(f"  Hyperglycemia without correction: {len(df_hyper_without_correction):,} rows")

    # Prediction horizons (indices 0..11 correspond to 5..60 min in 5-min steps)
    horizons = list(range((h_min_min // 5) - 1, (h_max_min // 5)))
    horizon_minutes = [(h + 1) * 5 for h in horizons]

    # Compute RMSE for each model and horizon for both conditions
    print("\nComputing RMSE across horizons...")

    results_with_correction = {model: [] for model in models}
    results_without_correction = {model: [] for model in models}

    for h in horizons:
        pred_col = f'pred_t{h}'
        label_col = f'label_t{h}'

        for model in models:
            # With correction
            model_df = df_hyper_with_correction[df_hyper_with_correction['model'] == model]
            rmse = compute_rmse(model_df, pred_col, label_col)
            results_with_correction[model].append(rmse)

            # Without correction
            model_df = df_hyper_without_correction[df_hyper_without_correction['model'] == model]
            rmse = compute_rmse(model_df, pred_col, label_col)
            results_without_correction[model].append(rmse)

    # Print summary at the largest horizon in the requested range
    summary_idx = len(horizons) - 1
    summary_min = horizon_minutes[summary_idx]
    print(f"\n--- RMSE at {summary_min}min horizon (t{horizons[summary_idx]}) ---")
    print("Model                      No Corr    With Corr    Diff")
    print("-" * 58)
    for model in models:
        no_corr = results_without_correction[model][summary_idx]
        with_corr = results_with_correction[model][summary_idx]
        diff = with_corr - no_corr
        print(f"{model:25s} {no_corr:8.2f}    {with_corr:8.2f}   {diff:+6.2f}")

    # Create the plot with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=tuple(args.figsize), sharey=True)

    # Subplot 1: Hyperglycemia without correction
    for model in models:
        ax1.plot(horizon_minutes, results_without_correction[model],
                 label=get_model_label(model, color_by=color_by),
                 color=get_model_color_for(model, color_by=color_by),
                 **line_style_for(model, color_by),
                 **get_marker_edge_kwargs())

    ax1.set_xlabel('Prediction Horizon (minutes)', fontsize=14)
    ax1.set_ylabel('RMSE (mg/dL)', fontsize=14)
    ax1.set_title(f'Hyperglycemia Without Correction\n(BG > 250, insulin ≤ 2u at t-5 to t-25min)\nn = {len(df_hyper_without_correction)//len(models):,} per model', fontsize=15)
    ax1.set_xticks(horizon_minutes)
    ax1.grid(alpha=0.3)

    # Subplot 2: Hyperglycemia with correction
    for model in models:
        ax2.plot(horizon_minutes, results_with_correction[model],
                 label=get_model_label(model, color_by=color_by),
                 color=get_model_color_for(model, color_by=color_by),
                 **line_style_for(model, color_by),
                 **get_marker_edge_kwargs())

    ax2.set_xlabel('Prediction Horizon (minutes)', fontsize=14)
    ax2.set_title(f'Hyperglycemia With Correction\n(BG > 250, insulin > 2u at t-5 to t-25min)\nn = {len(df_hyper_with_correction)//len(models):,} per model', fontsize=15)
    ax2.set_xticks(horizon_minutes)
    ax2.grid(alpha=0.3)

    if args.ylim is not None:
        ax1.set_ylim(args.ylim)
        ax2.set_ylim(args.ylim)

    plt.suptitle('Model RMSE During Hyperglycemia: With vs Without Corrective Bolus', fontsize=18, y=1.02)
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
        diff = [results_with_correction[model][h] - results_without_correction[model][h] for h in range(len(horizons))]
        ax3.plot(horizon_minutes, diff,
                 label=get_model_label(model, color_by=color_by),
                 color=get_model_color_for(model, color_by=color_by),
                 **line_style_for(model, color_by),
                 **get_marker_edge_kwargs())

    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Prediction Horizon (minutes)', fontsize=14)
    ax3.set_ylabel('RMSE Difference (With Correction - Without) (mg/dL)', fontsize=14)
    ax3.set_title('Impact of Corrective Bolus on Model RMSE During Hyperglycemia\n(Positive = worse performance with correction)', fontsize=17)
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
