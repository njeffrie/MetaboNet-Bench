#!/usr/bin/env python3
"""
Plot RMSE by prediction horizon, stratified by demographic group, with separate charts per model.

For each model, creates a line plot where:
- X-axis: prediction horizon (5, 10, 15, ..., 60 minutes)
- Y-axis: RMSE (mg/dL)
- Lines: one per demographic group
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model_styles import add_model_filter_args, apply_model_filter


# Age bin definitions
AGE_BINS = [0, 18, 30, 50, 65, 100]
AGE_LABELS = ['<18', '18-30', '30-50', '50-65', '65+']

# Ethnicity consolidation mapping
ETHNICITY_MAPPING = {
    'Black/African American': 'Black/African-American',
}


def compute_rmse_by_model_group_timestep(
    df: pd.DataFrame,
    group_col: str,
) -> pd.DataFrame:
    """
    Compute RMSE for each model, demographic group, and timestep.

    Args:
        df: DataFrame with model, group_col, label_t0-t11, pred_t0-t11 columns
        group_col: Column to group by

    Returns:
        DataFrame with model, group, timestep, minutes, rmse, n_samples columns
    """
    timesteps = list(range(12))
    results = []

    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        for group in model_df[group_col].dropna().unique():
            group_df = model_df[model_df[group_col] == group]
            n_samples = len(group_df)
            for t in timesteps:
                label_col = f'label_t{t}'
                pred_col = f'pred_t{t}'
                mse = ((group_df[label_col] - group_df[pred_col]) ** 2).mean()
                rmse = np.sqrt(mse)
                results.append({
                    'model': model,
                    'group': group,
                    'timestep': t,
                    'minutes': (t + 1) * 5,
                    'rmse': rmse,
                    'n_samples': n_samples,
                })

    return pd.DataFrame(results)


def plot_all_models_grid(
    rmse_df: pd.DataFrame,
    group_col: str,
    output_path: Path,
    dpi: int = 150,
    group_order: list = None,
) -> None:
    """
    Create a grid of line plots, one per model, showing RMSE by timestep with lines per demographic group.

    Args:
        rmse_df: DataFrame with model, group, minutes, rmse, n_samples columns
        group_col: Name of the grouping column (for labels)
        output_path: Path to save the plot
        dpi: Resolution for saved figure
        group_order: Optional list specifying the order of groups
    """
    models = sorted(rmse_df['model'].unique())
    n_models = len(models)

    # Determine grid dimensions
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False, sharey=True)

    # Determine groups and colors (consistent across all subplots)
    all_groups = rmse_df['group'].dropna().unique()
    if group_order is not None:
        groups = [g for g in group_order if g in set(all_groups)]
    else:
        groups = sorted(all_groups, key=lambda x: (isinstance(x, str), str(x)))

    colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))
    color_map = dict(zip(groups, colors))

    # Get sample counts per group (from first model, should be consistent)
    sample_counts = {}
    first_model_df = rmse_df[rmse_df['model'] == models[0]]
    for group in groups:
        group_data = first_model_df[first_model_df['group'] == group]
        sample_counts[group] = group_data['n_samples'].iloc[0] if len(group_data) > 0 else 0

    for idx, model in enumerate(models):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        model_df = rmse_df[rmse_df['model'] == model]

        for group in groups:
            group_data = model_df[model_df['group'] == group].sort_values('minutes')
            if len(group_data) == 0:
                continue

            ax.plot(
                group_data['minutes'],
                group_data['rmse'],
                marker='o',
                label=group,
                color=color_map[group],
                linewidth=2,
                markersize=4,
                markeredgecolor='black',
                markeredgewidth=0.5,
            )

        ax.set_xlabel('Horizon (min)', fontsize=10)
        ax.set_ylabel('RMSE (mg/dL)', fontsize=10)
        ax.set_title(model, fontsize=11, fontweight='bold')

        time_minutes = [(t + 1) * 5 for t in range(12)]
        ax.set_xticks(time_minutes[::2])  # Every 10 min to reduce clutter
        ax.set_xlim(0, 65)
        ax.grid(True, alpha=0.3)

    # Hide empty subplots
    for idx in range(n_models, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)

    # Create shared legend with sample counts
    legend_labels = [f"{g} (n={sample_counts[g]:,})" for g in groups]
    legend_handles = [plt.Line2D([0], [0], color=color_map[g], linewidth=2, marker='o', markersize=4, markeredgecolor='black', markeredgewidth=0.5) for g in groups]

    fig.legend(
        legend_handles,
        legend_labels,
        title=group_col.replace('_', ' ').title(),
        loc='center right',
        bbox_to_anchor=(1.12, 0.5),
    )

    fig.suptitle(
        f'RMSE by {group_col.replace("_", " ").title()} and Time Horizon',
        fontsize=14,
        fontweight='bold',
        y=1.02,
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot RMSE by prediction horizon, stratified by demographic, with separate charts per model"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("combined_results_new.parquet"),
        help="Input parquet file (default: combined_results_new.parquet)",
    )
    parser.add_argument(
        "--group-by",
        type=str,
        required=True,
        choices=['age', 'gender', 'ethnicity', 'insulin_delivery_modality', 'cgm_device', 'subject_split_across_traintest'],
        help="Demographic variable to stratify by",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output plot file (default: rmse_horizon_{group_by}.png)",
    )
    add_model_filter_args(parser)

    args = parser.parse_args()

    if args.output is None:
        args.output = Path(f"rmse_horizon_{args.group_by}.png")

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

    # Handle age binning
    group_col = args.group_by
    if args.group_by == 'age':
        print(f"Binning age into groups: {AGE_LABELS}")
        df['age_group'] = pd.cut(df['age'], bins=AGE_BINS, labels=AGE_LABELS, right=False)
        group_col = 'age_group'

    # Consolidate ethnicity categories
    if args.group_by == 'ethnicity':
        print("Consolidating redundant ethnicity categories...")
        df['ethnicity'] = df['ethnicity'].replace(ETHNICITY_MAPPING)

    # Check for missing values
    n_missing = df[group_col].isna().sum()
    n_total = len(df)
    print(f"{group_col}: {n_total - n_missing:,} non-null ({100 * (n_total - n_missing) / n_total:.1f}%)")

    # Print group distribution
    print(f"\nSamples by {group_col}:")
    for group, count in df[group_col].value_counts().sort_index().items():
        print(f"  {group}: {count:,}")

    # Compute RMSE
    print("\nComputing RMSE by model, group, and timestep...")
    rmse_df = compute_rmse_by_model_group_timestep(df, group_col)

    if len(rmse_df) == 0:
        print(f"\nError: No RMSE data computed. Check that {args.group_by} has valid values.")
        return 1

    # Plot grid of all models
    group_order = AGE_LABELS if args.group_by == 'age' else None

    print(f"\nGenerating grid plot...")
    plot_all_models_grid(rmse_df, args.group_by, args.output, group_order=group_order)

    print(f"\nDone!")

    return 0


if __name__ == "__main__":
    exit(main())
