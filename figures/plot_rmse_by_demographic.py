#!/usr/bin/env python3
"""
Plot RMSE by model, stratified by demographic variables at a specific time horizon.

Supports stratification by age (binned), gender, ethnicity, etc.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model_styles import (
    add_figsize_arg,
    add_model_filter_args,
    apply_model_filter,
    get_model_color_for,
    get_model_label,
    sort_models_for_render,
)


# Age bin definitions
AGE_BINS = [0, 18, 30, 50, 65, 100]
AGE_LABELS = ['<18', '18-30', '30-50', '50-65', '65+']

# Ethnicity consolidation mapping (normalize redundant categories)
ETHNICITY_MAPPING = {
    'Black/African American': 'Black/African-American',
    # Add any other mappings as needed
}

# Subject split label mapping
SUBJECT_SPLIT_MAPPING = {
    'task1': 'Task 1: Novel Patients Split',
    'task2': 'Task 2: Known Patients Split',
    1: 'Task 1: Novel Patients Split',
    2: 'Task 2: Known Patients Split',
    '1': 'Task 1: Novel Patients Split',
    '2': 'Task 2: Known Patients Split',
    True: 'Task 2: Known Patients Split',
    False: 'Task 1: Novel Patients Split',
}


def consolidate_ethnicity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Consolidate redundant ethnicity categories.

    Args:
        df: DataFrame with 'ethnicity' column

    Returns:
        DataFrame with consolidated ethnicity values
    """
    df = df.copy()
    df['ethnicity'] = df['ethnicity'].replace(ETHNICITY_MAPPING)
    return df


def compute_rmse_by_model_and_group(
    df: pd.DataFrame,
    group_col: str,
    timestep: int = None,
) -> pd.DataFrame:
    """
    Compute RMSE for each model and group at a specific timestep or averaged across all.

    Args:
        df: DataFrame with model, group_col, label_t0-t11, pred_t0-t11 columns
        group_col: Column to group by
        timestep: Timestep index (0-11), or None to average across all timesteps

    Returns:
        DataFrame with model, group, rmse, rmse_std, n_samples, n_patients columns
    """
    results = []

    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        for group in model_df[group_col].dropna().unique():
            group_df = model_df[model_df[group_col] == group]
            n_samples = len(group_df)
            n_patients = group_df['user_id'].nunique()

            if timestep is not None:
                # Single timestep - compute per-sample RMSE (which is just absolute error)
                label_col = f'label_t{timestep}'
                pred_col = f'pred_t{timestep}'
                squared_errors = (group_df[label_col] - group_df[pred_col]) ** 2
                mse = squared_errors.mean()
                # Standard error of MSE, then propagate to RMSE
                mse_std = squared_errors.std() / np.sqrt(n_samples)
            else:
                # Average across all timesteps, ignoring NaNs from horizons / windows
                # the model didn't predict for.
                sq_err_per_t = []
                for t in range(12):
                    err2 = (group_df[f'label_t{t}'].values - group_df[f'pred_t{t}'].values) ** 2
                    sq_err_per_t.append(err2)
                sq_err = np.stack(sq_err_per_t, axis=0)  # (12, n_samples)
                sample_mse = np.nanmean(sq_err, axis=0)  # per-sample MSE
                n_finite = int(np.isfinite(sample_mse).sum())
                if n_finite == 0:
                    mse = np.nan
                    mse_std = np.nan
                else:
                    mse = float(np.nanmean(sample_mse))
                    mse_std = float(np.nanstd(sample_mse) / np.sqrt(n_finite))

            rmse = np.sqrt(mse)
            # Propagate error: if RMSE = sqrt(MSE), then d(RMSE)/d(MSE) = 1/(2*sqrt(MSE))
            # So std(RMSE) ≈ std(MSE) / (2 * RMSE)
            rmse_std = mse_std / (2 * rmse) if rmse > 0 else 0

            results.append({
                'model': model,
                'group': group,
                'rmse': rmse,
                'rmse_std': rmse_std,
                'n_samples': n_samples,
                'n_patients': n_patients,
            })

    return pd.DataFrame(results)


def plot_rmse_grouped_bars(
    rmse_df: pd.DataFrame,
    group_col: str,
    horizon_label: str,
    output_path: Path,
    title: str = None,
    figsize: tuple = (12, 6),
    dpi: int = 150,
    group_order: list = None,
    bars_by: str = 'model',
    color_by: str = 'arch',
) -> None:
    """
    Create grouped bar chart of RMSE.

    Args:
        rmse_df: DataFrame with model, group, rmse, n_samples columns
        group_col: Name of the grouping column (for labels)
        horizon_label: Prediction horizon label for title (e.g., "30min" or "all")
        output_path: Path to save the plot
        title: Plot title (auto-generated if None)
        figsize: Figure size in inches
        dpi: Resolution for saved figure
        group_order: Optional list specifying the order of groups (e.g., age bins)
        bars_by: 'model' (demographics on x-axis, models as bars) or
                 'demographic' (models on x-axis, demographics as bars)
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Rank models by their mean RMSE across groups (ascending — best first).
    mean_rmse = rmse_df.groupby('model')['rmse'].mean()
    models = list(mean_rmse.sort_values(kind='stable').index)

    # Use specified group order if provided, otherwise sort
    if group_order is not None:
        # Filter to only groups that exist in the data, maintaining specified order
        available_groups = set(rmse_df['group'].unique())
        groups = [g for g in group_order if g in available_groups]
    else:
        groups = sorted(rmse_df['group'].unique(), key=lambda x: (isinstance(x, str), x))

    CI_MULT = 1.96  # 95% CI

    if bars_by == 'model':
        # Demographics on x-axis, models as bars within each group
        n_models = len(models)
        n_groups = len(groups)
        bar_width = 0.8 / n_models
        x = np.arange(n_groups)

        for i, model in enumerate(models):
            model_data = rmse_df[rmse_df['model'] == model]
            rmse_values = []
            rmse_errs = []
            for group in groups:
                group_model_data = model_data[model_data['group'] == group]
                if len(group_model_data) > 0:
                    rmse_values.append(group_model_data['rmse'].values[0])
                    rmse_errs.append(group_model_data['rmse_std'].values[0])
                else:
                    rmse_values.append(0)
                    rmse_errs.append(0)

            errs = CI_MULT * np.asarray(rmse_errs, dtype=float)
            offset = (i - n_models / 2 + 0.5) * bar_width
            bars = ax.bar(x + offset, rmse_values, bar_width,
                          yerr=errs, capsize=3,
                          label=get_model_label(model, color_by=color_by),
                          color=get_model_color_for(model, color_by=color_by),
                          edgecolor='white',
                          error_kw={'elinewidth': 0.8, 'ecolor': '0.2'})
            for bar, val, err in zip(bars, rmse_values, errs):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (err if np.isfinite(err) else 0) + 0.3,
                        f'{val:.1f}', ha='center', va='bottom',
                        rotation=90, fontsize=7)

        ax.set_xlabel(group_col.replace('_', ' ').title(), fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=45, ha='right')
        ax.legend(title='Model', bbox_to_anchor=(1.02, 1), loc='upper left')

    else:
        # Models on x-axis, demographics as bars within each group
        n_models = len(models)
        n_groups = len(groups)
        bar_width = 0.8 / n_groups
        x = np.arange(n_models)

        colors = plt.cm.tab10(np.linspace(0, 1, n_groups))

        for i, (group, color) in enumerate(zip(groups, colors)):
            group_data = rmse_df[rmse_df['group'] == group]
            rmse_values = []
            rmse_errs = []
            for model in models:
                model_group_data = group_data[group_data['model'] == model]
                if len(model_group_data) > 0:
                    rmse_values.append(model_group_data['rmse'].values[0])
                    rmse_errs.append(model_group_data['rmse_std'].values[0])
                else:
                    rmse_values.append(0)
                    rmse_errs.append(0)

            n_samples = group_data['n_samples'].sum() // len(models)
            n_patients = group_data['n_patients'].iloc[0] if len(group_data) > 0 else 0
            label = f"{group} ({n_patients:,} patients, {n_samples:,} samples)"

            errs = CI_MULT * np.asarray(rmse_errs, dtype=float)
            offset = (i - n_groups / 2 + 0.5) * bar_width
            bars = ax.bar(x + offset, rmse_values, bar_width, yerr=errs, capsize=3,
                          label=label, color=color, edgecolor='white',
                          error_kw={'elinewidth': 0.8, 'ecolor': '0.2'})
            for bar, val, err in zip(bars, rmse_values, errs):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (err if np.isfinite(err) else 0) + 0.3,
                        f'{val:.1f}', ha='center', va='bottom',
                        rotation=90, fontsize=7)

        ax.set_xlabel('Model', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend(title=group_col.replace('_', ' ').title(), bbox_to_anchor=(1.02, 1), loc='upper left')

    ax.set_ylabel('RMSE (mg/dL)', fontsize=12)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.18)

    if title is None:
        if horizon_label == 'all':
            title = f'RMSE by {group_col.replace("_", " ").title()} (All Horizons)'
        else:
            title = f'RMSE by {group_col.replace("_", " ").title()} at {horizon_label} Horizon'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot RMSE by model and demographic group at a specific time horizon"
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
        "--horizon",
        type=str,
        required=True,
        choices=['5', '10', '15', '20', '25', '30', '35', '40', '45', '50', '55', '60', 'all'],
        help="Prediction horizon in minutes (5, 10, 15, ..., 60) or 'all' to average across all horizons",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output plot file (auto-generated if not specified)",
    )
    parser.add_argument(
        "--bars-by",
        type=str,
        default='model',
        choices=['model', 'demographic'],
        help="How to group bars: 'model' (demographics on x-axis, models as bars) or 'demographic' (models on x-axis, demographics as bars)",
    )
    add_model_filter_args(parser)
    add_figsize_arg(parser, default=(12.0, 6.0))

    args = parser.parse_args()

    # Convert horizon to timestep index (5min -> 0, 10min -> 1, etc.), or None for 'all'
    if args.horizon == 'all':
        timestep = None
        horizon_label = 'all'
    else:
        timestep = (int(args.horizon) // 5) - 1
        horizon_label = f"{args.horizon}min"

    if args.output is None:
        args.output = Path(f"rmse_by_{args.group_by}_{horizon_label}.png")

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
    print(f"Bar grouping: {args.bars_by}")

    # Handle age binning
    group_col = args.group_by
    if args.group_by == 'age':
        print(f"Binning age into groups: {AGE_LABELS}")
        df['age_group'] = pd.cut(df['age'], bins=AGE_BINS, labels=AGE_LABELS, right=False)
        group_col = 'age_group'

    # Consolidate ethnicity categories
    if args.group_by == 'ethnicity':
        print("Consolidating redundant ethnicity categories...")
        df = consolidate_ethnicity(df)

    # Map subject split labels
    if args.group_by == 'subject_split_across_traintest':
        print("Mapping subject split labels...")
        df['subject_split_across_traintest'] = df['subject_split_across_traintest'].map(
            lambda x: SUBJECT_SPLIT_MAPPING.get(x, x)
        )

    # Check for missing values
    n_missing = df[group_col].isna().sum()
    n_total = len(df)
    print(f"{group_col}: {n_total - n_missing:,} non-null ({100 * (n_total - n_missing) / n_total:.1f}%)")

    # Print group distribution (unique samples and patients)
    n_models = df['model'].nunique()
    print(f"\nDistribution by {group_col}:")
    # Get one model's data to avoid counting duplicates
    first_model = df['model'].iloc[0]
    single_model_df = df[df['model'] == first_model]
    for group in df[group_col].dropna().unique():
        group_df = single_model_df[single_model_df[group_col] == group]
        n_samples = len(group_df)
        n_patients = group_df['user_id'].nunique()
        print(f"  {group}: {n_patients:,} patients, {n_samples:,} samples")

    # Check if we have any data for this demographic
    n_with_data = df[group_col].notna().sum()
    if n_with_data == 0:
        print(f"\nError: No data available for {args.group_by}. Cannot create plot.")
        return 1

    # Compute RMSE by model and group at specified timestep
    if timestep is not None:
        print(f"\nComputing RMSE at {args.horizon}min horizon (timestep {timestep})...")
    else:
        print("\nComputing RMSE averaged across all horizons...")
    rmse_df = compute_rmse_by_model_and_group(df, group_col, timestep)

    if len(rmse_df) == 0:
        print(f"\nError: No RMSE data computed. Check that {args.group_by} has valid values.")
        return 1

    # Print table
    pivot_table = rmse_df.pivot(index='model', columns='group', values='rmse')
    print(f"\nRMSE (mg/dL) at {horizon_label} by model and {args.group_by}:")
    print(pivot_table.round(2).to_string())

    # Plot with proper group ordering
    group_order = AGE_LABELS if args.group_by == 'age' else None
    plot_rmse_grouped_bars(rmse_df, args.group_by, horizon_label, args.output,
                           figsize=tuple(args.figsize),
                           group_order=group_order, bars_by=args.bars_by, color_by=color_by)

    return 0


if __name__ == "__main__":
    exit(main())
