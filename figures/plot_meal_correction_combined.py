#!/usr/bin/env python3
"""
Combined figure: side-by-side RMSE-difference plots for the meal-impact and
corrective-bolus analyses, sharing a single legend.

Left panel  : RMSE(has meal) - RMSE(no meal)            (per model, per horizon)
Right panel : RMSE(with correction) - RMSE(without)     (per model, per horizon)
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model_styles import (
    ARCH_DISPLAY_NAMES,
    ARCH_STYLES,
    FEATURE_COLORS,
    FEATURE_SUFFIX,
    add_model_filter_args,
    apply_model_filter,
    get_arch_legend_handles,
    get_feature_legend_handles,
    get_marker_edge_kwargs,
    get_model_color,
    get_model_color_for,
    get_model_label,
    get_model_linestyle,
    get_model_marker,
    line_style_for,
    sort_models_for_render,
    _parse_model,
)


def compute_rmse(df: pd.DataFrame, pred_col: str, label_col: str) -> float:
    valid = df[pred_col].notna() & df[label_col].notna()
    if valid.sum() == 0:
        return np.nan
    diff = df.loc[valid, pred_col] - df.loc[valid, label_col]
    return np.sqrt((diff ** 2).mean())


def rmse_per_model_per_horizon(cond_df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Return DataFrame indexed by model with one column per horizon."""
    label_cols = [f'label_t{h}' for h in horizons]
    pred_cols = [f'pred_t{h}' for h in horizons]
    sq = (cond_df[label_cols].to_numpy() - cond_df[pred_cols].to_numpy()) ** 2
    sq_df = pd.DataFrame(sq, index=cond_df.index, columns=horizons)
    sq_df['model'] = cond_df['model'].values
    mse = sq_df.groupby('model', observed=True)[horizons].mean()
    return np.sqrt(mse)


def rmse_dict(
    cond_df: pd.DataFrame,
    models: list[str],
    horizons: list[int],
) -> dict[str, list[float]]:
    """Per-model RMSE list (one entry per horizon)."""
    rmse_df = rmse_per_model_per_horizon(cond_df, horizons)
    nan_row = [np.nan] * len(horizons)
    return {
        m: rmse_df.loc[m, horizons].tolist() if m in rmse_df.index else nan_row
        for m in models
    }


def diff_by_model(
    df_pos: pd.DataFrame,
    df_neg: pd.DataFrame,
    models: list[str],
    horizons: list[int],
) -> dict[str, list[float]]:
    """Returns RMSE(df_pos) - RMSE(df_neg) per model per horizon."""
    pos = rmse_dict(df_pos, models, horizons)
    neg = rmse_dict(df_neg, models, horizons)
    return {m: [pos[m][i] - neg[m][i] for i in range(len(horizons))] for m in models}


def _plot_lines(ax, models, horizon_minutes, results, color_by):
    for model in models:
        ax.plot(
            horizon_minutes, results[model],
            label=get_model_label(model, color_by=color_by),
            color=get_model_color_for(model, color_by=color_by),
            **line_style_for(model, color_by),
            **get_marker_edge_kwargs(),
        )


def plot_side_by_side(
    output_path: Path,
    figsize: tuple,
    models: list[str],
    horizon_minutes: list[int],
    color_by: str,
    left_results: dict,
    right_results: dict,
    left_title: str,
    right_title: str,
    ylabel: str,
    suptitle: str,
    ylim: tuple | None,
    show_zero_line: bool,
    show_legend: bool,
    legend_fontsize: int,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)
    _plot_lines(ax1, models, horizon_minutes, left_results, color_by)
    _plot_lines(ax2, models, horizon_minutes, right_results, color_by)

    for ax in (ax1, ax2):
        if show_zero_line:
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Prediction Horizon (minutes)', fontsize=14)
        ax.set_xticks(horizon_minutes)
        ax.grid(alpha=0.3)

    ax1.set_ylabel(ylabel, fontsize=14)
    ax1.set_title(left_title, fontsize=16)
    ax2.set_title(right_title, fontsize=16)

    if ylim is not None:
        ax1.set_ylim(ylim)
        ax2.set_ylim(ylim)

    if suptitle:
        plt.suptitle(suptitle, fontsize=14, y=1.02)
    plt.tight_layout()

    if show_legend:
        add_shared_style_legend(
            fig, ax1,
            color_by=color_by,
            fontsize=legend_fontsize,
            title_fontsize=legend_fontsize + 1,
        )

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def add_shared_style_legend(
    fig,
    ax,
    color_by: str,
    fontsize: int = 13,
    title_fontsize: int = 14,
    y: float = -0.03,
) -> None:
    """One legend below the figure: feature-set + architecture (or per-model fallback)."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return

    if color_by == 'feature':
        parsed = [_parse_model(label) for label in labels]
        plotted_archs = [a for a, _ in parsed]
        plotted_features = [f for _, f in parsed]
        arch_keys = [a for a in ARCH_STYLES.keys() if a in plotted_archs] or list(ARCH_STYLES.keys())
        feat_keys = [k for k in FEATURE_COLORS.keys() if k in plotted_features] or list(FEATURE_COLORS.keys())

        common = dict(
            fontsize=fontsize,
            title_fontsize=title_fontsize,
            framealpha=0.9,
            columnspacing=1.2,
            handletextpad=0.5,
            borderpad=0.5,
            borderaxespad=0.0,
        )
        fig.legend(
            handles=get_feature_legend_handles(feat_keys),
            title='Feature set',
            loc='upper right',
            bbox_to_anchor=(0.49, y),
            ncol=2,
            **common,
        )
        fig.legend(
            handles=get_arch_legend_handles(arch_keys),
            title='Architecture',
            loc='upper left',
            bbox_to_anchor=(0.51, y),
            ncol=min(len(arch_keys), 4),
            **common,
        )
    else:
        ncol = min(len(handles), 4)
        fig.legend(
            handles, labels,
            loc='upper center',
            bbox_to_anchor=(0.5, y),
            ncol=ncol,
            fontsize=fontsize,
            title_fontsize=title_fontsize,
            framealpha=0.9,
        )


def plot_ablation_scatter(
    output_path: Path,
    cohorts: list[tuple[str, pd.DataFrame, pd.DataFrame, str, str]],
    models: list[str],
    horizons: list[int],
    figsize: tuple,
    ylim: tuple | None,
    show_legend: bool,
    legend_fontsize: int,
) -> None:
    """Paired scatter: ``no-event`` vs ``has-event`` mean RMSE per model variant.

    Within each architecture group the 4 feature variants are at small horizontal
    offsets, colored by feature set. At each x-offset two markers are drawn — an
    open circle for the ``no-event`` cohort and a filled circle for the
    ``has-event`` cohort — connected by a dotted vertical line so the gap is
    visually obvious.

    ``cohorts`` items are ``(title, df_neg, df_pos, neg_label, pos_label)``.
    """
    feature_order = ['none', 'carbs', 'insulin', 'insulin+carbs']
    arch_order = ['lstm', 'gluforecast', 'units', 'lightgbm', 'ridge']

    by_arch: dict[str, dict[str, str]] = {}
    for m in models:
        arch, feat = _parse_model(m)
        if arch in arch_order:
            by_arch.setdefault(arch, {})[feat] = m
    archs_present = [a for a in arch_order if a in by_arch]

    if not archs_present:
        print(f"  Skipping ablation scatter: no recognised ablation architectures in {models}")
        return

    n_panels = len(cohorts)
    fig, axes = plt.subplots(1, n_panels, figsize=figsize, sharey=True)
    if n_panels == 1:
        axes = [axes]

    n_groups = len(archs_present)
    arch_spacing = 1.8           # horizontal distance between architecture groups
    spread = 0.55                # total horizontal spread of the 4 feature dots within a group
    bracket_offset = 0.28        # gap between cluster edge and the bracket spine
    bracket_tick = 0.07          # length of the bracket caps
    text_offset = 0.04           # gap between the bracket and its annotation text
    edge_pad = 0.65              # extra x-axis padding so bracket annotations don't overrun
    x_centers = np.arange(n_groups) * arch_spacing
    n_pts = len(feature_order)
    step = spread / max(n_pts - 1, 1)

    def _draw_bracket(ax, x, y_lo, y_hi, label, side, color):
        """Vertical bracket at x spanning [y_lo, y_hi] with caps + annotation."""
        if y_hi - y_lo <= 0:
            return
        if side == 'right':
            # Bracket sits to the right of the cluster; caps point left toward the dots ( ] shape ).
            cap_xs = [x - bracket_tick, x]
            text_x = x + text_offset
            ha = 'left'
        else:
            # Bracket sits to the left of the cluster; caps point right toward the dots ( [ shape ).
            cap_xs = [x, x + bracket_tick]
            text_x = x - text_offset
            ha = 'right'
        ax.plot([x, x], [y_lo, y_hi], color=color, linewidth=1.5, zorder=5)
        ax.plot(cap_xs, [y_lo, y_lo], color=color, linewidth=1.5, zorder=5)
        ax.plot(cap_xs, [y_hi, y_hi], color=color, linewidth=1.5, zorder=5)
        ax.text(
            text_x, (y_lo + y_hi) / 2, label,
            ha=ha, va='center', fontsize=12, color=color, zorder=6,
        )

    for ax, (title, df_neg, df_pos, _neg_label, _pos_label) in zip(axes, cohorts):
        neg_mean = rmse_per_model_per_horizon(df_neg, horizons).mean(axis=1)
        pos_mean = rmse_per_model_per_horizon(df_pos, horizons).mean(axis=1)

        for xc in x_centers:
            ax.axvline(x=xc, color='lightgrey', linewidth=0.6, alpha=0.5, zorder=0)

        # Plot all dots and connectors.
        for i, feat in enumerate(feature_order):
            offset = (i - (n_pts - 1) / 2) * step
            color = FEATURE_COLORS[feat]
            for j, arch in enumerate(archs_present):
                m = by_arch.get(arch, {}).get(feat)
                if m is None:
                    continue
                yn = neg_mean.get(m, np.nan)
                yp = pos_mean.get(m, np.nan)
                x = x_centers[j] + offset
                if not (np.isnan(yn) or np.isnan(yp)):
                    ax.plot(
                        [x, x], [yn, yp],
                        color=color, linestyle=':', linewidth=2.0,
                        alpha=0.85, zorder=2,
                    )
                if not np.isnan(yn):
                    ax.scatter(
                        x, yn,
                        facecolors='white', edgecolors=color,
                        linewidths=2.2, s=95, zorder=3,
                    )
                if not np.isnan(yp):
                    ax.scatter(
                        x, yp,
                        facecolors=color, edgecolors='black',
                        linewidths=0.8, s=95, zorder=4,
                    )

        # Per-architecture brackets: best→worst feature-set spread per cohort.
        bracket_color_neg = '#444444'
        bracket_color_pos = 'black'
        for j, arch in enumerate(archs_present):
            xc = x_centers[j]
            ctrl = [neg_mean.get(by_arch[arch].get(f), np.nan) for f in feature_order]
            ctrl = [v for v in ctrl if not np.isnan(v)]
            evt = [pos_mean.get(by_arch[arch].get(f), np.nan) for f in feature_order]
            evt = [v for v in evt if not np.isnan(v)]
            if len(ctrl) >= 2:
                _draw_bracket(
                    ax, xc - spread / 2 - bracket_offset,
                    min(ctrl), max(ctrl),
                    f'Δ={max(ctrl)-min(ctrl):.2f}',
                    side='left', color=bracket_color_neg,
                )
            if len(evt) >= 2:
                _draw_bracket(
                    ax, xc + spread / 2 + bracket_offset,
                    min(evt), max(evt),
                    f'Δ={max(evt)-min(evt):.2f}',
                    side='right', color=bracket_color_pos,
                )

        ax.set_xticks(x_centers)
        ax.set_xticklabels([ARCH_DISPLAY_NAMES.get(a, a) for a in archs_present], fontsize=14)
        ax.set_title(title, fontsize=16)
        ax.tick_params(axis='y', labelsize=12)
        ax.grid(axis='y', alpha=0.3)
        ax.set_axisbelow(True)
        ax.set_xlim(x_centers[0] - arch_spacing / 2 - edge_pad, x_centers[-1] + arch_spacing / 2 + edge_pad)
        if ylim is not None:
            ax.set_ylim(ylim)

    axes[0].set_ylabel('Mean RMSE across horizons (mg/dL)', fontsize=14)

    plt.tight_layout()
    if show_legend:
        from matplotlib.lines import Line2D

        # Feature-set legend (color encoding)
        fig.legend(
            handles=get_feature_legend_handles(feature_order),
            title='Feature set',
            loc='upper right',
            bbox_to_anchor=(0.49, -0.02),
            ncol=2,
            fontsize=legend_fontsize,
            title_fontsize=legend_fontsize + 1,
            framealpha=0.9,
            columnspacing=1.2,
            handletextpad=0.5,
        )
        # Marker-style legend (open = control, filled = event, dotted = difference).
        marker_handles = [
            Line2D([0], [0], marker='o', linestyle='', color='black',
                   markerfacecolor='white', markeredgecolor='black',
                   markersize=9, markeredgewidth=1.6, label='Control'),
            Line2D([0], [0], marker='o', linestyle='', color='black',
                   markerfacecolor='dimgray', markeredgecolor='black',
                   markersize=9, markeredgewidth=0.5, label='Event'),
            Line2D([0], [0], linestyle=':', color='black', linewidth=1.2,
                   label='difference'),
        ]
        fig.legend(
            handles=marker_handles,
            title='Cohort',
            loc='upper left',
            bbox_to_anchor=(0.51, -0.02),
            ncol=3,
            fontsize=legend_fontsize,
            title_fontsize=legend_fontsize + 1,
            framealpha=0.9,
            columnspacing=1.2,
            handletextpad=0.5,
        )

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Combined meal/correction RMSE-difference figure with a shared legend"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("combined_results_new.parquet"),
        help="Input parquet file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("rmse_meal_correction_diff.png"),
        help="Output plot file",
    )
    add_model_filter_args(parser)
    parser.add_argument(
        "--ylim",
        type=float,
        nargs=2,
        default=None,
        metavar=('YMIN', 'YMAX'),
        help="Shared y-limits for the diff panels (e.g. --ylim -2 18)",
    )
    parser.add_argument(
        "--ylim-meal",
        type=float,
        nargs=2,
        default=None,
        metavar=('YMIN', 'YMAX'),
        help="Y-limits for the meal absolute side-by-side plot (no meal | has meal)",
    )
    parser.add_argument(
        "--ylim-correction",
        type=float,
        nargs=2,
        default=None,
        metavar=('YMIN', 'YMAX'),
        help="Y-limits for the correction absolute side-by-side plot (no corr | with corr)",
    )
    parser.add_argument(
        "--ylim-ablation",
        type=float,
        nargs=2,
        default=None,
        metavar=('YMIN', 'YMAX'),
        help="Y-limits for the per-architecture ablation scatter",
    )
    parser.add_argument(
        "--figsize-ablation",
        type=float,
        nargs=2,
        default=[12.0, 4.5],
        metavar=('W', 'H'),
        help="Figure size (inches) for the ablation scatter (default: 12 4.5)",
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
        default=[14.0, 5.0],
        metavar=('W', 'H'),
        help="Figure size (inches) for the side-by-side plot (default: 14 5)",
    )
    parser.add_argument(
        "--no-legend",
        action="store_true",
        help="Suppress the shared legend",
    )
    parser.add_argument(
        "--legend-fontsize",
        type=int,
        default=13,
        help="Font size for the shared legend body (default: 13)",
    )
    args = parser.parse_args()

    h_min_min, h_max_min = args.horizons
    if h_min_min < 5 or h_max_min > 60 or h_min_min % 5 or h_max_min % 5 or h_min_min > h_max_min:
        raise SystemExit(f"--horizons must be multiples of 5 in [5, 60] with MIN_MIN <= MAX_MIN; got {args.horizons}")

    print(f"Loading data from {args.input}...")
    df = pd.read_parquet(args.input)
    print(f"  Loaded {len(df):,} rows")

    models, color_by = apply_model_filter(df['model'].unique().tolist(), args)
    models = sort_models_for_render(models)
    df = df[df['model'].isin(models)]
    print(f"  Models: {models}")

    horizons = list(range((h_min_min // 5) - 1, (h_max_min // 5)))
    horizon_minutes = [(h + 1) * 5 for h in horizons]

    # ---- Meal subgroup ----
    carb_columns = ['carbs_tminus_1', 'carbs_tminus_2', 'carbs_tminus_3',
                    'carbs_tminus_4', 'carbs_tminus_5']
    has_all_carbs = df[carb_columns].notna().all(axis=1)
    df_with_carbs = df[has_all_carbs].copy()
    no_meal_mask = (df_with_carbs[carb_columns] <= 0).all(axis=1)
    df_no_meal = df_with_carbs[no_meal_mask]
    df_has_meal = df_with_carbs[(df_with_carbs[carb_columns] > 0).any(axis=1)]
    n_models = max(len(models), 1)
    print(f"  Meal cohort rows: no-meal={len(df_no_meal):,}, has-meal={len(df_has_meal):,} "
          f"(per-model ≈ {len(df_no_meal)//n_models:,} / {len(df_has_meal)//n_models:,})")

    meal_no = rmse_dict(df_no_meal, models, horizons)
    meal_yes = rmse_dict(df_has_meal, models, horizons)
    meal_diff = {m: [meal_yes[m][i] - meal_no[m][i] for i in range(len(horizons))] for m in models}

    # ---- Correction subgroup ----
    insulin_columns = ['insulin_tminus_1', 'insulin_tminus_2',
                       'insulin_tminus_3', 'insulin_tminus_4', 'insulin_tminus_5']
    has_bg = df['cgm_at_t0'].notna()
    has_insulin = df[insulin_columns].notna().all(axis=1)
    df_valid = df[has_bg & has_insulin].copy()
    df_hyper = df_valid[df_valid['cgm_at_t0'] > 250]
    has_correction = (df_hyper[insulin_columns] > 2).any(axis=1)
    df_hyper_with_correction = df_hyper[has_correction]
    df_hyper_without_correction = df_hyper[~has_correction]
    print(f"  Hyperglycemia cohort rows: with-corr={len(df_hyper_with_correction):,}, no-corr={len(df_hyper_without_correction):,} "
          f"(per-model ≈ {len(df_hyper_with_correction)//n_models:,} / {len(df_hyper_without_correction)//n_models:,})")

    corr_no = rmse_dict(df_hyper_without_correction, models, horizons)
    corr_yes = rmse_dict(df_hyper_with_correction, models, horizons)
    corr_diff = {m: [corr_yes[m][i] - corr_no[m][i] for i in range(len(horizons))] for m in models}

    # ---- Plots ----
    figsize = tuple(args.figsize)
    show_legend = not args.no_legend

    # 1) Combined diff: meal-impact diff | correction-impact diff
    plot_side_by_side(
        output_path=args.output,
        figsize=figsize,
        models=models,
        horizon_minutes=horizon_minutes,
        color_by=color_by,
        left_results=meal_diff,
        right_results=corr_diff,
        left_title='Impact of Recent Meal\n(Has Meal − No Meal)',
        right_title='Impact of Corrective Bolus During Hyperglycemia\n(With Correction − Without)',
        ylabel='RMSE Difference (mg/dL)',
        suptitle='',
        ylim=tuple(args.ylim) if args.ylim is not None else None,
        show_zero_line=True,
        show_legend=show_legend,
        legend_fontsize=args.legend_fontsize,
    )

    # 2) Meal absolute side-by-side: No Meal | Has Meal
    n_no_meal = len(df_no_meal) // n_models
    n_has_meal = len(df_has_meal) // n_models
    plot_side_by_side(
        output_path=args.output.with_stem(args.output.stem + '_meal'),
        figsize=figsize,
        models=models,
        horizon_minutes=horizon_minutes,
        color_by=color_by,
        left_results=meal_no,
        right_results=meal_yes,
        left_title=f'No Recent Meal\n(carbs = 0 for t-5 to t-25min)\nn = {n_no_meal:,} per model',
        right_title=f'Recent Meal Present\n(carbs > 0 for at least one of t-5 to t-25min)\nn = {n_has_meal:,} per model',
        ylabel='RMSE (mg/dL)',
        suptitle='Model RMSE by Prediction Horizon: Meal vs No Meal',
        ylim=tuple(args.ylim_meal) if args.ylim_meal is not None else None,
        show_zero_line=False,
        show_legend=show_legend,
        legend_fontsize=args.legend_fontsize,
    )

    # 3) Correction absolute side-by-side: No Correction | With Correction
    n_no_corr = len(df_hyper_without_correction) // n_models
    n_with_corr = len(df_hyper_with_correction) // n_models
    plot_side_by_side(
        output_path=args.output.with_stem(args.output.stem + '_correction'),
        figsize=figsize,
        models=models,
        horizon_minutes=horizon_minutes,
        color_by=color_by,
        left_results=corr_no,
        right_results=corr_yes,
        left_title=f'Hyperglycemia Without Correction\n(BG > 250, insulin ≤ 2u at t-5 to t-25min)\nn = {n_no_corr:,} per model',
        right_title=f'Hyperglycemia With Correction\n(BG > 250, insulin > 2u at t-5 to t-25min)\nn = {n_with_corr:,} per model',
        ylabel='RMSE (mg/dL)',
        suptitle='Model RMSE During Hyperglycemia: With vs Without Corrective Bolus',
        ylim=tuple(args.ylim_correction) if args.ylim_correction is not None else None,
        show_zero_line=False,
        show_legend=show_legend,
        legend_fontsize=args.legend_fontsize,
    )

    # 4) Paired ablation scatter: open marker = control, filled marker = event,
    #    dotted connector = difference, color = feature set, group = architecture.
    meal_count = len(df_has_meal) / len(models)
    no_meal_count = len(df_no_meal) / len(models)
    hyper_count = len(df_hyper_without_correction) / len(models)
    correction_count = len(df_hyper_with_correction) / len(models)
    plot_ablation_scatter(
        output_path=args.output.with_stem(args.output.stem + '_ablation'),
        cohorts=[
            (f'Meals n={meal_count} (Nonzero carbs in past 30m) vs \nNon Meals n={no_meal_count} (Zero carbs in past 30m)',
             df_no_meal,
             df_has_meal,
             'No meal',
             'Has meal'),
            (f'Correction n={correction_count} (Bolus 2U+ and BG > 250) vs \nNo Correction n={hyper_count} (BG > 250)',
             df_hyper_without_correction,
             df_hyper_with_correction,
             'No correction',
             'Has correction'),
        ],
        models=models,
        horizons=horizons,
        figsize=tuple(args.figsize_ablation),
        ylim=tuple(args.ylim_ablation) if args.ylim_ablation is not None else None,
        show_legend=show_legend,
        legend_fontsize=args.legend_fontsize,
    )

    return 0


if __name__ == "__main__":
    exit(main())
