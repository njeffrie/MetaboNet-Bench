"""Shared model colors and styles for consistent visualizations.

Encoding:
  color    → feature set (what inputs the model uses beyond CGM)
  marker   → model architecture
  linestyle→ model architecture (same axis as marker)
"""

import re

# Color by feature set
FEATURE_COLORS = {
    'none':          '#1f77b4',  # blue       — CGM only
    'carbs':         '#ff7f0e',  # orange     — CGM + carbs
    'insulin':       '#d62728',  # red        — CGM + insulin
    'insulin+carbs': '#9467bd',  # purple     — CGM + insulin + carbs
}

# Models used for the feature-set ablation study (same architecture, varying
# inputs). Every architecture that has a feature-set ablation contributes 4
# variants — CGM only, +carbs, +insulin, +insulin+carbs (full reference) —
# although the naming convention varies slightly across archs (some use
# ``-cgm-<feat>``, others just ``-<feat>``).
#
# ABLATION_MODELS         -> all 4 variants per arch (used by --ablation only).
# ABLATION_MODELS_PARTIAL -> the 3 sub-variants per arch, *not* the full
#                            *-insulin-carbs reference. --ablation exclude
#                            drops these so the best variant per arch survives
#                            alongside the non-ablation baselines (le, zoh, gluformer).
ABLATION_MODELS = [
    # gluforecast (uses '-cgm-<feat>' naming)
    'gluforecast-cgm',
    'gluforecast-cgm-carbs',
    'gluforecast-cgm-insulin',
    'gluforecast-cgm-insulin-carbs',
    # lstm (uses '-cgm-<feat>' naming)
    'lstm-cgm',
    'lstm-cgm-carbs',
    'lstm-cgm-insulin',
    'lstm-cgm-insulin-carbs',
    # units (uses '-cgm-<feat>' naming)
    'units-cgm',
    'units-cgm-carbs',
    'units-cgm-insulin',
    'units-cgm-insulin-carbs',
    # lightgbm (uses '-<feat>' naming, with '-cgm' for CGM-only)
    'lightgbm-cgm',
    'lightgbm-carbs',
    'lightgbm-insulin',
    'lightgbm-insulin-carbs',
    # ridge (uses '-<feat>' naming, with '-cgm' for CGM-only)
    'ridge-cgm',
    'ridge-carbs',
    'ridge-insulin',
    'ridge-insulin-carbs',
]
ABLATION_MODELS_PARTIAL = [
    'gluforecast-cgm',
    'gluforecast-cgm-carbs',
    'gluforecast-cgm-insulin',
    'lstm-cgm',
    'lstm-cgm-carbs',
    'lstm-cgm-insulin',
    'units-cgm',
    'units-cgm-carbs',
    'units-cgm-insulin',
    'lightgbm-cgm',
    'lightgbm-carbs',
    'lightgbm-insulin',
    'ridge-cgm',
    'ridge-carbs',
    'ridge-insulin',
]


def filter_ablation(models, mode: str):
    """Apply ablation filtering.

    - 'only'    -> keep ABLATION_MODELS (all 4 variants per ablation arch).
    - 'exclude' -> drop ABLATION_MODELS_PARTIAL (the 3 sub-variants per arch),
                   leaving the *-insulin-carbs full reference per ablation arch
                   plus the non-ablation baselines (le, zoh, gluformer).
    - 'all'     -> no-op.
    """
    if mode == 'only':
        return [m for m in models if m in ABLATION_MODELS]
    if mode == 'exclude':
        return [m for m in models if m not in ABLATION_MODELS_PARTIAL]
    return list(models)


FEATURE_SUFFIX = {
    'none':          'CGM only',
    'carbs':         'CGM + carbs',
    'insulin':       'CGM + insulin',
    'insulin+carbs': 'CGM + insulin + carbs',
}


def select_full_per_arch(models):
    """Pick one model per architecture, preferring the most-featured variant.

    Priority: insulin+carbs > insulin > carbs > none. Ties broken by shorter name.
    """
    rank = {'insulin+carbs': 3, 'insulin': 2, 'carbs': 1, 'none': 0}
    by_arch = {}
    for m in models:
        arch, feat = _parse_model(m)
        key = (rank.get(feat, -1), -len(m), m)
        if arch not in by_arch or key > by_arch[arch]:
            by_arch[arch] = key
    return [v[2] for v in by_arch.values()]


def add_model_filter_args(parser, default_excludes=('glucose_decoder', 'gluformer-tiny')):
    """Add the standard --exclude-models / --all-variants / --ablation CLI args."""
    parser.add_argument(
        "--exclude-models", type=str, nargs='*', default=list(default_excludes),
        help=f"Models to exclude from plot (default: {', '.join(default_excludes)})",
    )
    parser.add_argument(
        "--all-variants", action='store_true',
        help="Show every model variant. Default keeps only the fullest variant per architecture.",
    )
    parser.add_argument(
        "--ablation", choices=['all', 'only', 'exclude'], default='all',
        help="Filter for the feature-set ablation set (ABLATION_MODELS). "
             "'only' keeps just those (colored by feature set); 'exclude' drops them; 'all' (default) leaves them in.",
    )


def apply_model_filter(models, args):
    """Apply the standard filtering described by add_model_filter_args.

    Returns (sorted_filtered_model_list, color_by) where color_by is 'arch' or 'feature'.
    """
    excludes = set(getattr(args, 'exclude_models', []) or [])
    out = [m for m in models if m not in excludes]
    if args.ablation != 'all':
        out = filter_ablation(out, args.ablation)
    if not args.all_variants and args.ablation != 'only':
        out = select_full_per_arch(out)
    color_by = 'feature' if args.ablation == 'only' else 'arch'
    return sorted(out), color_by


def get_model_color_for(model: str, color_by: str = 'arch') -> str:
    """Return color for a model under the requested scheme ('arch' or 'feature')."""
    return get_model_color(model) if color_by == 'feature' else get_model_color_by_arch(model)


# Display-name override for special model IDs.
_MODEL_DISPLAY_NAMES = {
    'glucose_decoder': 'GluForecast',
}


def get_model_label(model: str, color_by: str = 'arch') -> str:
    """Legend label for a model. Adds a feature-set suffix when color_by='feature'."""
    if model in _MODEL_DISPLAY_NAMES:
        return _MODEL_DISPLAY_NAMES[model]
    arch, feat = _parse_model(model)
    base = ARCH_DISPLAY_NAMES.get(arch, model)
    if color_by == 'feature':
        suffix = FEATURE_SUFFIX.get(feat)
        return f"{base} ({suffix})" if suffix else base
    return base


def add_figsize_arg(parser, default=(10.0, 6.0), name='--figsize', help_suffix=''):
    """Add a --figsize W H CLI flag to ``parser``."""
    parser.add_argument(
        name, type=float, nargs=2, default=list(default), metavar=('W', 'H'),
        help=f"Figure size in inches as W H (default: {default[0]} {default[1]}){help_suffix}",
    )


def render_order_key(model: str) -> int:
    """Lower keys draw first (behind). Gluforecast variants always go last so they sit on top."""
    arch, _ = _parse_model(model)
    return 1 if arch == 'gluforecast' else 0


def sort_models_for_render(models, secondary_key=None):
    """Sort models so gluforecast variants render last (on top).

    secondary_key(model) -> any sortable; lower => drawn earlier within the same group.
    Defaults to alphabetical by name.
    """
    if secondary_key is None:
        secondary_key = lambda m: m
    return sorted(models, key=lambda m: (render_order_key(m), secondary_key(m)))


def add_model_legend_below(fig, ax, ncol=None):
    """Replace per-axes legend with a single model legend below the figure."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    if ncol is None:
        ncol = min(len(handles), 3)
    fig.legend(
        handles, labels,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.02),
        ncol=ncol,
        fontsize=10,
        framealpha=0.9,
    )

# Color by architecture (alternative scheme for plots that show only one variant per arch).
# `le` is the new short name for the linear baseline; we keep `linear` styled too
# so older combined parquets still render with the right color.
ARCH_COLORS = {
    'lstm':        '#1f77b4',  # blue
    'gluforecast': '#9467bd',  # purple
    'units':       '#2ca02c',  # green
    'gluformer':   '#ff7f0e',  # orange
    'lightgbm':    '#8c564b',  # brown
    'linear':      '#e377c2',  # pink
    'le':          '#e377c2',  # pink (renamed from 'linear')
    'ridge':       '#7f7f7f',  # grey
    'zoh':         '#17becf',  # cyan
}

# Marker and linestyle by architecture
ARCH_STYLES = {
    'lstm':        {'marker': 'o', 'linestyle': '-'},
    'gluforecast': {'marker': 's', 'linestyle': '--'},
    'units':       {'marker': 'D', 'linestyle': '-.'},
    'gluformer':   {'marker': '^', 'linestyle': ':'},
    'lightgbm':    {'marker': 'v', 'linestyle': (0, (3, 1, 1, 1))},        # dashdotted
    'linear':      {'marker': 'P', 'linestyle': (0, (5, 1))},              # densely dashed
    'le':          {'marker': 'P', 'linestyle': (0, (5, 1))},              # densely dashed (renamed from 'linear')
    'ridge':       {'marker': 'X', 'linestyle': (0, (1, 1))},              # densely dotted
    'zoh':         {'marker': '*', 'linestyle': (0, (3, 5, 1, 5, 1, 5))},  # dashdotdotted
}

# Thin black outline on every marker so distinct shapes are easier to read
MARKER_EDGE_COLOR = 'black'
MARKER_EDGE_WIDTH = 0.5

ARCH_DISPLAY_NAMES = {
    'lstm':        'LSTM',
    'gluforecast': 'GluForecast',
    'units':       'UNITS',
    'gluformer':   'Gluformer',
    'lightgbm':    'LightGBM',
    'linear':      'Linear',
    'le':          'LE',
    'ridge':       'Ridge',
    'zoh':         'ZOH',
}

# Match longest names first so 'gluforecast' wins over 'gluformer' if both ever appeared
_ARCH_KEYS_BY_LENGTH = sorted(ARCH_STYLES.keys(), key=len, reverse=True)


def _parse_model(model: str) -> tuple[str, str]:
    """Return (arch, feature_key) for a model name like 'lstm-cgm-insulin-carbs' or 'lightgbm-all'.

    Recognized feature tokens:
      - 'insulin'                       → adds insulin
      - 'carbs'                         → adds carbs
      - 'all' / 'full'                  → both insulin and carbs
      - any other → CGM only (color 'none')
    """
    # Split on dashes/underscores and also whitespace, parens, +, /, comma.
    # The wider delimiter set lets us parse display labels like
    # "LSTM (CGM + insulin)" the same as the canonical id "lstm-cgm-insulin",
    # which matters when add_legends_below() introspects ax legend labels.
    parts = set(re.split(r'[-_\s()+/,]+', model.lower()))
    parts.discard('')

    arch = next((a for a in _ARCH_KEYS_BY_LENGTH if a in parts), 'unknown')

    all_features = bool(parts & {'all', 'full'})
    has_insulin = all_features or 'insulin' in parts
    has_carbs = all_features or 'carbs' in parts
    if has_insulin and has_carbs:
        feature_key = 'insulin+carbs'
    elif has_insulin:
        feature_key = 'insulin'
    elif has_carbs:
        feature_key = 'carbs'
    else:
        feature_key = 'none'

    return arch, feature_key


def get_model_style(model: str) -> dict:
    """Return dict with color, marker, linestyle, and marker-edge attrs for a model."""
    arch, feature_key = _parse_model(model)
    arch_style = ARCH_STYLES.get(arch, {'marker': 'x', 'linestyle': ':'})
    return {
        'color':           FEATURE_COLORS.get(feature_key, '#000000'),
        'marker':          arch_style['marker'],
        'linestyle':       arch_style['linestyle'],
        'markeredgecolor': MARKER_EDGE_COLOR,
        'markeredgewidth': MARKER_EDGE_WIDTH,
    }


def get_marker_edge_kwargs() -> dict:
    """Return kwargs to splat into plot/errorbar/scatter for a thin black marker outline."""
    return {
        'markeredgecolor': MARKER_EDGE_COLOR,
        'markeredgewidth': MARKER_EDGE_WIDTH,
    }


def get_model_color(model: str) -> str:
    """Get color for a model (encodes feature set)."""
    _, feature_key = _parse_model(model)
    return FEATURE_COLORS.get(feature_key, '#000000')


def get_model_color_by_arch(model: str) -> str:
    """Get color for a model keyed on architecture (use when plotting one variant per arch)."""
    arch, _ = _parse_model(model)
    return ARCH_COLORS.get(arch, '#000000')


def get_model_marker(model: str) -> str:
    """Get marker for a model (encodes architecture)."""
    arch, _ = _parse_model(model)
    return ARCH_STYLES.get(arch, {'marker': 'x'})['marker']


def get_model_linestyle(model: str) -> str:
    """Get linestyle for a model (encodes architecture)."""
    arch, _ = _parse_model(model)
    return ARCH_STYLES.get(arch, {'linestyle': ':'})['linestyle']


def get_feature_legend_handles(features=None):
    """Return Line2D handles explaining the color → feature-set encoding.

    Pass an iterable of feature keys (subset of FEATURE_COLORS) to limit the legend
    to feature sets actually plotted; otherwise all four are shown.
    """
    from matplotlib.lines import Line2D
    labels = {
        'none':          'CGM only',
        'carbs':         'CGM + carbs',
        'insulin':       'CGM + insulin',
        'insulin+carbs': 'CGM + insulin + carbs',
    }
    keys = list(features) if features is not None else list(labels.keys())
    return [
        Line2D([0], [0], color=FEATURE_COLORS[k], linewidth=3, label=labels[k])
        for k in keys if k in labels
    ]


def get_arch_legend_handles(archs=None):
    """Return Line2D handles explaining the marker/linestyle → architecture encoding.

    Pass an iterable of arch keys to limit the legend to architectures actually plotted;
    otherwise all known architectures are shown.
    """
    from matplotlib.lines import Line2D
    keys = list(archs) if archs is not None else list(ARCH_STYLES.keys())
    return [
        Line2D([0], [0], color='grey',
               marker=ARCH_STYLES[k]['marker'],
               linestyle=ARCH_STYLES[k]['linestyle'],
               markeredgecolor=MARKER_EDGE_COLOR,
               markeredgewidth=MARKER_EDGE_WIDTH,
               linewidth=2, markersize=6, label=ARCH_DISPLAY_NAMES.get(k, k))
        for k in keys if k in ARCH_STYLES
    ]


def add_style_legends(ax, feature_loc='lower right', arch_loc='lower center'):
    """Add color-key and architecture-key legends to ax as separate artists."""
    import matplotlib.pyplot as plt
    feat_legend = ax.legend(
        handles=get_feature_legend_handles(),
        title='Feature set',
        loc=feature_loc,
        fontsize=8,
        title_fontsize=8,
        framealpha=0.9,
    )
    ax.add_artist(feat_legend)
    arch_legend = ax.legend(
        handles=get_arch_legend_handles(),
        title='Architecture',
        loc=arch_loc,
        fontsize=8,
        title_fontsize=8,
        framealpha=0.9,
    )
    ax.add_artist(arch_legend)


def add_legends_below(fig, ax, model_ncol: int = 6, include_model_legend: bool = False):
    """Place the feature-set and architecture legends side-by-side below the figure.

    Side-by-side (feature on the left half, architecture on the right half) avoids
    the vertical-spacing fragility of stacked legends on short figures. Optionally
    a model legend can be added as a row above; it's off by default.

    The feature-set and architecture legends are restricted to only the values
    actually represented in the labels on `ax` (so excluded models don't appear).
    Save with `fig.savefig(..., bbox_inches='tight')` so the bottom-anchored legends
    are not clipped from the rendered file.
    """
    common = dict(
        fontsize=8,
        title_fontsize=9,
        framealpha=0.9,
        columnspacing=1.0,
        handletextpad=0.4,
        borderpad=0.4,
        borderaxespad=0.0,
    )

    handles, labels = ax.get_legend_handles_labels()

    parsed = [_parse_model(label) for label in labels] if labels else []
    plotted_archs = [a for a, _ in parsed]
    plotted_features = [f for _, f in parsed]

    arch_keys_present = [a for a in ARCH_STYLES.keys() if a in plotted_archs] or list(ARCH_STYLES.keys())
    feat_keys_present = [k for k in FEATURE_COLORS.keys() if k in plotted_features] or list(FEATURE_COLORS.keys())

    # Default y position for the feature/arch row.
    y_pair = -0.02

    # Optional model legend stacked above the pair.
    if include_model_legend and handles:
        n_rows_model = -(-len(handles) // model_ncol)
        fig.legend(
            handles, labels,
            title='Model',
            loc='upper center',
            bbox_to_anchor=(0.5, y_pair),
            ncol=model_ncol,
            **common,
        )
        # Push the feature/arch row down by enough rows to clear the model legend.
        # Convert legend row height (~0.22 inches at fontsize 8) to figure fraction.
        fig_h = fig.get_size_inches()[1]
        row_h_frac = 0.22 / fig_h
        y_pair = y_pair - row_h_frac * (n_rows_model + 1.5)

    # Feature set legend: anchored so its top-right sits just left of the figure midline.
    fig.legend(
        handles=get_feature_legend_handles(feat_keys_present),
        title='Feature set',
        loc='upper right',
        bbox_to_anchor=(0.49, y_pair),
        ncol=2,
        **common,
    )
    # Architecture legend: anchored so its top-left sits just right of the midline.
    fig.legend(
        handles=get_arch_legend_handles(arch_keys_present),
        title='Architecture',
        loc='upper left',
        bbox_to_anchor=(0.51, y_pair),
        ncol=min(len(arch_keys_present), 4),
        **common,
    )
