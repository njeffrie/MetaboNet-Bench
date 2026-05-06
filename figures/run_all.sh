#!/usr/bin/env bash
# Generate every figure in the paper from a single combined-results parquet.
#
# Usage: bash figures/run_all.sh [INPUT_PARQUET] [OUT_DIR]
#   INPUT_PARQUET defaults to combined_results_with_aux.parquet
#   OUT_DIR       defaults to figures/out (created if missing)

set -euo pipefail

INPUT="${1:-combined_results_with_aux.parquet}"
OUT_DIR="${2:-figures/out}"

if [[ ! -f "$INPUT" ]]; then
    echo "Error: input parquet not found: $INPUT" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"
echo "Writing figures to: $OUT_DIR"

# Figure 3
python figures/plot_rmse_by_bg.py --input "$INPUT" --combined --output "$OUT_DIR/rmse_vs_bg_h30.png" --horizon 30 --ylim 0 70 --figsize-support 7 5 --figsize 7 5 --legend none
python figures/plot_rmse_by_bg.py --input "$INPUT" --combined --output "$OUT_DIR/rmse_vs_bg_h60.png" --horizon 60 --ylim 0 70 --figsize-support 7 5 --figsize 7 5 --legend right

# Figure 4 + Supplemental D3 Figure 8
python figures/plot_rmse_by_meal_presence.py --input "$INPUT" --output "$OUT_DIR/results_metabonet_bench_meal.png" --exclude-models zoh le gluformer --ablation only  --figsize 10 4 --figsize-diff 7 5
python figures/plot_rmse_by_correction.py --input "$INPUT" --output "$OUT_DIR/results_metabonet_bench_correction.png" --exclude-models zoh le gluformer --ablation only --figsize 10 4 --figsize-diff 7 5

# Figure 4 combined: meal-impact + correction-impact diff side-by-side, shared legend
python figures/plot_meal_correction_combined.py --input combined_results_with_aux.parquet --exclude-models zoh le gluformer --ablation only --figsize 17 5 --legend-fontsize 16 --figsize-ablation 17 6

# Figure 2 right
python figures/plot_model_rmse.py --input "$INPUT" --output "$OUT_DIR/model_rmse_by_timestep.png" --ablation exclude

# Supplemental D1 Figure 5
python figures/plot_signed_error_dist.py --input "$INPUT" --output "$OUT_DIR/signed_error_ridge.png" --plot-type ridge --ablation exclude

# Supplemental D2 figure 6
python figures/plot_rmse_heatmap.py --input "$INPUT" --output "$OUT_DIR/rmse_heatmap_bg_horizon_d2.png" --ablation exclude

# RMSE by Task 1/Task 2 (subject split) across all horizons
python figures/plot_rmse_by_demographic.py --input "$INPUT" --group-by subject_split_across_traintest --horizon all --bars-by demographic --output "$OUT_DIR/rmse_by_subject_split.png" --ablation exclude

# RMSE by gender across all horizons
python figures/plot_rmse_by_demographic.py --input "$INPUT" --group-by gender --horizon all --output "$OUT_DIR/rmse_by_gender.png" --ablation exclude

# RMSE binned by age / weight / height (one line per model)
python figures/plot_rmse_by_variable.py --input "$INPUT" --variable age    --output "$OUT_DIR/rmse_by_age.png"    --ablation exclude --figsize 7 4.5
python figures/plot_rmse_by_variable.py --input "$INPUT" --variable weight --output "$OUT_DIR/rmse_by_weight.png" --ablation exclude --figsize 7 4.5
python figures/plot_rmse_by_variable.py --input "$INPUT" --variable height --output "$OUT_DIR/rmse_by_height.png" --ablation exclude --figsize 7 4.5
