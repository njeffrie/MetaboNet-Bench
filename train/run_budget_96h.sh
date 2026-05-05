#!/usr/bin/env bash
# Optuna hyperparameter search then feature ablation training.
# Optuna tunes lr, batch_size, weight_decay only; one tune pass per
# --tune_feature_sets entry under OUT_DIR/<feature_set>/.
#
# Override via env: DATA_PATH OUT_DIR DEVICE MAX_EPOCHS_TRIAL ABLATION_EPOCHS N_TRIALS
# N_TRIALS_LSTM N_TRIALS_UNITS N_TRIALS_GLUFORECAST (optional, asymmetric Optuna trials)
# TUNE_FEATURE_SETS (optional, comma list; default: all four ablations)
# MODELS (optional, default: lstm,units,gluforecast)

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_PATH="${DATA_PATH:-data/metabonet_train.parquet}"
OUT_DIR="${OUT_DIR:-train/optuna}"
DEVICE="${DEVICE:-cuda}"
MAX_EPOCHS_TRIAL="${MAX_EPOCHS_TRIAL:-10}"
ABLATION_EPOCHS="${ABLATION_EPOCHS:-12}"
N_TRIALS="${N_TRIALS:-6}"
TUNE_FEATURE_SETS="${TUNE_FEATURE_SETS:-cgm,cgm_insulin,cgm_carbs,cgm_insulin_carbs}"
MODELS="${MODELS:-lstm,units,gluforecast}"

SEARCH_ARGS=(
  --data_path "$DATA_PATH"
  --device "$DEVICE"
  --n_trials "$N_TRIALS"
  --models "$MODELS"
  --max_epochs_per_trial "$MAX_EPOCHS_TRIAL"
  --out_dir "$OUT_DIR"
  --tune-feature-sets "$TUNE_FEATURE_SETS"
  --amp
  --tf32
)
if [[ -n "${N_TRIALS_LSTM:-}" ]]; then
  SEARCH_ARGS+=(--n-trials-lstm "$N_TRIALS_LSTM")
fi
if [[ -n "${N_TRIALS_UNITS:-}" ]]; then
  SEARCH_ARGS+=(--n-trials-units "$N_TRIALS_UNITS")
fi
if [[ -n "${N_TRIALS_GLUFORECAST:-}" ]]; then
  SEARCH_ARGS+=(--n-trials-gluforecast "$N_TRIALS_GLUFORECAST")
fi

echo "=== Optuna search ==="
python -m train.optuna_search "${SEARCH_ARGS[@]}"

echo ""
echo "=== Ablation training (matching --epochs to search cap is recommended) ==="
python -m train.train \
  --optuna_dir "$OUT_DIR" \
  --data_path "$DATA_PATH" \
  --models "$MODELS" \
  --device "$DEVICE" \
  --epochs "$ABLATION_EPOCHS" \
  --amp \
  --tf32
