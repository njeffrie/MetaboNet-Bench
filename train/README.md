# Training (feature ablation)

Scripts for Optuna search and training LSTM, UniTS, and GluForecast on MetaboNet with different input feature sets.

## Prerequisites

From the repository root, install dependencies ([requirements.txt](../requirements.txt)) and download the MetaboNet train split from [metabo-net.org](https://metabo-net.org/).

## Workflow

1. Place the train split at `data/metabonet_public_train.parquet`.

2. Preprocess:

```bash
python data/preprocess.py --path_to_dataset data/metabonet_public_train.parquet
```

This writes `data/metabonet_train.parquet`.

3. Optuna hyperparameter search:

```bash
python -m train.optuna_search \
  --data_path data/metabonet_train.parquet \
  --device cuda
```

4. Train from Optuna outputs:

```bash
python -m train.train \
  --data_path data/metabonet_train.parquet \
  --optuna_dir train/optuna \
  --device cuda
```

Optional flags (CUDA): `--amp --tf32` on both commands.

Checkpoints are written under `checkpoints/` at the repo root (layout may include a nested `checkpoints/` directory depending on your setup). The benchmark loads ablation weights from `checkpoints/` via [models/models.py](../models/models.py).

## Convenience script

```bash
./train/run_budget_96h.sh
```

Override paths and device via environment variables documented in that script.
