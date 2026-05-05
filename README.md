# MetaboNet Benchmark

Code for reproducing MetaboNet paper benchmarks and training the local LSTM,
UniTS, and GluForecast baselines.

## Setup

```bash
uv venv .env
source .env/bin/activate
uv pip install -r requirements.txt
```

Download the public MetaboNet splits from [metabo-net.org](https://metabo-net.org/).

## Benchmark Workflow

1. Download the MetaboNet test split and place it at
   `data/metabonet_public_test.parquet`.

2. Preprocess the test split:

```bash
python data/preprocess.py --path_to_dataset data/metabonet_public_test.parquet
```

This writes `data/metabonet_test.parquet`.

3. Run the benchmark for one or more models:

```bash
python benchmark.py --model lstm,units,gluforecast --batch_size 16 --device cuda
```

The local model names `lstm`, `units`, and `gluforecast` use the fully featured
`*-cgm-insulin-carbs` checkpoints in `studies/feature_ablation/checkpoints/`.

4. Combine model outputs and calculate summary metrics:

```bash
python results_eval.py --results_dir results --output_path results/combined_results.parquet
```

5. Optionally calculate bootstrapped confidence intervals:

```bash
python calculate_metrics_with_ci.py \
  --input_path results/combined_results.parquet \
  --output_path results/metrics_with_ci.csv \
  --device cuda
```

Use `--skip_dts` to omit DTS error-grid confidence intervals.

## Training Workflow

1. Download the MetaboNet train split and place it at
   `data/metabonet_public_train.parquet`.

2. Preprocess the train split:

```bash
python data/preprocess.py --path_to_dataset data/metabonet_public_train.parquet
```

This writes `data/metabonet_train.parquet`.

3. Run Optuna hyperparameter search:

```bash
python studies/feature_ablation/optuna_search.py \
  --data_path data/metabonet_train.parquet \
  --device cuda
```

4. Train final models from the selected hyperparameters:

```bash
python studies/feature_ablation/train.py \
  --data_path data/metabonet_train.parquet \
  --optuna_dir studies/feature_ablation/optuna \
  --device cuda
```

Checkpoints are written to `studies/feature_ablation/checkpoints/` and can be
benchmarked with `benchmark.py`.
