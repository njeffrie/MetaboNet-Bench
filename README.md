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
`*-cgm-insulin-carbs` checkpoints in `train/checkpoints/`.

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

See [train/README.md](train/README.md) for preprocessing the train split, Optuna search, and training commands. Checkpoints are produced under `train/checkpoints/` for use with `benchmark.py`.
