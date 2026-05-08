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

## Benchmark workflow

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

Tabular Ridge and LightGBM baselines load from the Hugging Face Hub ([Ridge](https://huggingface.co/anonymous-4FAD/Ridge), [LightGBM](https://huggingface.co/anonymous-4FAD/LightGBM)); use e.g. `--model ridge,ridge-cgm,lightgbm-insulin` (`ridge` / `lightgbm` alone selects the `all` ablation).

The local model names `lstm`, `units`, and `gluforecast` use the fully featured
`*-cgm-insulin-carbs` checkpoints in `checkpoints/`.

On Apple Silicon, `--device mps` still applies to LSTM and GluForecast. UniTS (`units`)
redirects to CPU automatically: PyTorch MPS `scaled_dot_product_attention` misreports output shape when the value dim differs from query/key ([issue](https://github.com/pytorch/pytorch/issues/176767), [fix PR](https://github.com/pytorch/pytorch/pull/176843)).

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

## Training workflow

See [train/README.md](train/README.md) for preprocessing the train split, Optuna search, and training commands. Checkpoints are produced under `checkpoints/` for use with `benchmark.py`.

## Benchmark custom models

Create a model runner in `models/<model_name>.py` and register it in
[`models/models.py`](models/models.py), then run `python benchmark.py --model=<model_name>`.

See `models/gluformer.py` for an example. For Gluformer-style models you can share weights on the Hugging Face Hub (e.g. the pretrained [Gluformer model](https://huggingface.co/anonymous-4FAD/Gluformer)); local LSTM, UniTS, and GluForecast baselines use checkpoints from `checkpoints/` as described above.

### Ridge and LightGBM (tabular baselines)

The Ridge and LightGBM baselines trained under [other_models/results/](other_models/results) are packaged as Hub models. Each repo holds all four feature ablations (`cgm`, `insulin`, `carbs`, `all`) and the active one is selected via the `ablation=` kwarg:

```bash
python benchmark.py --model ridge,ridge-cgm,lightgbm-insulin --batch_size 16 --device cpu
```

Bare `ridge` / `lightgbm` defaults to the `all` ablation. To stage the Hub repos locally before pushing:

```bash
python scripts/build_other_models_hub.py        # writes hub/ridge and hub/lightgbm
bash hub/ridge/push.sh                          # uploads to anonymous-4FAD/Ridge
bash hub/lightgbm/push.sh                       # uploads to anonymous-4FAD/LightGBM
```

See [hub/README.md](hub/README.md) for details.

## Generate figures

Producing the paper's figures takes three steps: (1) run the benchmark for
each model you want to compare, (2) merge the per-model result files into a
single parquet enriched with subject demographics from MetaboNet, and (3) run
the figure scripts against the merged parquet.

### 1. Generate per-model results

Each `python benchmark.py --model <name>` invocation writes a long-format
parquet to `results/<name>_results.parquet`. Run it once per model:

```bash
python benchmark.py --model gluforecast --batch_size 16 --device mps
python benchmark.py --model lstm        --batch_size 16 --device mps
python benchmark.py --model gluformer   --batch_size 16 --device mps
# ...add as many as you want to plot
```

After this you should have a `results/` directory containing one
`*_results.parquet` per model.

### 2. Merge results with demographics

`figures/combine_results.py combine` reads every `*_results.parquet` in a
directory, pivots them into wide format (`label_t0..t11`, `pred_t0..t11`),
and merges in age/gender/CGM stats/insulin/carbs from the MetaboNet public
test split.

`--results-dir` accepts either a directory (per-model `*_results.parquet` or
legacy npy subdirs) **or a single multi-model parquet**. Single-file layouts
auto-detected: long-with-`model`-column, long-by-horizon + wide-by-model
(one prediction column per model name), or wide-by-horizon + wide-by-model
(`<model>_pred_t0..t11` plus shared `label_t0..t11`). The single-file path
reads one model's columns at a time via pyarrow, so the full table is never
held in memory.

You need a copy of `metabonet_public_test.parquet`. Point at it either via
the `METABONET_TEST_PARQUET` environment variable (a local path or `s3://`
URI works — set this in `.env` if you prefer) or via `--metabonet-test`:

```bash
# Option A: env var (set once per shell, or in .env)
export METABONET_TEST_PARQUET=/path/to/metabonet_public_test.parquet

# Option B: pass it explicitly, against a directory of per-model files
python figures/combine_results.py combine \
    --results-dir results \
    --output combined_results.parquet \
    --metabonet-test /path/to/metabonet_public_test.parquet

# Option C: a single multi-model parquet (use --input-file or --results-dir)
python figures/combine_results.py combine \
    --input-file all_models.parquet \
    --output combined_results.parquet
```

Use `--no-demographics` to skip the MetaboNet merge if you only need the
combined predictions/labels.

To append a newly-trained model into an existing combined file without
re-running everything:

```bash
python figures/combine_results.py add \
    --combined combined_results.parquet \
    --add-files results/my_new_model_results.parquet \
    --replace          # optional: replace prior rows for the same model
```

### 3. Render the figures

`figures/run_all.sh` drives every plot script against the combined parquet
and writes PNGs into a single output directory:

```bash
bash figures/run_all.sh combined_results.parquet figures/out
```

Both arguments are optional:

- Arg 1 (`INPUT_PARQUET`) defaults to `combined_results_with_aux.parquet`.
- Arg 2 (`OUT_DIR`) defaults to `figures/out` (created if missing).

To render a single chart instead of the whole sweep, run the corresponding
script directly. See `figures/README.md` for the full list. Most figure scripts share a common filter surface
(`--exclude-models`, `--all-variants`, `--ablation {all,only,exclude}`,
`--figsize W H`).
