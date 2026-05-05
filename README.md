# MetaboNet Benchmark

This benchmark provides a framework to fairly evaulate blood glucose prediction models for Type 1 Diabetes.

## Installation

1. Install the required dependencies:

```bash
uv venv .env
source .env/bin/activate
uv pip install -r requirements.txt
```

Download [metabonet](https://metabo-net.org/) to `data/downloads`


Run the extraction and preprocessing scripts (may take a few minutes):
```bash
cd data
python preprocess.py
cd ..
```

## Run the benchmark

For example:
```bash
python benchmark.py --model gluforecast --batch_size 16 --device mps
```

## Benchmark Custom Models

Create a model runner in `models/<model_name>.py` and add your model to the model name to runner dictionary in `models/model.py` then run the benchmark with `python benchmark.py --model=<model_name>`

See models/gluformer for an example. It is strongly encouraged to share the model with weights on huggingface hub. See the pretrained [Gluformer model](https://huggingface.co/njeffrie/Gluformer) for an example.

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

You need a copy of `metabonet_public_test.parquet`. Point at it either via
the `METABONET_TEST_PARQUET` environment variable (a local path or `s3://`
URI works — set this in `.env` if you prefer) or via `--metabonet-test`:

```bash
# Option A: env var (set once per shell, or in .env)
export METABONET_TEST_PARQUET=/path/to/metabonet_public_test.parquet

# Option B: pass it explicitly
python figures/combine_results.py combine \
    --results-dir results \
    --output combined_results.parquet \
    --metabonet-test /path/to/metabonet_public_test.parquet
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
script directly. A few common ones (see `figures/README.md` for the full
list):

```bash
# RMSE vs prediction horizon, one line per model
python figures/plot_model_rmse.py --input combined_results.parquet \
    --output figures/out/model_rmse.png --ablation exclude

# RMSE vs reference CGM, all models on one axes, at the 30-min horizon
python figures/plot_rmse_by_bg.py --input combined_results.parquet \
    --combined --horizon 30 --output figures/out/rmse_vs_bg_h30.png \
    --ablation exclude

# Bar chart stratified by a demographic (e.g. gender, all horizons averaged)
python figures/plot_rmse_by_demographic.py --input combined_results.parquet \
    --group-by gender --horizon all \
    --output figures/out/rmse_by_gender.png --ablation exclude
```

Most figure scripts share a common filter surface
(`--exclude-models`, `--all-variants`, `--ablation {all,only,exclude}`,
`--figsize W H`); see `figures/README.md` for details.
