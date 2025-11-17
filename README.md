# Glucose Prediction Benchmark

This benchmark provides methods to compare glucose prediction models across a set of high quality datasets.

## Installation

1. Install the required dependencies:

```bash
uv venv .env
source .env/bin/activate
uv pip install -r requirements.txt
```

Download the following dataset zipfiles and save to `data/downloads`:
- [Anderson 2016](https://public.jaeb.org/jdrfapp2/stdy/download/465)
- [Brown 2019](https://public.jaeb.org/dataset/573)
- [Manchester 2024](https://github.com/sharpic/ManchesterCSCoordinatedDiabetesStudy/archive/refs/tags/V1.0.4.zip)
- [AZT1D](https://data.mendeley.com/public-files/datasets/gk9m674wcx/files/b02a20be-27c4-4dd0-8bb5-9171c66262fb/file_downloaded)

Run the extraction and preprocessing scripts (may take a few minutes):
```bash
cd data
python preprocess.py
cd ..
```

## Run the benchmark

For example:
```bash
./run_benchmark.sh --model=gluformer
```

## Benchmark Custom Models

Create a model runner in `models/<model_name>.py` and add your model to the model name to runner dictionary in `models/model.py` then run the benchmark with `./run_benchmark.sh --model=<model_name>`

See models/gluformer for an example. It is strongly encouraged to share the model with weights on huggingface hub. See the pretrained [Gluformer model](https://huggingface.co/njeffrie/Gluformer) for an example.

## Add a Dataset

Create a dataset preprocessor for your dataset and add the preprocessor to `data/preprocess.py`. See [this change](https://github.com/njeffrie/bg_prediction_benchmark/commit/62064552d92b880c12b70c4ff7576e9f17378aca) for an example.
