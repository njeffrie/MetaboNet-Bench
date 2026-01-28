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
