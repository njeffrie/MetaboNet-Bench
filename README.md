# Glucose Predction Benchmark

This benchmark provides methods to compare glucose prediction models across a set of high quality datasets.

## Installation

1. Install the required dependencies:

```bash
uv venv .env
source .env/bin/activate
pip install -r requirements.txt
```

Download the following dataset zipfiles and save to the `data` directory:
- [Anderson 2016](https://public.jaeb.org/jdrfapp2/stdy/download/465)
- [Brown 2019](https://public.jaeb.org/dataset/573)
- [Lynch 2022](https://public.jaeb.org/dataset/579)

Run the extraction and preprocessing scripts (may take a few minutes):
```bash
pushd && cd data
python extract.py
python preprocess.py
popd
```

## Benchmark the pre-trained Gluformer model (default model and datasets)
```bash
./run_benchmark.sh
```

## Benchmark Custom Model

Create a model runner in `models/<model_name>.py` and add your model to the model name to runner dictionary in `models/model.py` then run the benchmark with `./run_benchmark.sh --model=<model_name>`

See models/gluformer for an example. It is strongly encouraged to share the model with weights on huggingface hub. See the pretrained [Gluformer model](https://huggingface.co/njeffrie/Gluformer) for an example.