# CGM Dataset Processor

This module provides a `CGMDataSet` class that converts CGM (Continuous Glucose Monitoring) data from a pipe-separated text file into a HuggingFace Dataset format.

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt

Download the [Brown 2019 dataset](https://public.jaeb.org/dataset/573) and extract to `./brown_dataset`
```

## Usage

Run the example script to see the class in action:

```bash
python brown_2019_example.py
```

## Dataset Statistics

Brown 2019:
- **Total records**: 9,032,236
- **File size**: 401MB
- **Columns**: 4 (PtID, Period, DataDtTm, CGM)
