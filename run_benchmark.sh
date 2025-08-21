#!/bin/bash

# Default values
models="gluformer"
datasets="Anderson2016,Brown2019,Lynch2022"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model=*)
            models="${1#*=}"
            shift
            ;;
        --datasets=*)
            datasets="${1#*=}"
            shift
            ;;
        --help)
            echo "Usage: $0 [--model=<model1>,<model2>,...] [--datasets=<dataset1>,<dataset2>,...]"
            echo "  --model: Comma-separated list of models to benchmark (default: gluformer)"
            echo "  --datasets: Comma-separated list of datasets to benchmark (default: Brown2019)"
            echo "  --help: Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Convert comma-separated strings to arrays
IFS=',' read -ra MODEL_ARRAY <<< "$models"
IFS=',' read -ra DATASET_ARRAY <<< "$datasets"

echo "Running benchmarks for:"
echo "  Models: ${MODEL_ARRAY[*]}"
echo "  Datasets: ${DATASET_ARRAY[*]}"
echo ""

mkdir -p plots

# Run benchmark for every permutation of model and dataset
for model in "${MODEL_ARRAY[@]}"; do
    for dataset in "${DATASET_ARRAY[@]}"; do
        echo "=========================================="
        echo "Running benchmark: Model=$model, Dataset=$dataset"
        echo "=========================================="
        
        python benchmark.py --dataset "$dataset" --model "$model" --plot --save_plot "plots/${model}-${dataset}.png"
        
        if [ $? -ne 0 ]; then
            echo "✗ Benchmark failed for $model on $dataset"
        fi
        echo ""
    done
done

echo "All benchmarks completed!"