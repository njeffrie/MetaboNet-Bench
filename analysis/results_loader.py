import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union


def load_model_results(results_dir: str = "results", model_name: str = None, 
                      dataset_names: Optional[List[str]] = None) -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """
    Load results for one or all models from the results directory.
    
    Args:
        results_dir: Path to results directory
        model_name: Specific model to load (if None, loads all models)
        dataset_names: List of specific datasets to load (if None, loads all datasets)
    
    Returns:
        Dict with structure: {model_name: {dataset_name: (predictions, labels)}}
        where predictions and labels are numpy arrays of shape (n_samples, n_horizons)
    """
    results_path = Path(results_dir)
    results = {}
    
    # Get model directories
    if model_name:
        model_dirs = [results_path / model_name] if (results_path / model_name).exists() else []
    else:
        model_dirs = [d for d in results_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    for model_dir in model_dirs:
        model_results = {}
        
        # Get dataset files
        dataset_files = [f for f in model_dir.glob("*.npy")]
        
        for dataset_file in dataset_files:
            dataset_name = dataset_file.stem  # filename without extension
            
            # Filter by dataset names if specified
            if dataset_names and dataset_name not in dataset_names:
                continue
                
            # Load the numpy file
            arr = np.load(dataset_file)
            
            # Extract predictions and labels
            predictions = arr[:, 0, :]  # shape: (n_samples, n_horizons)
            labels = arr[:, 1, :]       # shape: (n_samples, n_horizons)
            
            model_results[dataset_name] = (predictions, labels)
        
        if model_results:  # Only add if we found datasets
            results[model_dir.name] = model_results
    
    return results


def load_single_dataset(results_dir: str = "results", model_name: str = None, 
                       dataset_name: str = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load results for a single model-dataset combination.
    
    Args:
        results_dir: Path to results directory
        model_name: Name of the model
        dataset_name: Name of the dataset
    
    Returns:
        Tuple of (predictions, labels) as numpy arrays
    """
    results_path = Path(results_dir) / model_name / f"{dataset_name}.npy"
    
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    arr = np.load(results_path)
    predictions = arr[:, 0, :]
    labels = arr[:, 1, :]
    
    return predictions, labels


def get_available_models(results_dir: str = "results") -> List[str]:
    """Get list of available model names."""
    results_path = Path(results_dir)
    return [d.name for d in results_path.iterdir() if d.is_dir() and not d.name.startswith('.')]


def get_available_datasets(results_dir: str = "results", model_name: str = None) -> Dict[str, List[str]]:
    """
    Get available datasets for models.
    
    Args:
        results_dir: Path to results directory
        model_name: Specific model (if None, returns for all models)
    
    Returns:
        Dict with structure: {model_name: [dataset_names]}
    """
    results_path = Path(results_dir)
    datasets = {}
    
    if model_name:
        model_dirs = [results_path / model_name] if (results_path / model_name).exists() else []
    else:
        model_dirs = [d for d in results_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    for model_dir in model_dirs:
        dataset_files = [f.stem for f in model_dir.glob("*.npy")]
        datasets[model_dir.name] = sorted(dataset_files)
    
    return datasets


def combine_datasets(results: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]], 
                    model_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Combine all datasets for a given model into single arrays.
    
    Args:
        results: Results dict from load_model_results
        model_name: Name of the model
    
    Returns:
        Combined (predictions, labels) arrays
    """
    if model_name not in results:
        raise ValueError(f"Model {model_name} not found in results")
    
    all_predictions = []
    all_labels = []
    
    for dataset_name, (preds, labels) in results[model_name].items():
        all_predictions.append(preds)
        all_labels.append(labels)
    
    combined_predictions = np.concatenate(all_predictions, axis=0)
    combined_labels = np.concatenate(all_labels, axis=0)
    
    return combined_predictions, combined_labels