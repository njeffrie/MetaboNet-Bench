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
        
    Note:
        Input arrays have shape (n_samples, 3, n_horizons) where axis 1 contains:
        [0] user_ids, [1] predictions, [2] true_labels
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
            
            # Extract predictions and labels from new structure
            # arr shape: (n_samples, 3, n_horizons) where axis 1 = [user_ids, predictions, labels]
            predictions = arr[:, 1, :]  # shape: (n_samples, n_horizons)
            labels = arr[:, 2, :]       # shape: (n_samples, n_horizons)
            
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
        Tuple of (predictions, labels) as numpy arrays of shape (n_samples, n_horizons)
        
    Note:
        Input arrays have shape (n_samples, 3, n_horizons) where axis 1 contains:
        [0] user_ids, [1] predictions, [2] true_labels
    """
    results_path = Path(results_dir) / model_name / f"{dataset_name}.npy"
    
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    arr = np.load(results_path)
    # Extract from new structure: axis 1 = [user_ids, predictions, labels]
    predictions = arr[:, 1, :]
    labels = arr[:, 2, :]
    
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
        results: Results dict from load_model_results (predictions, labels tuples)
        model_name: Name of the model
    
    Returns:
        Combined (predictions, labels) arrays of shape (total_samples, n_horizons)
        
    Note:
        Input data structure: each dataset contains (n_samples, 3, n_horizons) arrays
        where axis 1 = [user_ids, predictions, labels]
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


def load_model_results_with_user_ids(results_dir: str = "results", model_name: str = None, 
                                    dataset_names: Optional[List[str]] = None) -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """
    Load results for one or all models including user IDs.
    
    Args:
        results_dir: Path to results directory
        model_name: Specific model to load (if None, loads all models)
        dataset_names: List of specific datasets to load (if None, loads all datasets)
    
    Returns:
        Dict with structure: {model_name: {dataset_name: (predictions, labels, user_ids)}}
        where all arrays have shape (n_samples, n_horizons)
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
            dataset_name = dataset_file.stem
            
            # Filter by dataset names if specified
            if dataset_names and dataset_name not in dataset_names:
                continue
                
            # Load the numpy file
            arr = np.load(dataset_file)
            
            # Extract user_ids, predictions and labels
            user_ids = arr[:, 0, :]     # shape: (n_samples, n_horizons)
            predictions = arr[:, 1, :]  # shape: (n_samples, n_horizons)
            labels = arr[:, 2, :]       # shape: (n_samples, n_horizons)
            
            model_results[dataset_name] = (predictions, labels, user_ids)
        
        if model_results:
            results[model_dir.name] = model_results
    
    return results


def load_single_dataset_with_user_ids(results_dir: str = "results", model_name: str = None, 
                                     dataset_name: str = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load results for a single model-dataset combination including user IDs.
    
    Args:
        results_dir: Path to results directory
        model_name: Name of the model
        dataset_name: Name of the dataset
    
    Returns:
        Tuple of (predictions, labels, user_ids) as numpy arrays of shape (n_samples, n_horizons)
    """
    results_path = Path(results_dir) / model_name / f"{dataset_name}.npy"
    
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    arr = np.load(results_path)
    user_ids = arr[:, 0, :]
    predictions = arr[:, 1, :]
    labels = arr[:, 2, :]
    
    return predictions, labels, user_ids


def get_user_ids(results_dir: str = "results", model_name: str = None, 
                dataset_name: str = None) -> np.ndarray:
    """
    Extract just the user IDs for a specific model-dataset combination.
    
    Args:
        results_dir: Path to results directory
        model_name: Name of the model
        dataset_name: Name of the dataset
    
    Returns:
        User IDs array of shape (n_samples, n_horizons)
    """
    results_path = Path(results_dir) / model_name / f"{dataset_name}.npy"
    
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    arr = np.load(results_path)
    return arr[:, 0, :]


def get_unique_user_ids(results_dir: str = "results", model_name: str = None, 
                       dataset_name: str = None) -> np.ndarray:
    """
    Get unique user IDs from a dataset.
    
    Args:
        results_dir: Path to results directory
        model_name: Name of the model
        dataset_name: Name of the dataset
    
    Returns:
        Array of unique user IDs
    """
    user_ids = get_user_ids(results_dir, model_name, dataset_name)
    # User IDs are typically constant across time horizons, so take first column
    return np.unique(user_ids[:, 0])


def filter_by_user_ids(predictions: np.ndarray, labels: np.ndarray, user_ids: np.ndarray, 
                      target_user_ids: Union[int, List[int], np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter results by specific user IDs.
    
    Args:
        predictions: Predictions array of shape (n_samples, n_horizons)
        labels: Labels array of shape (n_samples, n_horizons)
        user_ids: User IDs array of shape (n_samples, n_horizons)
        target_user_ids: Single user ID or list of user IDs to keep
    
    Returns:
        Filtered (predictions, labels, user_ids) arrays
    """
    if isinstance(target_user_ids, (int, float)):
        target_user_ids = [target_user_ids]
    
    # User IDs are typically constant across time horizons, so use first column for filtering
    sample_user_ids = user_ids[:, 0]
    mask = np.isin(sample_user_ids, target_user_ids)
    
    return predictions[mask], labels[mask], user_ids[mask]