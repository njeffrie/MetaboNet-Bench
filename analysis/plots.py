import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Tuple
from .results_loader import load_model_results, combine_datasets
from .metrics import calculate_rmse


def plot_rmse_by_horizon(results_dir: str = "results", dataset_filter: str = None, save_path: str = None, show: bool = True):
    """
    Plot RMSE by forecast horizon for all models.
    
    Args:
        results_dir: Path to results directory
        dataset_filter: Filter to specific dataset (if None, uses all datasets combined)
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
    """
    # Load model results with optional dataset filter
    dataset_names = [dataset_filter] if dataset_filter else None
    results = load_model_results(results_dir, dataset_names=dataset_names)
    
    # Forecast horizons (5, 10, 15, ..., 60 minutes)
    horizons = np.arange(1, 13) * 5
    
    plt.figure(figsize=(12, 8))
    
    for model_name in sorted(results.keys()):
        if dataset_filter:
            # Use specific dataset
            if dataset_filter in results[model_name]:
                combined_preds, combined_labels = results[model_name][dataset_filter]
            else:
                continue  # Skip this model if it doesn't have the dataset
        else:
            # Combine all datasets for this model
            combined_preds, combined_labels = combine_datasets(results, model_name)
        
        # Calculate RMSE for each horizon
        rmse_values = []
        for h_idx in range(12):  # 12 forecast horizons
            preds_h = combined_preds[:, h_idx]
            labels_h = combined_labels[:, h_idx]
            rmse = calculate_rmse(preds_h, labels_h)
            rmse_values.append(rmse)
        
        # Plot line for this model
        plt.plot(horizons, rmse_values, marker='o', linewidth=2, markersize=6, label=model_name)
    
    plt.xlabel('Forecast Horizon (minutes)', fontsize=12)
    plt.ylabel('RMSE (mg/dL)', fontsize=12)
    
    # Update title based on dataset filter
    if dataset_filter:
        title = f'RMSE by Forecast Horizon for All Models on {dataset_filter}'
    else:
        title = 'RMSE by Forecast Horizon for All Models (All Datasets)'
    plt.title(title, fontsize=14, fontweight='bold')
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    if show:
        plt.show()