import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Tuple, List, Union
from .results_loader import load_model_results, combine_datasets
from .metrics import calculate_rmse


def plot_rmse_by_horizon(results_dir: str = "results", dataset_filter: str = None, 
                         cgm_filter: str = None, save_path: str = None, show: bool = True):
    """
    Plot RMSE by forecast horizon for all models.
    
    Args:
        results_dir: Path to results directory
        dataset_filter: Filter to specific dataset (if None, uses all datasets combined)
        cgm_filter: Filter by CGM range ("<70", "<50", "70-140", "70-180", ">180", ">250")
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
    """
    # Load model results with optional dataset filter
    dataset_names = [dataset_filter] if dataset_filter else None
    results = load_model_results(results_dir, dataset_names=dataset_names)
    
    # Define CGM filter functions
    cgm_filters = {
        "<70": lambda x: x < 70,
        "<50": lambda x: x < 50,
        "70-140": lambda x: (x >= 70) & (x < 140),
        "70-180": lambda x: (x >= 70) & (x < 180),
        ">180": lambda x: x > 180,
        ">250": lambda x: x > 250
    }
    
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
            
            # Apply CGM filter if specified
            if cgm_filter and cgm_filter in cgm_filters:
                mask = cgm_filters[cgm_filter](labels_h)
                if np.any(mask):
                    preds_h = preds_h[mask]
                    labels_h = labels_h[mask]
                else:
                    rmse_values.append(np.nan)  # No data in this range
                    continue
            
            rmse = calculate_rmse(preds_h, labels_h)
            rmse_values.append(rmse)
        
        # Plot line for this model
        plt.plot(horizons, rmse_values, marker='o', linewidth=2, markersize=6, label=model_name)
    
    plt.xlabel('Forecast Horizon (minutes)', fontsize=12)
    plt.ylabel('RMSE (mg/dL)', fontsize=12)
    
    # Update title based on filters
    title_parts = ['RMSE by Forecast Horizon for All Models']
    if dataset_filter:
        title_parts.append(f'on {dataset_filter}')
    if cgm_filter:
        title_parts.append(f'(CGM {cgm_filter} mg/dL)')
    
    plt.title(' '.join(title_parts), fontsize=14, fontweight='bold')
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    if show:
        plt.show()


def plot_rmse_by_cgm_interval(results_dir: str = "results", dataset_filter: str = None, 
                              horizons: Union[List[int], str] = "all", save_path: str = None, show: bool = True):
    """
    Plot RMSE by CGM intervals for all models.
    
    Args:
        results_dir: Path to results directory
        dataset_filter: Filter to specific dataset (if None, uses all datasets combined)
        horizons: Forecast horizons to include (list of indices 0-11, or "all")
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
    """
    # Load model results with optional dataset filter
    dataset_names = [dataset_filter] if dataset_filter else None
    results = load_model_results(results_dir, dataset_names=dataset_names)
    
    # Define CGM intervals
    intervals = [
        ("<50", lambda x: x < 50),
        ("50-70", lambda x: (x >= 50) & (x < 70)),
        ("70-140", lambda x: (x >= 70) & (x < 140)),
        ("140-180", lambda x: (x >= 140) & (x < 180)),
        ("180-250", lambda x: (x >= 180) & (x < 250)),
        (">250", lambda x: x >= 250)
    ]
    
    # Handle horizons parameter
    if horizons == "all":
        horizon_indices = list(range(12))
        horizon_label = "all horizons"
    else:
        horizon_indices = horizons
        if len(horizon_indices) == 1:
            horizon_label = f"horizon {(horizon_indices[0] + 1) * 5} min"
        else:
            horizon_times = [(h + 1) * 5 for h in horizon_indices]
            horizon_label = f"horizons {horizon_times}"
    
    plt.figure(figsize=(12, 8))
    
    # Calculate RMSE for each model and interval
    interval_names = [interval[0] for interval in intervals]
    
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
        
        rmse_by_interval = []
        
        for interval_name, interval_func in intervals:
            # Collect predictions and labels for this interval across specified horizons
            interval_preds = []
            interval_labels = []
            
            for h_idx in horizon_indices:
                preds_h = combined_preds[:, h_idx]
                labels_h = combined_labels[:, h_idx]
                
                # Filter by CGM interval based on true values
                mask = interval_func(labels_h)
                if np.any(mask):
                    interval_preds.extend(preds_h[mask])
                    interval_labels.extend(labels_h[mask])
            
            # Calculate RMSE for this interval
            if len(interval_preds) > 0:
                rmse = calculate_rmse(np.array(interval_preds), np.array(interval_labels))
                rmse_by_interval.append(rmse)
            else:
                rmse_by_interval.append(np.nan)  # No data in this interval
        
        # Plot bar for this model
        x_pos = np.arange(len(interval_names))
        plt.plot(x_pos, rmse_by_interval, marker='o', linewidth=2, markersize=6, label=model_name)
    
    plt.xlabel('CGM Interval (mg/dL)', fontsize=12)
    plt.ylabel('RMSE (mg/dL)', fontsize=12)
    
    # Update title based on dataset filter and horizons
    title_parts = ['RMSE by CGM Interval for All Models']
    if dataset_filter:
        title_parts.append(f'on {dataset_filter}')
    if horizons != "all":
        title_parts.append(f'({horizon_label})')
    
    plt.title(' '.join(title_parts), fontsize=14, fontweight='bold')
    plt.xticks(range(len(interval_names)), interval_names, rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    if show:
        plt.show()