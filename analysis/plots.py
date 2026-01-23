import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Tuple, List, Union, Optional
from .results_loader import load_model_results, combine_datasets, load_demographics, load_all_demographics
from .metrics import calculate_rmse


def plot_rmse_by_horizon(results_dir: str = "results", dataset_filter: str = None, 
                         cgm_filter: str = None, save_path: str = None, show: bool = True,
                         ylim: Optional[Tuple[float, float]] = (0, 80), show_oob_indicators: bool = True):
    """
    Plot RMSE by forecast horizon for all models.
    
    Args:
        results_dir: Path to results directory
        dataset_filter: Filter to specific dataset (if None, uses all datasets combined)
        cgm_filter: Filter by CGM range ("<70", "<50", "70-140", "70-180", ">180", ">250")
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
        ylim: Y-axis limits as (min, max) tuple. Set to None for auto-scaling. Default: (0, 80)
        show_oob_indicators: Show arrows and values for out-of-bounds points. Default: True
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
    
    # Set Times New Roman font
    plt.rcParams['font.family'] = 'Times New Roman'
    
    plt.figure(figsize=(6, 4))
    
    # Store out-of-bounds information
    oob_data = []  # List of (x, y, value, color, model_name) for out-of-bounds points
    
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
        line = plt.plot(horizons, rmse_values, marker='o', linewidth=2, markersize=6, label=model_name)
        line_color = line[0].get_color()
        
        # Detect out-of-bounds values for indicators
        if show_oob_indicators and ylim is not None:
            upper_bound = ylim[1]
            for i, (horizon, rmse_val) in enumerate(zip(horizons, rmse_values)):
                if not np.isnan(rmse_val) and rmse_val > upper_bound:
                    oob_data.append((horizon, upper_bound, rmse_val, line_color, model_name))
    
    plt.xlabel('Forecast Horizon (minutes)', fontsize=12)
    plt.ylabel('RMSE (mg/dL)', fontsize=12)
    
    # Update title based on filters
    title_parts = ['RMSE by Forecast Horizon for All Models']
    if dataset_filter:
        title_parts.append(f'on {dataset_filter}')
    if cgm_filter:
        title_parts.append(f'(CGM {cgm_filter} mg/dL)')
    
    plt.title(' '.join(title_parts), fontsize=14, fontweight='bold')
    
    # Add out-of-bounds indicators before setting ylim
    if oob_data:
        for x, y_pos, actual_value, color, model in oob_data:
            # Draw upward arrow at the upper bound
            plt.annotate('', xy=(x, y_pos), xytext=(x, y_pos - 2),
                        arrowprops={'arrowstyle': '->', 'facecolor': color, 'edgecolor': color, 'lw': 1.5})
            
            # Add value annotation above the arrow
            plt.annotate(f'{actual_value:.1f}', xy=(x, y_pos), xytext=(x, y_pos + 1),
                        ha='center', va='bottom', fontsize=9, color=color, fontweight='bold')
    
    # Set y-axis limits if specified
    if ylim is not None:
        plt.ylim(ylim)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    if show:
        plt.show()


def plot_rmse_by_cgm_interval(results_dir: str = "results", dataset_filter: str = None, 
                              horizons: Union[List[int], str] = "all", save_path: str = None, show: bool = True,
                              ylim: Optional[Tuple[float, float]] = (0, 80), show_oob_indicators: bool = True):
    """
    Plot RMSE by CGM intervals for all models.
    
    Args:
        results_dir: Path to results directory
        dataset_filter: Filter to specific dataset (if None, uses all datasets combined)
        horizons: Forecast horizons to include (list of indices 0-11, or "all")
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
        ylim: Y-axis limits as (min, max) tuple. Set to None for auto-scaling. Default: (0, 90)
        show_oob_indicators: Show arrows and values for out-of-bounds points. Default: True
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
    
    # Set Times New Roman font
    plt.rcParams['font.family'] = 'Times New Roman'
    
    plt.figure(figsize=(6, 4))
    
    # Calculate RMSE for each model and interval
    interval_names = [interval[0] for interval in intervals]
    
    # Store out-of-bounds information
    oob_data = []  # List of (x, y, value, color, model_name) for out-of-bounds points
    
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
        
        # Plot line for this model
        x_pos = np.arange(len(interval_names))
        line = plt.plot(x_pos, rmse_by_interval, marker='o', linewidth=2, markersize=6, label=model_name)
        line_color = line[0].get_color()
        
        # Detect out-of-bounds values for indicators
        if show_oob_indicators and ylim is not None:
            upper_bound = ylim[1]
            for i, (x, rmse_val) in enumerate(zip(x_pos, rmse_by_interval)):
                if not np.isnan(rmse_val) and rmse_val > upper_bound:
                    oob_data.append((x, upper_bound, rmse_val, line_color, model_name))
    
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
    
    # Add out-of-bounds indicators before setting ylim
    if oob_data:
        for x, y_pos, actual_value, color, model in oob_data:
            # Draw upward arrow at the upper bound
            plt.annotate('', xy=(x, y_pos), xytext=(x, y_pos - 2),
                        arrowprops={'arrowstyle': '->', 'facecolor': color, 'edgecolor': color, 'lw': 1.5})
            
            # Add value annotation above the arrow
            plt.annotate(f'{actual_value:.1f}', xy=(x, y_pos), xytext=(x, y_pos + 1),
                        ha='center', va='bottom', fontsize=9, color=color, fontweight='bold')
    
    # Set y-axis limits if specified
    if ylim is not None:
        plt.ylim(ylim)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    if show:
        plt.show()


def plot_count_by_cgm_interval(results_dir: str = "results", dataset_filter: str = None, 
                               horizons: Union[List[int], str] = "all", save_path: str = None, show: bool = True,
                               ylim: Optional[Tuple[float, float]] = None, show_oob_indicators: bool = True):
    """
    Plot count of true label values by CGM intervals (uses first available model).
    
    Args:
        results_dir: Path to results directory
        dataset_filter: Filter to specific dataset (if None, uses all datasets combined)
        horizons: Forecast horizons to include (list of indices 0-11, or "all")
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
        ylim: Y-axis limits as (min, max) tuple. Set to None for auto-scaling.
        show_oob_indicators: Show arrows and values for out-of-bounds points. Default: True
    """
    # Load model results with optional dataset filter
    dataset_names = [dataset_filter] if dataset_filter else None
    results = load_model_results(results_dir, dataset_names=dataset_names)
    
    if not results:
        raise ValueError("No results found")
    
    # Use the first available model (labels are consistent across models)
    first_model_name = sorted(results.keys())[0]
    
    if dataset_filter:
        # Use specific dataset
        if dataset_filter in results[first_model_name]:
            combined_preds, combined_labels = results[first_model_name][dataset_filter]
        else:
            raise ValueError(f"Dataset {dataset_filter} not found")
    else:
        # Combine all datasets for this model
        combined_preds, combined_labels = combine_datasets(results, first_model_name)
    
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
    
    # Set Times New Roman font
    plt.rcParams['font.family'] = 'Times New Roman'
    
    plt.figure(figsize=(4, 4))
    
    # Calculate count for each interval (using only labels)
    interval_names = [interval[0] for interval in intervals]
    count_by_interval = []
    
    for interval_name, interval_func in intervals:
        # Count only true labels for this interval across specified horizons
        total_count = 0
        
        for h_idx in horizon_indices:
            labels_h = combined_labels[:, h_idx]
            
            # Count values in this CGM interval based on true values only
            mask = interval_func(labels_h)
            total_count += np.sum(mask)
        
        count_by_interval.append(total_count)
    
    # Plot bar chart
    x_pos = np.arange(len(interval_names))
    bars = plt.bar(x_pos, count_by_interval, alpha=0.7, edgecolor='black', linewidth=1)
    
    # Store out-of-bounds information
    oob_data = []
    if show_oob_indicators and ylim is not None:
        upper_bound = ylim[1]
        for i, (x, count_val) in enumerate(zip(x_pos, count_by_interval)):
            if count_val > upper_bound:
                oob_data.append((x, upper_bound, count_val, bars[i].get_facecolor(), 'counts'))
    
    plt.xlabel('CGM Interval (mg/dL)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    
    # Update title based on dataset filter and horizons
    title_parts = ['True Values Count by CGM Interval']
    if dataset_filter:
        title_parts.append(f'on {dataset_filter}')
    if horizons != "all":
        title_parts.append(f'({horizon_label})')
    
    plt.title(' '.join(title_parts), fontsize=14, fontweight='bold')
    plt.xticks(range(len(interval_names)), interval_names, rotation=45)
    
    # Add out-of-bounds indicators before setting ylim
    if oob_data:
        for x, y_pos, actual_value, color, model in oob_data:
            # Draw upward arrow at the upper bound
            plt.annotate('', xy=(x, y_pos), xytext=(x, y_pos - 50),
                        arrowprops={'arrowstyle': '->', 'facecolor': color, 'edgecolor': color, 'lw': 1.5})
            
            # Add value annotation above the arrow
            plt.annotate(f'{int(actual_value)}', xy=(x, y_pos), xytext=(x, y_pos + 50),
                        ha='center', va='bottom', fontsize=9, color=color, fontweight='bold')
    
    # Set y-axis limits if specified
    if ylim is not None:
        plt.ylim(ylim)
    
    # Remove legend since we're not showing multiple models
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    if show:
        plt.show()


def plot_rmse_by_demographics(results_dir: str = "results", dataset_filter: str = None,
                             demographic: str = "age", bin_size: float = 10,
                             horizons: Union[List[int], str] = "all", save_path: str = None, show: bool = True,
                             ylim: Optional[Tuple[float, float]] = (0, 80), show_oob_indicators: bool = True):
    """
    Plot RMSE by demographic intervals (e.g., age groups) for all models.
    
    Args:
        results_dir: Path to results directory
        dataset_filter: Filter to specific dataset (if None, uses all datasets combined)
        demographic: Demographic column name (default: "age")
        bin_size: Interval size for demographic bins (default: 10)
        horizons: Forecast horizons to include (list of indices 0-11, or "all")
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
        ylim: Y-axis limits as (min, max) tuple. Set to None for auto-scaling.
        show_oob_indicators: Show arrows and values for out-of-bounds points. Default: True
    """
    # Import here to avoid circular imports
    from .results_loader import load_single_dataset_with_user_ids
    
    # Load model results
    dataset_names = [dataset_filter] if dataset_filter else None
    results = load_model_results(results_dir, dataset_names=dataset_names)
    
    if not results:
        raise ValueError("No results found")
    
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
    
    # Load demographics based on dataset filter
    if dataset_filter:
        # Single dataset case
        demographics = load_demographics(results_dir, dataset_filter, [demographic])
        user_demo_mapping = {}
        
        # Get user IDs from the first model for this dataset
        first_model = sorted(results.keys())[0]
        if dataset_filter in results[first_model]:
            _, combined_labels, combined_user_ids = load_single_dataset_with_user_ids(
                results_dir, first_model, dataset_filter)
            
            # Map result user IDs to demographics
            for user_id in np.unique(combined_user_ids[:, 0]):
                user_id_str = str(int(user_id))
                if user_id_str in demographics:
                    user_demo_mapping[user_id] = demographics[user_id_str]
        
        title_suffix = f"on {dataset_filter}"
    else:
        # All datasets case
        demographics = load_all_demographics(results_dir, [demographic])
        user_demo_mapping = {}
        
        # Process each dataset and model to build user mapping
        for model_name in results.keys():
            for dataset_name in results[model_name].keys():
                # Get user IDs for this dataset
                _, combined_labels, combined_user_ids = load_single_dataset_with_user_ids(
                    results_dir, model_name, dataset_name)
                
                # Map to global demographics
                for user_id in np.unique(combined_user_ids[:, 0]):
                    user_id_str = str(int(user_id))
                    global_user_id = f"{dataset_name}_{user_id_str}"
                    if global_user_id in demographics:
                        # Use dataset_user_id as key to avoid conflicts
                        key = f"{dataset_name}_{user_id}"
                        user_demo_mapping[key] = demographics[global_user_id]
        
        title_suffix = "on All Datasets"
    
    if not user_demo_mapping:
        raise ValueError(f"No demographic data found for {demographic}")
    
    # Create demographic bins
    demo_values = list(user_demo_mapping.values())
    
    # Filter out NaN values before computing min/max
    valid_demo_values = [v for v in demo_values if not (isinstance(v, float) and np.isnan(v))]
    
    if not valid_demo_values:
        raise ValueError(f"No valid (non-NaN) {demographic} data found. Some datasets may have missing {demographic} values.")
    
    print(f"Found {len(valid_demo_values)} users with valid {demographic} data (out of {len(demo_values)} total users)")
    
    min_demo = float(np.floor(min(valid_demo_values) / bin_size) * bin_size)
    max_demo = float(np.ceil(max(valid_demo_values) / bin_size) * bin_size)
    
    # Create bin edges and labels
    bin_edges = np.arange(min_demo, max_demo + bin_size, bin_size)
    bin_labels = [f"{int(edge)}-{int(edge + bin_size)}" for edge in bin_edges[:-1]]
    
    # Set Times New Roman font
    plt.rcParams['font.family'] = 'Times New Roman'
    
    plt.figure(figsize=(6, 4))
    
    # Store out-of-bounds information
    oob_data = []
    
    # Calculate RMSE for each model and demographic bin
    for model_name in sorted(results.keys()):
        rmse_by_bin = []
        
        for i, (bin_start, bin_end) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
            # Collect all predictions and labels for users in this demographic bin
            bin_predictions = []
            bin_labels_list = []
            
            if dataset_filter:
                # Single dataset case
                if dataset_filter in results[model_name]:
                    combined_preds, combined_labels, combined_user_ids = load_single_dataset_with_user_ids(
                        results_dir, model_name, dataset_filter)
                    
                    # Filter by users in this demographic bin
                    for sample_idx, user_id in enumerate(combined_user_ids[:, 0]):
                        if user_id in user_demo_mapping:
                            demo_value = user_demo_mapping[user_id]
                            # Skip NaN values
                            if isinstance(demo_value, float) and np.isnan(demo_value):
                                continue
                            if bin_start <= demo_value < bin_end:
                                for h_idx in horizon_indices:
                                    bin_predictions.append(combined_preds[sample_idx, h_idx])
                                    bin_labels_list.append(combined_labels[sample_idx, h_idx])
            else:
                # All datasets case
                for dataset_name in results[model_name].keys():
                    combined_preds, combined_labels, combined_user_ids = load_single_dataset_with_user_ids(
                        results_dir, model_name, dataset_name)
                    
                    # Filter by users in this demographic bin
                    for sample_idx, user_id in enumerate(combined_user_ids[:, 0]):
                        key = f"{dataset_name}_{user_id}"
                        if key in user_demo_mapping:
                            demo_value = user_demo_mapping[key]
                            # Skip NaN values
                            if isinstance(demo_value, float) and np.isnan(demo_value):
                                continue
                            if bin_start <= demo_value < bin_end:
                                for h_idx in horizon_indices:
                                    bin_predictions.append(combined_preds[sample_idx, h_idx])
                                    bin_labels_list.append(combined_labels[sample_idx, h_idx])
            
            # Calculate RMSE for this bin
            if len(bin_predictions) > 0:
                bin_predictions = np.array(bin_predictions)
                bin_labels_array = np.array(bin_labels_list)
                rmse = calculate_rmse(bin_predictions, bin_labels_array)
                rmse_by_bin.append(rmse)
            else:
                rmse_by_bin.append(np.nan)  # No data in this bin
        
        # Plot line for this model
        x_pos = np.arange(len(bin_labels))
        line = plt.plot(x_pos, rmse_by_bin, marker='o', linewidth=2, markersize=6, label=model_name)
        line_color = line[0].get_color()
        
        # Detect out-of-bounds values for indicators
        if show_oob_indicators and ylim is not None:
            upper_bound = ylim[1]
            for i, (x, rmse_val) in enumerate(zip(x_pos, rmse_by_bin)):
                if not np.isnan(rmse_val) and rmse_val > upper_bound:
                    oob_data.append((x, upper_bound, rmse_val, line_color, model_name))
    
    plt.xlabel(f'{demographic.title()} Range', fontsize=12)
    plt.ylabel('RMSE (mg/dL)', fontsize=12)
    
    # Update title
    title_parts = [f'RMSE by {demographic.title()}', title_suffix]
    if horizons != "all":
        title_parts.append(f'({horizon_label})')
    
    plt.title(' '.join(title_parts), fontsize=14, fontweight='bold')
    plt.xticks(range(len(bin_labels)), bin_labels, rotation=45)
    
    # Add out-of-bounds indicators before setting ylim
    if oob_data:
        for x, y_pos, actual_value, color, model in oob_data:
            # Draw upward arrow at the upper bound
            plt.annotate('', xy=(x, y_pos), xytext=(x, y_pos - 2),
                        arrowprops={'arrowstyle': '->', 'facecolor': color, 'edgecolor': color, 'lw': 1.5})
            
            # Add value annotation above the arrow
            plt.annotate(f'{actual_value:.1f}', xy=(x, y_pos), xytext=(x, y_pos + 1),
                        ha='center', va='bottom', fontsize=9, color=color, fontweight='bold')
    
    # Set y-axis limits if specified
    if ylim is not None:
        plt.ylim(ylim)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    if show:
        plt.show()