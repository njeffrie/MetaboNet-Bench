from models.models import get_model
import numpy as np
from tqdm import tqdm
from datasets import Dataset
import click
import matplotlib.pyplot as plt

def load_dataset(name: str, split: str):
    ds_path = f'data/{name}/dataset/{split}'
    ds = Dataset.load_from_disk(ds_path)
    ds.set_format('torch')
    return ds

def calculate_rmse(pred, label):
    return np.sqrt(np.mean((pred - label) ** 2))

def calculate_ape(pred, label):
    return np.mean(np.abs(pred - label) / np.abs(label))

def plot_prediction_percentiles(predictions, labels, rmse, ape, dataset_name, model_name, save_path=None):
    """
    Plot percentiles of predictions vs labels throughout the prediction window.
    
    Args:
        predictions: numpy array of shape (n_samples, prediction_length)
        labels: numpy array of shape (n_samples, prediction_length)
        dataset_name: name of the dataset for the plot title
        model_name: name of the model for the plot title
        save_path: optional path to save the plot
    """
    # Calculate percentiles across samples for each time point
    pred_differences = predictions - labels
    label_percentiles = np.percentile(pred_differences, [10, 25, 75, 90], axis=0)    
    
    # Time points (assuming 5-minute intervals)
    time_points = np.arange(1, predictions.shape[1] + 1) * 5  # minutes
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # Subplot 1: Percentiles
    ax1.fill_between(time_points, label_percentiles[0], label_percentiles[3], 
                     alpha=0.3, color='blue', label='Predictions 10th-90th percentile')
    ax1.fill_between(time_points, label_percentiles[1], label_percentiles[2], 
                     alpha=0.5, color='blue', label='Predictions 25th-75th percentile')
    ax1.plot(time_points, label_percentiles[1], 'b-', linewidth=2, label='Predictions median')
    
    ax1.set_xlabel('Time (minutes)')
    ax1.set_ylabel('Absolute Prediction Error')
    ax1.set_title(f'{model_name.upper()} Prediction Error Percentiles on {dataset_name}\n'
                  f'Percentiles across {predictions.shape[0]} samples')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    
    ax1.text(0.02, 0.98, f'Final RMSE: {rmse}\nFinal APE: {ape}%', 
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Subplot 2: Individual error lines
    for pred_diff in pred_differences[-1000:]:
        ax2.plot(time_points, pred_diff, alpha=0.1, linewidth=1)
    
    ax2.set_ylabel('Absolute Prediction Error')
    ax2.set_xlabel('Time (minutes)')
    ax2.set_title(f'{model_name.upper()} Individual Prediction Errors on {dataset_name}')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

@click.command()
@click.option('--dataset', type=str, default='Brown2019', help='Dataset to run the benchmark on')
@click.option('--model', type=str, default='gluformer', help='Model to run the benchmark on')
@click.option('--subset_size', type=int, default=None, help='Subset size to run the benchmark on')
@click.option('--plot', is_flag=True, help='Generate prediction vs label plots')
@click.option('--save_plot', type=str, default=None, help='Path to save the plot (e.g., "plots/gluformer_brown2019.png")')
def main(dataset, model, plot, save_plot, split='test', subset_size=None):
    ds = load_dataset(dataset, split)
    if subset_size is not None:
        ds = ds.take(subset_size)
    ds_len = len(ds)
    model_runner = get_model(model.lower())
    horizons = [3, 6, 9, 12] # 15, 30, 45, 60 minutes.
    
    rmses = np.zeros((ds_len, len(horizons)))
    apes = np.zeros((ds_len, len(horizons)))
    
    # Store all predictions and labels for plotting
    all_predictions = []
    all_labels = []
    
    patient_ids = ds['PtID'].unique()

    i = 0
    for patient_id in tqdm(patient_ids):
        patient_data = ds[ds['PtID'] == patient_id]
        timestamps = patient_data['DataDtTm']

        # Calculate time differences between consecutive readings
        timestamps_diff = timestamps.diff().dt.total_seconds() / 60  # Convert to minutes
        
        # Identify sequence breaks (gaps larger than max_gap_minutes)
        # First row will have NaN time_diff, so we mark it as False (not a break)
        sequence_breaks = timestamps_diff > max_gap_minutes
        sequence_breaks.iloc[0] = False  # First reading starts a new sequence

        # Create sequence IDs
        sequence_ids = sequence_breaks.cumsum()

        for sequence_id in sequence_ids.unique():
            sequence_data = patient_data[sequence_ids == sequence_id]
            if len(sequence_data) < min_sequence_length:
                continue
            for i in range(0, len(sequence_data) - min_sequence_length + 1, 12): # increment by one hour.
                model_input = sequence_data['CGM'].values[-192:-12]
                label = sequence_data['CGM'].values[-12:].numpy()
                pred = model_runner.predict(patient_id, timestamps, model_input)
                pred = pred.flatten()

                # Calculate metrics for each horizon
                for j in range(len(horizons)):
                    rmses[i, j] = calculate_rmse(pred[horizons[j]-1], label[horizons[j]-1])
                    apes[i, j] = calculate_ape(pred[horizons[j]-1], label[horizons[j]-1])
                i += 1

    # Print results
    print(f'\nResults for {model} on {dataset}:')
    print(f'Root Mean Squared Error (RMSE) at 15/30/45/60 minutes: {np.mean(rmses, axis=0)}')
    print(f'Absolute Percent Error (APE) at 15/30/45/60 minutes: {np.mean(apes, axis=0)}')

    # Generate plots if requested
    if plot:
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        print(f"\nGenerating prediction vs label plots...")
        plot_prediction_percentiles(
            all_predictions, 
            all_labels,
            np.round(np.mean(rmses, axis=0), 2),
            np.round(100 * np.mean(apes, axis=0), 2),
            dataset, 
            model, 
            save_path=save_plot
        )

if __name__ == "__main__":
    main()