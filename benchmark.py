from models.models import get_model
import numpy as np
from tqdm import tqdm
from datasets import Dataset
import click
import matplotlib.pyplot as plt


def load_dataset():
    ds_path = f'data/metabonet_test.parquet'
    ds = Dataset.from_parquet(ds_path)
    ds.set_format('pandas')
    return ds


def calculate_rmse(pred, label):
    return np.sqrt(np.mean((pred - label)**2))


def calculate_ape(pred, label):
    return np.mean(np.abs(pred - label) / np.abs(label))


def plot_prediction_percentiles(predictions,
                                labels,
                                rmse,
                                ape,
                                dataset_name,
                                model_name,
                                save_path=None):
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
    print(f'predictions: {predictions.shape}, labels: {labels.shape}')
    pred_differences = predictions - labels
    label_percentiles = np.percentile(pred_differences, [10, 25, 50, 75, 90],
                                      axis=0)

    # Time points (assuming 5-minute intervals)
    time_points = np.arange(1, predictions.shape[1] + 1) * 5  # minutes

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    # Subplot 1: Percentiles
    ax1.fill_between(time_points,
                     label_percentiles[0],
                     label_percentiles[4],
                     alpha=0.3,
                     color='blue',
                     label='Predictions 10th-90th percentile')
    ax1.fill_between(time_points,
                     label_percentiles[1],
                     label_percentiles[3],
                     alpha=0.5,
                     color='blue',
                     label='Predictions 25th-75th percentile')
    ax1.plot(time_points,
             label_percentiles[2],
             'b-',
             linewidth=2,
             label='Predictions median')

    ax1.set_xlabel('Time (minutes)')
    ax1.set_ylabel('Absolute Prediction Error')
    ax1.set_title(
        f'{model_name.upper()} Prediction Error Percentiles on {dataset_name}\n'
        f'Percentiles across {predictions.shape[0]} samples')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax1.text(0.02,
             0.98,
             f'Final RMSE: {rmse}\nFinal APE: {ape}%',
             transform=ax1.transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Subplot 2: Individual error lines
    for pred_diff in pred_differences[-1000:]:
        ax2.plot(time_points, pred_diff, alpha=0.1, linewidth=1)

    ax2.set_ylabel('Absolute Prediction Error')
    ax2.set_xlabel('Time (minutes)')
    ax2.set_title(
        f'{model_name.upper()} Individual Prediction Errors on {dataset_name}')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

def run_batch(model_runner, input_batch, label_batch, horizons):
    input_batch = np.stack(input_batch, axis=0)
    label_batch = np.stack(label_batch, axis=0)
    ts, cgm, insulin, carbs = np.split(input_batch, 4, axis=1)
    pred = model_runner.predict(ts.squeeze(1), cgm.squeeze(1), insulin.squeeze(1), carbs.squeeze(1))
    preds = pred[:, np.array(horizons)-1]
    labels = label_batch[:,np.array(horizons)-1]
    return preds, labels

def run_benchmark(model, ds, plot, csv_file=None, batch_size=1024):
    model_runner = get_model(model.lower())
    horizons = [3, 6, 9, 12]  # 15, 30, 45, 60 minutes.

    min_sequence_length = 192

    total_predictions = np.zeros((0, len(horizons)))
    total_labels = np.zeros((0, len(horizons)))
    for _, selected_dataset in ds.to_pandas().groupby('DatasetName'):
        ds_name = selected_dataset['DatasetName'].values[0]

        all_predictions = np.zeros((0, len(horizons)))
        all_labels = np.zeros((0, len(horizons)))
        input_batch = []
        label_batch = []
        for _, patient_data in tqdm(selected_dataset.groupby('PtID'), position=0, leave=False):
            for _, sequence_data in tqdm(patient_data.groupby('SequenceID'),
                                        position=1,
                                        leave=False):
                if len(sequence_data) < min_sequence_length:
                    continue
                for i in tqdm(range(0,
                                    len(sequence_data) - min_sequence_length + 1,
                                    12),
                            position=2,
                            leave=False):  # increment by one hour.
                    cgm_values = sequence_data['CGM'].values[i:i + 192]
                    timestamps = sequence_data['DataDtTm'].values[i:i + 180].astype(np.int64)
                    model_input = cgm_values[-192:-12]
                    label = cgm_values[-12:]
                    insulin_values = sequence_data['Insulin'].values[i:i + 180]
                    carbs_values = sequence_data['Carbs'].values[i:i + 180]
                    input_batch.append(np.stack([timestamps, model_input, insulin_values, carbs_values], axis=0))
                    label_batch.append(label)
                    if len(input_batch) == batch_size:
                        preds, labels = run_batch(model_runner, input_batch, label_batch, horizons)
                        all_predictions = np.concatenate([all_predictions, preds], axis=0)
                        all_labels = np.concatenate([all_labels, labels], axis=0)
                        input_batch = []
                        label_batch = []

        if len(input_batch) > 0:
            preds, labels = run_batch(model_runner, input_batch, label_batch, horizons)
            all_predictions = np.concatenate([all_predictions, preds], axis=0)
            all_labels = np.concatenate([all_labels, labels], axis=0)

        rmses = np.sqrt(np.mean((all_labels - all_predictions)**2, axis=0))
        apes = np.mean(np.abs(all_labels - all_predictions) / np.abs(all_labels), axis=0)
        # Print results
        print(
            f'{model},{ds_name},{",".join([str(round(float(x), 2)) for x in list(rmses.round(2))])}'
        )
        print(
            f'{model},{ds_name},{",".join([str(round(float(x), 2)) for x in list(apes.round(4) * 100)])}'
        )
        total_predictions = np.concatenate([total_predictions, all_predictions], axis=0)
        total_labels = np.concatenate([total_labels, all_labels], axis=0)
        if csv_file is not None:
            predictions_str = ",".join([str(round(float(x), 2)) for x in list(rmses.round(2))])
            labels_str = ",".join([str(round(float(x), 2)) for x in list(apes.round(2))])
            csv_file.write(f'{model},{ds_name},{predictions_str},{labels_str}\n')
        # Generate plots if requested
        if plot:
            save_plot = f'plots/{model}-{dataset}.png'

            print(f"\nGenerating prediction vs label plots...")
            plot_prediction_percentiles(total_predictions,
                                        total_labels,
                                        rmses,
                                        apes,
                                        dataset,
                                        model,
                                        save_path=save_plot)
    total_rmses = np.sqrt(np.mean((total_labels - total_predictions)**2, axis=0))
    total_apes = np.mean(np.abs(total_labels - total_predictions) / np.abs(total_labels), axis=0)
    print(f'Total RMSE: {total_rmses.round(2)}')
    print(f'Total APE: {total_apes.round(4)}')

@click.command()
@click.option('--model',
              type=str,
              default='gluformer',
              help='List of models to run the benchmark on')
@click.option('--subset_size',
              type=int,
              default=None,
              help='Subset size to run the benchmark on')
@click.option('--plot', is_flag=True, help='Generate prediction vs label plots')
@click.option('--save_csv',
              type=str,
              default='results.csv',
              help='Path to save the results (e.g., "results.csv")')
def main(model, plot, save_csv, subset_size=None):
    ds = load_dataset()
    if subset_size is not None:
        ds = ds.take(subset_size)
    models = model.split(',')
    csv_file = open(save_csv, 'w')
    csv_file.write(f'model,dataset,rmse_15,rmse_30,rmse_45,rmse_60,ape_15,ape_30,ape_45,ape_60\n')

    for model in models:
        run_benchmark(model, ds, plot, csv_file)



if __name__ == "__main__":
    main()
