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

def dts_zone_counts(labels,
                    predictions,
                    dts_grid_path,
                    extent=(-62, 835, -47, 646)):
    # Pre-calculated RGB values for each DTS zone.
    zone_rgb = {
        'A': np.array([0.5647059, 0.72156864, 0.5019608], dtype=np.float32),
        'B': np.array([1.0039216, 1.0039216, 0.59607846], dtype=np.float32),
        'C': np.array([0.972549, 0.8156863, 0.5647059], dtype=np.float32),
        'D': np.array([0.9411765, 0.53333336, 0.5019608], dtype=np.float32),
        'E': np.array([0.78431374, 0.53333336, 0.65882355], dtype=np.float32),
    }
    r, p = map(lambda x: np.asarray(x).ravel(), (labels, predictions))

    img = plt.imread(dts_grid_path).astype(np.float32)
    h, w = img.shape[:2]
    xmin, xmax, ymin, ymax = extent

    xi = np.round((r - xmin) / (xmax - xmin) * (w - 1)).astype(int)
    yi = np.round((ymax - p) / (ymax - ymin) * (h - 1)).astype(int)
    pix = img[yi, xi, :3]

    keys = np.array(list(zone_rgb), dtype='<U1')
    cols = np.stack([zone_rgb[k] for k in keys], axis=0)
    z = keys[np.argmin(((pix[:, None] - cols)**2).sum(-1), axis=1)]

    return {k: int((z == k).sum()) for k in 'ABCDE'}

def plot_prediction_percentiles(predictions,
                                labels,
                                dataset_name,
                                model_name,
                                save_path=None,
                                dts_grid_path='data/dts_grid.png',
                                plot_subset_size=None):
    ph_index = 1
    zone_counts = dts_zone_counts(labels[:, ph_index], predictions[:, ph_index], dts_grid_path)
    total_samples = len(labels)
    if plot_subset_size is not None:
        predictions = predictions[-plot_subset_size:, :]
        labels = labels[-plot_subset_size:, :]
    r, p = map(lambda x: np.asarray(x).ravel(), (labels[:, ph_index], predictions[:, ph_index]))
    plt.figure(figsize=(10, 7.5), dpi=150)
    plt.imshow(
        plt.imread(dts_grid_path),
        extent=(-62, 835, -47, 646),
        origin='upper',
        aspect='auto',
    )
    plt.scatter(
        r,
        p,
        s=6,
        facecolors='white',
        edgecolors='black',
        linewidths=0.4,
    )
    plt.axis('off')
    zone_counts = {k: round(v/total_samples*100, 2) for k, v in zone_counts.items()}
    plt.text(0.05, -0.01, f'Zone A: {zone_counts["A"]}%, Zone B: {zone_counts["B"]}%, Zone C: {zone_counts["C"]}%, Zone D: {zone_counts["D"]}%, Zone E: {zone_counts["E"]}%', transform=plt.gca().transAxes, fontsize=12, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.savefig(f'plots/{model_name}-{dataset_name}-DTS.png')

    # Percentile + individual error plots (was plot_prediction_percentiles)
    diffs = predictions - labels
    pct = np.percentile(diffs, [10, 25, 50, 75, 90], axis=0)
    print(predictions.shape, labels.shape)
    t = np.arange(1, predictions.shape[1] + 1) * 5

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    ax1.fill_between(
        t,
        pct[0],
        pct[4],
        alpha=0.3,
        color='blue',
        label='Predictions 10th-90th percentile',
    )
    ax1.fill_between(
        t,
        pct[1],
        pct[3],
        alpha=0.5,
        color='blue',
        label='Predictions 25th-75th percentile',
    )
    ax1.plot(t, pct[2], 'b-', linewidth=2, label='Predictions median')
    ax1.set(
        xlabel='Time (minutes)',
        ylabel='Absolute Prediction Error',
        title=(f'{model_name.upper()} Prediction Error Percentiles on '
               f'{dataset_name}\nPercentiles across {predictions.shape[0]} '
               'samples'),
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    for y in diffs:
        ax2.plot(t, y, alpha=0.1, linewidth=1)
    ax2.set(
        xlabel='Time (minutes)',
        ylabel='Absolute Prediction Error',
        title=(f'{model_name.upper()} Individual Prediction Errors on '
               f'{dataset_name}'),
    )
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Plot saved to {save_path}')

def run_batch(model_runner, input_batch, label_batch, horizons):
    input_batch = np.stack(input_batch, axis=0)
    label_batch = np.stack(label_batch, axis=0)
    ts, cgm, insulin, carbs = np.split(input_batch, 4, axis=1)
    pred = model_runner.predict(ts.squeeze(1), cgm.squeeze(1), insulin.squeeze(1), carbs.squeeze(1))
    preds = pred[:, np.array(horizons)-1]
    labels = label_batch[:,np.array(horizons)-1]
    return preds, labels

def run_benchmark(model, ds, plot, csv_file=None, batch_size=1):
    model_runner = get_model(model.lower())
    horizons = [3, 6, 9, 12]  # 15, 30, 45, 60 minutes.
    #flops, macs, params = calculate_flops(model=model_runner.model,
    #                                      args = [0, np.random.rand(1, 180), np.random.rand(1, 180)],
    #                                      print_results=False)
    #print(f'{model} FLOPS: {flops}, MACS: {macs}, PARAMS: {params}')
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
            save_plot = f'plots/{model}-{ds_name}-percentiles.png'

            print(f"\nGenerating prediction vs label plots...")
            # Plot for 30 minute PH.
            plot_prediction_percentiles(all_predictions,
                                        all_labels,
                                        ds_name,
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
@click.option('--batch_size',
              type=int,
              default=1,
              help='Batch size to run the benchmark on')
def main(model, plot, save_csv, subset_size=None, batch_size=1):
    ds = load_dataset()
    if subset_size is not None:
        ds = ds.take(subset_size)
    models = model.split(',')
    csv_file = open(save_csv, 'w')
    csv_file.write(f'model,dataset,rmse_15,rmse_30,rmse_45,rmse_60,ape_15,ape_30,ape_45,ape_60\n')

    for model in models:
        run_benchmark(model, ds, plot, csv_file, batch_size=batch_size)



if __name__ == "__main__":
    main()
