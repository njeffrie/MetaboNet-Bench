from models.models import get_model
import numpy as np
from tqdm import tqdm
from datasets import Dataset
import click
import matplotlib.pyplot as plt
import os
import re

def load_dataset():
    ds_path = f'data/metabonet_test.parquet'
    ds = Dataset.from_parquet(ds_path)
    ds.set_format('pandas')
    return ds

def run_batch(model_runner, input_batch, label_batch):
    input_batch = np.stack(input_batch, axis=0)
    labels = np.stack(label_batch, axis=0)
    ts, cgm, insulin, carbs = np.split(input_batch, 4, axis=1)
    preds = model_runner.predict(ts.squeeze(1), cgm.squeeze(1), insulin.squeeze(1), carbs.squeeze(1))
    return preds, labels

def run_benchmark(model, ds, batch_size=1):
    model_runner = get_model(model.lower())
    min_sequence_length = 192
    pred_len = 12

    total_predictions = np.zeros((0, pred_len))
    total_labels = np.zeros((0, pred_len))
    for ds_name, selected_dataset in ds.to_pandas().groupby('DatasetName'):
        print(len(selected_dataset))

        all_predictions = np.zeros((0, pred_len))
        all_labels = np.zeros((0, pred_len))
        all_patient_ids = np.zeros((0, pred_len))
        input_batch = []
        label_batch = []
        all_ts = np.zeros((0, pred_len))
        for patient_id, patient_data in tqdm(selected_dataset.groupby('PtID'), position=0, leave=False):
            # Remove any non-numeric portions of the patient id.
            patient_id = int(re.findall(r'\d+', patient_id)[-1])
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
                    all_ts = np.concatenate([all_ts, np.array(timestamps[-12:]).reshape(1, -1)], axis=0)
                    if len(input_batch) == batch_size:
                        preds, labels = run_batch(model_runner, input_batch, label_batch)
                        all_predictions = np.concatenate([all_predictions, preds], axis=0)
                        all_labels = np.concatenate([all_labels, labels], axis=0)
                        all_patient_ids = np.concatenate([all_patient_ids, np.full((batch_size, labels.shape[1]), patient_id)], axis=0)
                        input_batch = []
                        label_batch = []

        if len(input_batch) > 0:
            preds, labels = run_batch(model_runner, input_batch, label_batch)
            all_predictions = np.concatenate([all_predictions, preds], axis=0)
            all_labels = np.concatenate([all_labels, labels], axis=0)
            all_patient_ids = np.concatenate([all_patient_ids, np.full((labels.shape[0], labels.shape[1]), patient_id)], axis=0)
        all_predictions = np.clip(all_predictions, 40, 600)

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
        total_results = np.stack([all_patient_ids, all_predictions, all_labels], axis=1)
        np.save(f'results/{model}/{ds_name}.npy', total_results)

    total_rmses = np.sqrt(np.mean((total_labels - total_predictions)**2, axis=0))
    total_apes = np.mean(np.abs(total_labels - total_predictions) / np.abs(total_labels), axis=0)
    print(f'Total RMSE: {total_rmses.round(2)}')
    print(f'Total APE: {total_apes.round(4)}')

@click.command()
@click.option('--model',
              type=str,
              default='zoh',
              help='List of models to run the benchmark on')
@click.option('--subset_size',
              type=int,
              default=None,
              help='Subset size to run the benchmark on')
@click.option('--batch_size',
              type=int,
              default=1,
              help='Batch size to run the benchmark on')
def main(model, subset_size=None, batch_size=1):
    ds = load_dataset()
    if subset_size is not None:
        ds = ds.take(subset_size)
    models = model.split(',')
    for model in models:
        if not os.path.exists(f'results/{model}'):
            os.makedirs(f'results/{model}')

    for model in models:
        run_benchmark(model, ds, batch_size=batch_size)



if __name__ == "__main__":
    main()
