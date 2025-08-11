from models.gluformer import Gluformer
import numpy as np
from tqdm import tqdm
from datasets import Dataset
import pandas as pd
import click

def load_dataset(name: str, split: str):
    parquet_files = f'data/{name}/dataset_{split}.parquet'
    return Dataset.from_parquet(parquet_files)['sequences']

def calculate_rmse(pred, label):
    return np.sqrt(np.mean((pred - label) ** 2))

def calculate_ape(pred, label):
    return np.mean(np.abs(pred - label) / np.abs(label))

@click.command()
@click.option('--dataset', type=str, default='Brown2019', help='Dataset to run the benchmark on')
def main(dataset, split='test'):
    cgm_data_set = load_dataset(dataset, split)
    ds_len = len(cgm_data_set)
    gluformer = Gluformer()
    horizons = [3, 6, 9, 12] # 15, 30, 45, 60 minutes.
    rmses = np.zeros((ds_len, len(horizons)))
    apes = np.zeros((ds_len, len(horizons)))
    for i, sample in tqdm(enumerate(cgm_data_set)):
        timestamps = pd.to_datetime(sample[1][:180]).to_list()
        glucose_values = sample[2]
        model_input = glucose_values[:180]
        label = glucose_values[180:192]
        subject_id = sample[0][0]

        pred, log_var = gluformer.predict(subject_id, timestamps, model_input)
        pred = pred.flatten()
        #print(f'pred {pred.shape} label {label.shape}')
        for j in range(len(horizons)):
            rmses[i, j] = calculate_rmse(pred[horizons[j]-1], label[horizons[j]-1])
            apes[i, j] = calculate_ape(pred[horizons[j]-1], label[horizons[j]-1])
        i += 1

    np.save('rmses.npy', rmses)
    np.save('apes.npy', apes)
    print(f'average mse {np.mean(rmses, axis=0)}')
    print(f'average ape {np.mean(apes, axis=0)}')

if __name__ == "__main__":
    main()