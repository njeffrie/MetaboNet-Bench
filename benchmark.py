from models.gluformer import Gluformer
import numpy as np
from tqdm import tqdm
from datasets import Dataset

def load_dataset(name: str):
    parquet_files = f'data/{name}/*.parquet'
    return Dataset.from_parquet(parquet_files)

def calculate_rmse(pred, label):
    return np.sqrt(np.mean((pred - label) ** 2))

def calculate_ape(pred, label):
    return np.mean(np.abs(pred - label) / np.abs(label))

def main():
    cgm_data_set = load_dataset('Brown2019')
    ds_len = 1000#len(cgm_data_set)
    gluformer = Gluformer()
    horizons = [3, 6, 9, 12] # 15, 30, 45, 60 minutes.
    rmses = np.zeros((ds_len, len(horizons)))
    apes = np.zeros((ds_len, len(horizons)))
    i = 0
    for idx in tqdm(range(0, ds_len * 192, 192)):
        df = cgm_data_set[idx:idx+192]
        timestamps = df['DataDtTm'][:180]
        glucose_values = df['CGM']
        model_input = glucose_values[:180]
        label = glucose_values[180:192]
        subject_id = df['PtID'][0]

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