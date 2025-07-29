from brown_2019 import CGMDataSet
from models.gluformer import Gluformer
import numpy as np
from tqdm import tqdm

def calculate_rmse(pred, label):
    return np.sqrt(np.mean((pred - label) ** 2))

def calculate_ape(pred, label):
    return np.mean(np.abs(pred - label) / np.abs(label))

def main():
    cgm_data_set = CGMDataSet()
    ds_len = cgm_data_set.num_consecutive_sequences(min_sequence_length=192)
    gluformer = Gluformer()
    horizons = [3, 6, 9, 12] # 15, 30, 45, 60 minutes.
    rmses = np.zeros((ds_len, len(horizons)))
    apes = np.zeros((ds_len, len(horizons)))
    for i, sequence in tqdm(enumerate(cgm_data_set.yield_consecutive_cgm_sequences(min_sequence_length=192)), total=ds_len):
        df = sequence.to_pandas()
        timestamps = df['DataDtTm'].tolist()[:180]
        glucose_values = df['CGM'].tolist()
        model_input = glucose_values[:180]
        label = glucose_values[180:192]
        subject_id = df['PtID'].iloc[0]

        pred, log_var = gluformer.predict(subject_id, timestamps, model_input)
        pred = pred.flatten()
        #print(f'pred {pred.shape} label {label.shape}')
        for j in range(len(horizons)):
            rmses[i, j] = calculate_rmse(pred[horizons[j]-1], label[horizons[j]-1])
            apes[i, j] = calculate_ape(pred[horizons[j]-1], label[horizons[j]-1])
        
    print(f'average mse {np.mean(rmses)}')
    print(f'average ape {np.mean(apes)}')


if __name__ == "__main__":
    main()