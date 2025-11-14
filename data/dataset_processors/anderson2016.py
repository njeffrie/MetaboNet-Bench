import pandas as pd
import numpy as np
from tqdm import tqdm


# 1. normalize cgm time
# 2. normalize insulin time
# 3. merge cgm and insulin data
# 4. 
def preprocess(ds_dir):
    cgm_file = f'{ds_dir}/Data Tables/MonitorCGM.txt'
    insulin_file = f'{ds_dir}/Data Tables/MonitorTotalBolus.txt'

    cgm_data = pd.read_csv(cgm_file, sep='|')
    insulin_data = pd.read_csv(insulin_file, sep='|')

    cgm_data['LocalDtTm'] = pd.to_datetime(cgm_data['LocalDtTm'], format='%Y-%m-%d %H:%M:%S', errors='coerce').dt.floor('5min')
    insulin_data['LocalDeliveredDtTm'] = pd.to_datetime(insulin_data['LocalDeliveredDtTm'], format='%Y-%m-%d %H:%M:%S', errors='coerce').dt.floor('5min')
    insulin_datetimes = set(insulin_data['LocalDeliveredDtTm'].unique())

    patients = set(cgm_data['DeidentID'].unique()) & set(insulin_data['DeidentID'].unique())

    dataset_output = {'PtID': [], 'DataDtTm': [], 'CGM': [], 'Insulin': []}
    for patient in tqdm(patients):
        patient_cgm = cgm_data[cgm_data['DeidentID'] == patient]
        patient_insulin = insulin_data[insulin_data['DeidentID'] == patient]

        for idx, sample in patient_cgm.iterrows():
            dt = sample['LocalDtTm']
            if dt not in insulin_datetimes:
                continue
            insulin_delivered = patient_insulin[patient_insulin['LocalDeliveredDtTm'] == dt]['DeliveredValue'].values
            insulin_delivered = round(insulin_delivered[0], 2) if len(insulin_delivered) > 0 else 0
            dataset_output['PtID'].append(patient)
            dataset_output['DataDtTm'].append(dt)
            dataset_output['CGM'].append(sample['CGM'])
            dataset_output['Insulin'].append(insulin_delivered)

    dataset = pd.DataFrame(dataset_output)
    return dataset


