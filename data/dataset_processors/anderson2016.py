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

    cgm_data['LocalDtTm'] = pd.to_datetime(cgm_data['LocalDtTm'],
                                           format='%Y-%m-%d %H:%M:%S',
                                           errors='coerce').dt.floor('5min')
    insulin_data['LocalDeliveredDtTm'] = pd.to_datetime(
        insulin_data['LocalDeliveredDtTm'],
        format='%Y-%m-%d %H:%M:%S',
        errors='coerce').dt.floor('5min')
    insulin_datetimes = set(insulin_data['LocalDeliveredDtTm'].unique())

    patients = set(cgm_data['DeidentID'].unique()) & set(
        insulin_data['DeidentID'].unique())

    dataset_output = pd.DataFrame()
    for patient in tqdm(patients):
        patient_cgm = cgm_data[cgm_data['DeidentID'] == patient]
        patient_insulin = insulin_data[insulin_data['DeidentID'] == patient]

        patient_cgm = patient_cgm.rename(columns={'LocalDtTm': 'DataDtTm', 'DeidentID': 'PtID', 'CGM': 'CGM'})
        patient_insulin = patient_insulin.rename(columns={'LocalDeliveredDtTm': 'DataDtTm', 'DeidentID': 'PtID', 'DeliveredValue': 'Insulin'})

        patient_cgm['DataDtTm'] = patient_cgm['DataDtTm'].dt.floor('5min')
        patient_insulin['DataDtTm'] = patient_insulin['DataDtTm'].dt.floor('5min')
        dataset_output = pd.concat([dataset_output, patient_cgm.merge(patient_insulin, how='outer')])

    return dataset_output