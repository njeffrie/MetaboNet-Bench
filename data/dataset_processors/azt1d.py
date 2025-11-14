import pandas as pd
import os
import re

# EventDateTime,DeviceMode,BolusType,Basal,CorrectionDelivered,TotalBolusInsulinDelivered,FoodDelivered,CarbSize,CGM

def preprocess(ds_dir):
    data_dir = f'{ds_dir}/AZT1D 2025/CGM Records'

    dataset_output = {'PtID': [], 'DataDtTm': [], 'CGM': [], 'Insulin': []}

    for patient_dir in os.listdir(data_dir):
        patient_match = re.search(r'\d+', patient_dir)
        if patient_match is None:
            continue
        patient_id = int(patient_match.group(0))
        patient_data = pd.read_csv(f'{data_dir}/{patient_dir}/{patient_dir}.csv')
        patient_data['EventDateTime'] = pd.to_datetime(patient_data['EventDateTime'], format='%Y-%m-%d %H:%M:%S')

        patient_data = patient_data.sort_values('EventDateTime')
        patient_data = patient_data.reset_index(drop=True)

        # Basal info is missing from the start of the data for each user.
        start_time = patient_data[~patient_data['Basal'].isna()].iloc[0]['EventDateTime']
        patient_data = patient_data[patient_data['EventDateTime'] >= start_time]

        dataset_output['PtID'] = [patient_id] * len(patient_data)
        dataset_output['CGM'] = patient_data['CGM'] if 'CGM' in patient_data.columns else patient_data['Readings (CGM / BGM)']
        dataset_output['Insulin'] = patient_data['TotalBolusInsulinDelivered'] + patient_data['Basal'] / 12.0
        dataset_output['DataDtTm'] = patient_data['EventDateTime'].dt.floor('5min')

    dataset = pd.DataFrame(dataset_output)

    return dataset
