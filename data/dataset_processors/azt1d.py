import pandas as pd
import os
import re

# EventDateTime,DeviceMode,BolusType,Basal,CorrectionDelivered,TotalBolusInsulinDelivered,FoodDelivered,CarbSize,CGM


def preprocess(ds_dir):
    data_dir = f'{ds_dir}/AZT1D 2025/CGM Records'

    dataset_output = pd.DataFrame()

    for patient_dir in os.listdir(data_dir):
        patient_match = re.search(r'\d+', patient_dir)
        if patient_match is None:
            continue
        patient_id = int(patient_match.group(0))
        patient_data = pd.read_csv(
            f'{data_dir}/{patient_dir}/{patient_dir}.csv')
        patient_data['EventDateTime'] = pd.to_datetime(
            patient_data['EventDateTime'], format='%Y-%m-%d %H:%M:%S')

        # Basal info is missing from the start of the data for each user.
        start_time = patient_data[~patient_data['Basal'].isna(
        )].iloc[0]['EventDateTime']
        patient_data = patient_data[patient_data['EventDateTime'] >= start_time]
        patient_data['PtID'] = [patient_id] * len(patient_data)
        patient_data['DataDtTm'] = patient_data['EventDateTime'].dt.floor(
            '5min')
        patient_data['CGM'] = patient_data[
            'CGM'] if 'CGM' in patient_data.columns else patient_data[
                'Readings (CGM / BGM)']
        patient_data[
            'Insulin'] = patient_data['TotalBolusInsulinDelivered'].values + (
                patient_data['Basal'] / 12.0).values

        dataset_output = pd.concat([
            dataset_output, patient_data[['PtID', 'DataDtTm', 'CGM', 'Insulin']]
        ])

    dataset = pd.DataFrame(dataset_output)

    return dataset
