import os
import pandas as pd
import numpy as np
import re


def preprocess(ds_dir):
    cgm_dir = f'{ds_dir}/ManchesterCSCoordinatedDiabetesStudy-1.0.4/Glucose Data'
    basal_dir = f'{ds_dir}/ManchesterCSCoordinatedDiabetesStudy-1.0.4/Insulin Data/Basal Data'
    bolus_dir = f'{ds_dir}/ManchesterCSCoordinatedDiabetesStudy-1.0.4/Insulin Data/Bolus Data'

    dataset_output = {'PtID': [], 'DataDtTm': [], 'CGM': [], 'Insulin': []}

    patient_ids = [
        re.search(r'\d+', patient_dir).group(0)
        for patient_dir in os.listdir(basal_dir)
        if re.search(r'\d+', patient_dir) is not None
    ]
    for patient_id in patient_ids:
        cgm_data = pd.read_csv(f'{cgm_dir}/UoMGlucose{patient_id}.csv')
        basal_data = pd.read_csv(f'{basal_dir}/UoMBasal{patient_id}.csv')
        bolus_data = pd.read_csv(f'{bolus_dir}/UoMBolus{patient_id}.csv')

        # Remove patients on long-acting insulin.
        if 'L' in basal_data['insulin_kind'].values:
            continue

        cgm_data['DataDtTm'] = pd.to_datetime(
            cgm_data['bg_ts'], format='%d/%m/%Y %H:%M').dt.floor('5min')
        basal_data['DataDtTm'] = pd.to_datetime(
            basal_data['basal_ts'], format='%d/%m/%Y %H:%M').dt.floor('5min')
        bolus_data['DataDtTm'] = pd.to_datetime(
            bolus_data['bolus_ts'], format='%d/%m/%Y %H:%M').dt.floor('5min')

        # Convert CGM from mmol/L to mg/dL.
        cgm_data['value'] = cgm_data['value'].astype(float) * 18.0182
        start_time = max(cgm_data['DataDtTm'].min(),
                         basal_data['DataDtTm'].min())
        end_time = min(cgm_data['DataDtTm'].max(), basal_data['DataDtTm'].max())

        # Fill in all 5 minute intervals during the study period.
        basal_times = pd.date_range(start_time, end_time, freq='5min')
        basal_data = basal_data.merge(pd.DataFrame({'DataDtTm': basal_times}),
                                      on='DataDtTm',
                                      how='outer')

        # Interpolate CGM and calculate insulin per 5 minutes.
        total_data = cgm_data.merge(basal_data, on='DataDtTm',
                                    how='outer').merge(bolus_data,
                                                       on='DataDtTm',
                                                       how='outer')
        total_data['value'] = total_data['value'].interpolate(
            method='linear', limit_direction='forward', limit=6).astype(float)
        total_data['basal_dose'] = total_data['basal_dose'].ffill().astype(
            float) / 12.0
        total_data['bolus_dose'] = total_data['bolus_dose'].fillna(
            value=0).astype(float)
        total_data = total_data[(total_data['DataDtTm'] >= start_time) &
                                (total_data['DataDtTm'] <= end_time)]

        # Remove rows with missing CGM data.
        total_data = total_data[total_data['value'].notna()]
        total_data = total_data.sort_values('DataDtTm').reset_index(drop=True)

        # Append data to the dataset.
        dataset_output['PtID'].extend([int(patient_id)] * len(total_data))
        dataset_output['DataDtTm'].extend(total_data['DataDtTm'])
        dataset_output['Insulin'].extend(total_data['basal_dose'] +
                                         total_data['bolus_dose'])
        dataset_output['CGM'].extend(total_data['value'])

    dataset = pd.DataFrame(dataset_output)
    return dataset
