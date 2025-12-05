import pandas as pd
import numpy as np
from tqdm import tqdm


# Preprocess the Brown 2019 dataset.
def preprocess(ds_dir):
    cgm_data_file = f"{ds_dir}/Data Files/Pump_CGMGlucoseValue.txt"
    basal_data_file = f"{ds_dir}/Data Files/InsulinPumpSettings_a.txt"
    basal_change_data_file = f"{ds_dir}/Data Files/Pump_BasalRateChange.txt"
    bolus_data_file = f"{ds_dir}/Data Files/Pump_BolusDelivered.txt"

    # Load entire file at once, skipping the header
    df = pd.read_csv(cgm_data_file, sep='|', dtype={'DataDtTm_adjusted': str})
    df['PtID'] = df['PtID'].astype(int)
    df['CGM'] = pd.to_numeric(df['CGMValue'], errors='coerce')
    df['DataDtTm'] = pd.to_datetime(df['DataDtTm'],
                                    format='%Y-%m-%d %H:%M:%S',
                                    errors='coerce').dt.floor('5min')

    basal_info = pd.read_csv(basal_data_file, sep='|', encoding='utf-16')
    basal_changes = pd.read_csv(basal_change_data_file,
                                sep='|',
                                dtype={'DataDtTm_adjusted': str})
    bolus_info = pd.read_csv(bolus_data_file, sep='|')

    # Align basal datetime information to the day.
    basal_info['InsTherapyDt'] = pd.to_datetime(basal_info['InsTherapyDt'])
    basal_info = basal_info.sort_values('InsTherapyDt').reset_index(drop=True)
    basal_info['InsTherapyDt'] = basal_info['InsTherapyDt'].dt.floor('D')
    basal_info.rename(columns={'InsTherapyDt': 'DataDtTm'}, inplace=True)

    # Align basal change datetime information to the 5 minute interval.
    basal_changes['DataDtTm'] = pd.to_datetime(basal_changes['DataDtTm'])
    basal_changes = basal_changes.sort_values('DataDtTm').reset_index(drop=True)
    basal_changes['DataDtTm'] = basal_changes['DataDtTm'].dt.floor('5min')

    # Align bolus datetime information to the 5 minute interval.
    bolus_info['DataDtTm'] = pd.to_datetime(bolus_info['DataDtTm'])
    bolus_info = bolus_info.sort_values('DataDtTm').reset_index(drop=True)
    bolus_info['DataDtTm'] = bolus_info['DataDtTm'].dt.floor('5min')

    dataset_output = pd.DataFrame()
    # Iterate only over patients present in CGM and all insulin-related datasets
    cgm_ids = set(df['PtID'].unique())
    basal_ids = set(basal_info['PtID'].unique())
    basal_change_ids = set(basal_changes['PtID'].unique())
    bolus_ids = set(bolus_info['PtID'].unique())
    common_patient_ids = list(cgm_ids & basal_ids & basal_change_ids &
                              bolus_ids)
    for patient_id in tqdm(common_patient_ids):
        patient_basal_info = basal_info[basal_info['PtID'] == patient_id]
        patient_basal_changes = basal_changes[basal_changes['PtID'] ==
                                              patient_id]
        patient_bolus_info = bolus_info[bolus_info['PtID'] == patient_id]
        patient_cgm = df[(df['PtID'] == patient_id)]
        patient_basal_info = patient_basal_info[
            patient_basal_info['InsBasal0000'] != '']

        # Limit the dataset to times where both basal and bolus data are available.
        start_date = max(patient_bolus_info['DataDtTm'].min(),
                         patient_basal_info['DataDtTm'].min())
        end_date = patient_bolus_info['DataDtTm'].max()

        # Fill in all 5 minute intervals during the study period.
        basal_times = pd.date_range(patient_basal_info['DataDtTm'].min(),
                                    end_date,
                                    freq='5min')
        insulin_data = pd.DataFrame()
        current_basal = None
        for day in pd.date_range(basal_times.min().date(),
                                 basal_times.max().date(),
                                 freq='D'):
            five_minute_increments = pd.date_range(day,
                                                   day + pd.Timedelta(days=1),
                                                   freq='5min')
            if day in patient_basal_info['DataDtTm'].values:
                basal_day = patient_basal_info[patient_basal_info['DataDtTm'] ==
                                               day]
                ha_increments = pd.date_range(day,
                                              day + pd.Timedelta(days=1),
                                              freq='30min')
                key_list = [
                    f'InsBasal{dt.hour:02d}{dt.minute:02d}'
                    for dt in ha_increments
                ]
                current_basal = pd.DataFrame({
                    'DataDtTm': ha_increments,
                    'Insulin': [basal_day[key].values[0] for key in key_list]
                })
                current_basal = current_basal.merge(pd.DataFrame(
                    {'DataDtTm': five_minute_increments}),
                                                    on='DataDtTm',
                                                    how='outer')
                current_basal.ffill(inplace=True)
            current_basal['DataDtTm'] = five_minute_increments
            insulin_data = pd.concat([insulin_data, current_basal])

        patient_basal_changes = patient_basal_changes.drop_duplicates(
            subset=['DataDtTm'])
        patient_basal_changes.loc[:,
                                  'CommandedBasalRate'] = patient_basal_changes[
                                      'CommandedBasalRate'].values / 12.0
        insulin_data.loc[:, 'Insulin'] = insulin_data['Insulin'].values / 12.0
        insulin_data = insulin_data.sort_values('DataDtTm').reset_index(
            drop=True)

        # Change basal rate from default when present in basal change data.
        basal_changes_mask = insulin_data['DataDtTm'].isin(
            patient_basal_changes['DataDtTm'])
        patient_basal_changes = patient_basal_changes.merge(
            insulin_data[['DataDtTm']], on='DataDtTm', how='right')
        insulin_data.loc[
            basal_changes_mask,
            'Insulin'] = patient_basal_changes['CommandedBasalRate']

        # Add bolus amount to insulin data.
        bolus_mask = insulin_data['DataDtTm'].isin(
            patient_bolus_info['DataDtTm'])
        patient_bolus_info = patient_bolus_info.merge(insulin_data[['DataDtTm'
                                                                   ]],
                                                      on='DataDtTm',
                                                      how='right')
        insulin_data.loc[bolus_mask, 'Insulin'] = insulin_data[
            'Insulin'] + patient_bolus_info['BolusAmount']

        dataset_patient = pd.merge(patient_cgm,
                                   insulin_data,
                                   on='DataDtTm',
                                   how='outer')
        dataset_patient = dataset_patient[dataset_patient['DataDtTm'].between(
            start_date, end_date)]
        dataset_output = pd.concat([
            dataset_output,
            dataset_patient[['PtID', 'DataDtTm', 'CGM', 'Insulin']]
        ])
        dataset_output['PtID'] = dataset_output['PtID'].fillna(value=patient_id)
    return dataset_output
