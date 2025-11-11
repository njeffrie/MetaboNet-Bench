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
    df = pd.read_csv(cgm_data_file, sep='|')
    df['PtID'] = df['PtID'].astype(int)
    df['CGM'] = pd.to_numeric(df['CGMValue'], errors='coerce')
    df['DataDtTm'] = pd.to_datetime(df['DataDtTm'], format='%Y-%m-%d %H:%M:%S', errors='coerce').dt.floor('5min')

    dataset_output = {'PtID': [], 'DataDtTm': [], 'CGM': [], 'Insulin': []}
    # Load insulin data if available
    basal_info = pd.read_csv(basal_data_file, sep='|', encoding='utf-16')
    basal_changes = pd.read_csv(basal_change_data_file, sep='|')
    bolus_info = pd.read_csv(bolus_data_file, sep='|')
    
    # Align basal datetime information to the day.
    assert 'InsTherapyDt' in basal_info.columns
    basal_info['InsTherapyDt'] = pd.to_datetime(basal_info['InsTherapyDt'], errors='coerce')
    basal_info = basal_info.sort_values('InsTherapyDt').reset_index(drop=True)
    basal_info['InsTherapyDt'] = basal_info['InsTherapyDt'].dt.floor('D')

    # Align basal change datetime information to the 5 minute interval.
    assert 'DataDtTm' in basal_changes.columns
    basal_changes['DataDtTm'] = pd.to_datetime(basal_changes['DataDtTm'], errors='coerce')
    basal_changes = basal_changes.sort_values('DataDtTm').reset_index(drop=True)
    basal_changes['DataDtTm'] = basal_changes['DataDtTm'].dt.floor('5min')

    # Align bolus datetime information to the 5 minute interval.
    assert 'DataDtTm' in bolus_info.columns
    bolus_info['DataDtTm'] = pd.to_datetime(bolus_info['DataDtTm'], errors='coerce')
    bolus_info = bolus_info.sort_values('DataDtTm').reset_index(drop=True)
    bolus_info['DataDtTm'] = bolus_info['DataDtTm'].dt.floor('5min')

    # Iterate only over patients present in CGM and all insulin-related datasets
    cgm_ids = set(df['PtID'].unique())
    basal_ids = set(basal_info['PtID'].unique())
    basal_change_ids = set(basal_changes['PtID'].unique())
    bolus_ids = set(bolus_info['PtID'].unique())
    common_patient_ids = sorted(list(cgm_ids & basal_ids & basal_change_ids & bolus_ids))
    for patient_id in tqdm(common_patient_ids):
        patient_basal_info = basal_info[basal_info['PtID'] == patient_id]
        patient_basal_changes = basal_changes[basal_changes['PtID'] == patient_id]
        patient_bolus_info = bolus_info[bolus_info['PtID'] == patient_id]
        patient_basal_info = patient_basal_info[patient_basal_info['InsBasal0000'] != '']
        starting_date = max(df['DataDtTm'].min(), patient_basal_info['InsTherapyDt'].min())
        patient_cgm = df[(df['PtID'] == patient_id) & (df['DataDtTm'] >= starting_date)]
        
        def get_basal_rate(basal_info, dt, current_basal=None):
            if dt not in patient_basal_info['InsTherapyDt'].values:
                if current_basal is not None:
                    return current_basal
                else:
                    dt = basal_info['InsTherapyDt'].min()

            # Get last pump profile before the given date.
            basal_info = patient_basal_info[patient_basal_info['InsTherapyDt'] == dt]
            key_list = [f'InsBasal{dt.hour:02d}{dt.minute:02d}' for dt in pd.date_range(dt.date(), dt.date() + pd.Timedelta(hours=24), freq='30min')]
            basal_rates = [basal_info[key].values[0] for key in key_list]
            
            # Fill in rates across the day.
            #print(basal_rates)
            current_rate = basal_rates[0]
            #print(current_rate)
            for i in range(len(basal_rates)):
                if basal_rates[i] is None or np.isnan(basal_rates[i]):
                    basal_rates[i] = current_rate
                else:
                    current_rate = basal_rates[i]

            rates = {'DataDtTm': [], 'Insulin': []}
            for i, dt in enumerate(pd.date_range(dt.date(), dt.date() + pd.Timedelta(hours=24), freq='5min')):
                current_rate = basal_rates[i // 6]
                rates['DataDtTm'].append(dt.time())
                rates['Insulin'].append(current_rate)
            return pd.DataFrame(rates)
        
        basal_rates = get_basal_rate(patient_basal_info, starting_date)
        if basal_rates is None or basal_rates.iloc[0]['Insulin'] is None or np.isnan(basal_rates.iloc[0]['Insulin']):
            print(f'No basal rates found for patient {patient_id}')
            continue
        for idx, sample in tqdm(patient_cgm.iterrows(), total=len(patient_cgm)):
            dt = sample['DataDtTm']
            cgm = sample['CGM']
            basal_rates = get_basal_rate(patient_basal_info, dt, basal_rates)
            insulin_delivered = basal_rates[basal_rates['DataDtTm'] == dt.time()]['Insulin'].values[0] / 12.0
            #print(f'base insulin {insulin_delivered}')
            if dt in patient_basal_changes['DataDtTm'].unique():
                insulin_delivered = patient_basal_changes[patient_basal_changes['DataDtTm'] == dt]['CommandedBasalRate'].values[0] / 12.0
                #print(f'after basal change {insulin_delivered}')
            if dt in patient_bolus_info['DataDtTm'].unique():
                bolus_delivered = patient_bolus_info[patient_bolus_info['DataDtTm'] == dt]['BolusAmount'].values[0]
                insulin_delivered += bolus_delivered
                #print(f'after bolus {insulin_delivered}')

            dataset_output['PtID'].append(patient_id)
            dataset_output['DataDtTm'].append(dt)
            dataset_output['CGM'].append(cgm)
            dataset_output['Insulin'].append(insulin_delivered)

    dataset = pd.DataFrame(dataset_output)
    
    print(f"Successfully loaded {len(df)} CGM records")
    return dataset