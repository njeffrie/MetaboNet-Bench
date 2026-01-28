import pandas as pd
import numpy as np
import os

# Fill nan values with zero only if there is a valid prior value within the horizon.
def fill_nan_with_zero_within_horizon(df,
                                      time_col,
                                      target_col,
                                      horizon):
    t = pd.to_datetime(df[time_col])
    h = pd.Timedelta(horizon)
    s = df[target_col]

    last = t.where(s.notna()).ffill()
    out = df.copy()
    out.loc[s.isna() & last.notna() & (t - last <= h), target_col] = 0
    return out

def preprocess(ds_path: str):
    """
    Preprocess the MetaBonet dataset from parquet file.
    
    Args:
        ds_path: Path to the dataset (should contain metabonet_public_2025.parquet)
    
    Returns:
        DataFrame with columns: PtID, DataDtTm, CGM, Insulin
    """
    df = pd.read_parquet(ds_path, columns = ['CGM', 'insulin', 'carbs', 'id', 'date', 'source_file', 'insulin_delivery_device', 'insulin_delivery_modality'])
    df = df.rename(columns = {'id': 'PtID', 'date': 'DataDtTm', 'insulin': 'Insulin', 'carbs': 'Carbs', 'source_file': 'DatasetName'})
    df = df[df['DataDtTm'].notna()]
    df = df[df['PtID'].notna()]

    df = fill_nan_with_zero_within_horizon(df, time_col='DataDtTm', target_col='Insulin', horizon='1h')
    df = fill_nan_with_zero_within_horizon(df, time_col='DataDtTm', target_col='Carbs', horizon='6h')
    df = df[df['Insulin'].notna()]  
    df = df[df['Carbs'].notna()]

    df['DataDtTm'] = pd.to_datetime(df['DataDtTm'])
    df['DataDtTm'] = df['DataDtTm'].dt.floor('5min')
    df['CGM'] = pd.to_numeric(df['CGM'], errors='coerce')
    df['Insulin'] = pd.to_numeric(df['Insulin'], errors='coerce')
    df['Carbs'] = pd.to_numeric(df['Carbs'], errors='coerce')
    mdi_filter = df['insulin_delivery_device'] != 'Multiple Daily Injections'
    mdi_filter = mdi_filter | (df['insulin_delivery_modality'] != 'MDI')
    df = df[mdi_filter]
    df['Carbs'] = df['Carbs'].clip(lower=0.0, upper=200.0)

    df = df.sort_values(['DatasetName', 'PtID', 'DataDtTm'])
    df = df.reset_index(drop=True)
    return df[['PtID', 'DataDtTm', 'CGM', 'Insulin', 'Carbs', 'DatasetName']]