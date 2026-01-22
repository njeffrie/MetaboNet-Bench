import pandas as pd
import numpy as np
import os

def preprocess(ds_path: str):
    """
    Preprocess the MetaBonet dataset from parquet file.
    
    Args:
        ds_path: Path to the dataset (should contain metabonet_public_2025.parquet)
    
    Returns:
        DataFrame with columns: PtID, DataDtTm, CGM, Insulin
    """
    df = pd.read_parquet(ds_path, columns = ['CGM', 'insulin', 'carbs', 'id', 'date', 'source_file', 'insulin_delivery_device'])
    df = df.rename(columns = {'id': 'PtID', 'date': 'DataDtTm', 'insulin': 'Insulin', 'carbs': 'Carbs', 'source_file': 'DatasetName'})
    df = df[df['DataDtTm'].notna()]
    df = df[df['PtID'].notna()]
    df['Insulin'] = df['Insulin'].fillna(method='ffill', limit = 12)
    df['Carbs'] = df['Carbs'].fillna(method='ffill', limit = 12*6)
    print(f'Before insulin: {len(df)}')
    df = df[df['Insulin'].notna()]  
    print(f'After insulin: {len(df)}')
    df = df[df['Carbs'].notna()]
    print(f'After carbs: {len(df)}')

    df['DataDtTm'] = pd.to_datetime(df['DataDtTm'])
    df['DataDtTm'] = df['DataDtTm'].dt.floor('5min')
    df['CGM'] = pd.to_numeric(df['CGM'], errors='coerce')
    df['Insulin'] = pd.to_numeric(df['Insulin'], errors='coerce')
    df['Carbs'] = pd.to_numeric(df['Carbs'], errors='coerce')
    df = df[df['insulin_delivery_device'] != 'Multiple Daily Injections']
    print(f'After insulin delivery device: {len(df)}')
    df['Carbs'] = df['Carbs'].clip(lower=0.0, upper=200.0)

    df = df.sort_values(['DatasetName', 'PtID', 'DataDtTm'])
    df = df.reset_index(drop=True)
    return df[['PtID', 'DataDtTm', 'CGM', 'Insulin', 'Carbs','DatasetName']]