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
    df = pd.read_parquet(ds_path, columns = ['CGM', 'insulin', 'carbs','id', 'date', 'source_file'])
    df = df.rename(columns = {'id': 'PtID', 'date': 'DataDtTm', 'insulin': 'Insulin', 'carbs': 'Carbs', 'source_file': 'DatasetName'})
    df = df[df['DataDtTm'].notna()]

    df['DataDtTm'] = pd.to_datetime(df['DataDtTm'])
    df['DataDtTm'] = df['DataDtTm'].dt.floor('5min')
    df['CGM'] = pd.to_numeric(df['CGM'], errors='coerce')
    df['Insulin'] = pd.to_numeric(df['Insulin'], errors='coerce')
    df['PtID'] = pd.to_numeric(df['PtID'], errors='coerce')
    df['Carbs'] = pd.to_numeric(df['Carbs'], errors='coerce')
    df = df[df['PtID'].notna()]

    # Combine source dataset and user ID to generate truly unique patient IDs.
    df = df.sort_values(['DatasetName', 'PtID']).reset_index(drop=True)
    df['PtID']= df['PtID'].diff().ne(0).cumsum()

    df = df.sort_values(['PtID', 'DataDtTm'])
    df = df.reset_index(drop=True)
    return df[['PtID', 'DataDtTm', 'CGM', 'Insulin', 'Carbs','DatasetName']]