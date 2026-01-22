import os
from tqdm import tqdm
from datasets import Dataset
import pandas as pd
import numpy as np
import zipfile
import click
import dataset_info
from dataset_processors import metabonet


class DatsetPreprocessor:
    """
    Class to preprocess datasets.

    Converts text files to prepared datasets for training and testing.
    """

    def __init__(self, ds_path: str):
        self.dataset_path = ds_path
        self.dataset = self.load_data()

    def load_data(self):
        return metabonet.preprocess(self.dataset_path)

    def interpolate_data(self, max_gap_minutes: int = 30):
        patient_ids = self.dataset['PtID'].unique()
        ds = pd.DataFrame()

        sequence_id = 0
        for ds_name, ds_data in self.dataset.groupby('DatasetName'):
            for patient_id in tqdm(ds_data['PtID'].unique()):
                # Get data for the specified patient.
                patient_data = ds_data[ds_data['PtID'] == patient_id]
                start_time = patient_data['DataDtTm'].min()
                end_time = patient_data['DataDtTm'].max()
                sequence_times = pd.date_range(start_time, end_time, freq='5min')
                patient_data = patient_data.merge(
                    pd.DataFrame({'DataDtTm': sequence_times}),
                    on='DataDtTm',
                    how='outer').sort_values('DataDtTm').reset_index(drop=True)
                patient_data['Insulin'] = patient_data['Insulin'].fillna(value=0.0)
                patient_data['Carbs'] = patient_data['Carbs'].fillna(value=0.0)
                patient_data['PtID'] = patient_data['PtID'].ffill()

                # Interpolate CGM but only keep sub-30 minute gaps bookended by valid data.
                patient_data['CGM'] = patient_data['CGM'].interpolate(
                    method='linear',
                    limit_direction='forward',
                    limit=6,
                    limit_area='inside')
                mask = patient_data['CGM'].isna()
                mask = mask.where(mask, other=np.nan).infer_objects(copy=False)
                mask = mask.bfill(limit=6, limit_area='outside')
                mask = mask.isna()
                patient_data['CGM'] = patient_data['CGM'].where(mask, other=np.nan)

                # Find sequences of non-missing CGM data and number them.
                is_not_nan = patient_data['CGM'].notna()
                starts = is_not_nan & (~is_not_nan.shift(1, fill_value=False))
                patient_data['SequenceID'] = starts.cumsum() - 1 + sequence_id

                sequence_id = patient_data['SequenceID'].max()
                patient_data = patient_data[patient_data['CGM'].notna()]

                ds = pd.concat([
                    ds, patient_data[[
                        'PtID', 'DataDtTm', 'CGM', 'Insulin', 'Carbs', 'SequenceID', 'DatasetName'
                    ]]
                ])
        # Filter out sequences shorter than 192 samples
        seq_counts = ds.groupby('SequenceID').size()
        valid_sequences = seq_counts[seq_counts >= 192].index
        self.dataset = ds[ds['SequenceID'].isin(valid_sequences)]


    def save_data(self):
        split = 'test' if 'test' in self.dataset_path else 'train'
        self.dataset.to_parquet(
        f'metabonet_{split}.parquet',
        engine="pyarrow",
        compression="zstd",
        index=False)


@click.command()
@click.option('--force',
              is_flag=True,
              help='Force re-download and preprocess the dataset')
@click.option('--path_to_dataset',
              type=str,
              default='metabonet_public_test.parquet',
              help='Path to the dataset to preprocess')
def main(force: bool = False, path_to_dataset: str = 'metabonet_public_test.parquet'):
    cgm_dataset = DatsetPreprocessor(path_to_dataset)
    cgm_dataset.interpolate_data()
    cgm_dataset.save_data()
    print(f"Preprocessed {path_to_dataset} dataset")


if __name__ == "__main__":
    main()
