import os
from tqdm import tqdm
from datasets import Dataset
import pandas as pd
import numpy as np
import zipfile
import click
import dataset_info
from dataset_processors import brown2019
from dataset_processors import anderson2016
from dataset_processors import azt1d
from dataset_processors import manchester2024
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

        for patient_id in tqdm(patient_ids):
            # Get data for the specified patient.
            patient_data = self.dataset[self.dataset['PtID'] == patient_id]
            start_time = patient_data['DataDtTm'].min()
            end_time = patient_data['DataDtTm'].max()
            sequence_times = pd.date_range(start_time, end_time, freq='5min')
            patient_data = patient_data.merge(
                pd.DataFrame({'DataDtTm': sequence_times}),
                on='DataDtTm',
                how='outer').sort_values('DataDtTm').reset_index(drop=True)
            patient_data['Insulin'] = patient_data['Insulin'].fillna(value=0.0)
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
            patient_data['SequenceID'] = starts.cumsum() - 1
            patient_data = patient_data[patient_data['CGM'].notna()]

            ds = pd.concat([
                ds, patient_data[[
                    'PtID', 'DataDtTm', 'CGM', 'Insulin', 'SequenceID', 'DatasetName'
                ]]
            ])
        self.dataset = ds

    def save_data(self):
        self.dataset.to_parquet(self.dataset_path)


def extract_zipfile(ds_name):
    os.makedirs(ds_name, exist_ok=True)
    data_dir = f'{os.path.dirname(os.path.realpath(__file__))}/downloads'
    dataset_source_path = f'{data_dir}/{dataset_info.dataset_names_map[ds_name]}'
    assert os.path.exists(
        dataset_source_path), f"Zip file not found: {dataset_source_path}"
    with zipfile.ZipFile(dataset_source_path, 'r') as zip_src:
        zip_src.extractall(ds_name)


@click.command()
@click.option('--force',
              is_flag=True,
              help='Force re-download and preprocess the dataset')
@click.option('--path_to_dataset',
              type=str,
              default='metabonet_public_2025.parquet',
              help='Path to the dataset to preprocess')
def main(force: bool = False, path_to_dataset: str = 'metabonet_public_2025.parquet'):
    cgm_dataset = DatsetPreprocessor(path_to_dataset)
    cgm_dataset.interpolate_data()
    cgm_dataset.save_data()
    print(f"Preprocessed {path_to_dataset} dataset")


if __name__ == "__main__":
    main()
