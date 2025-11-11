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

class DatsetPreprocessor:
    """
    Class to preprocess datasets.

    Converts text files to prepared datasets for training and testing.
    """
    
    def __init__(self, ds_name: str):
        self.dataset_dir = ds_name
        self.dataset = self.load_data(ds_name)
    
    def load_data(self, dataset_name: str):
        if dataset_name == 'Anderson2016':
            return anderson2016.preprocess(os.path.abspath(dataset_name))
        if dataset_name == "Brown2019":
            return brown2019.preprocess(os.path.abspath(dataset_name))

    def interpolate_data(self, max_gap_minutes: int = 30):
        patient_ids = self.dataset['PtID'].unique()
        print(self.dataset)
        interpolated_dataset = {'PtID': [], 'DataDtTm': [], 'CGM': [], 'Insulin': []}

        for patient_id in patient_ids:
            # Get data for the specified patient.
            patient_data = self.dataset[self.dataset['PtID'] == patient_id]
            # Normalize the timestamp to 5 minute intervals.
            patient_data['DataDtTm'] = pd.to_datetime(patient_data['DataDtTm'])
            patient_data['DataDtTm'] = patient_data['DataDtTm'].dt.floor('5min')
            patient_data = patient_data.sort_values('DataDtTm')
            patient_data = patient_data.reset_index(drop=True)
            # Identify sequence breaks beyond the specified threshold.
            patient_data['time_diff'] = patient_data['DataDtTm'].diff().dt.total_seconds() / 60  # Convert to minutes
            sequence_breaks = patient_data['time_diff'] > max_gap_minutes
            sequence_breaks.iloc[0] = False  # First reading starts a new sequence
            patient_data['sequence_id'] = sequence_breaks.cumsum()
            for idx, sequence in patient_data.groupby('sequence_id'):
                sequence_times = pd.date_range(sequence['DataDtTm'].min(), sequence['DataDtTm'].max(), freq='5min')
                sequence_data = sequence.merge(pd.DataFrame({'DataDtTm': sequence_times}), on='DataDtTm', how='left').sort_values('DataDtTm').reset_index(drop=True)
                sequence_data['CGM'].interpolate(method='linear', inplace=True)
                sequence_data['Insulin'].interpolate(method='nearest', limit_direction='forward', inplace=True)
                sequence_data['PtID'].interpolate(method='nearest', limit_direction='forward', inplace=True)

                interpolated_dataset['CGM'].extend(list(sequence_data['CGM']))
                interpolated_dataset['Insulin'].extend(list(sequence_data['Insulin']))
                interpolated_dataset['PtID'].extend(list(sequence_data['PtID']))
                interpolated_dataset['DataDtTm'].extend(list(sequence_data['DataDtTm']))
        self.dataset = pd.DataFrame(interpolated_dataset)

    def save_data(self):
        np.random.seed(42)
        patient_ids = sorted(list(self.dataset['PtID'].unique()))
        np.random.shuffle(patient_ids)
        ds_train = Dataset.from_pandas(self.dataset[self.dataset['PtID'].isin(patient_ids[:int(len(patient_ids) * 0.9)])])
        ds_test = Dataset.from_pandas(self.dataset[self.dataset['PtID'].isin(patient_ids[int(len(patient_ids) * 0.9):])])
        ds_train.save_to_disk(f'{self.dataset_dir}/train')
        ds_test.save_to_disk(f'{self.dataset_dir}/test')

def extract_zipfile(ds_name):
    os.makedirs(ds_name, exist_ok=True)
    data_dir = f'{os.path.dirname(os.path.realpath(__file__))}/downloads'
    dataset_source_path = f'{data_dir}/{dataset_info.dataset_names_map[ds_name]}'
    assert os.path.exists(dataset_source_path), f"Zip file not found: {dataset_source_path}"
    with zipfile.ZipFile(dataset_source_path, 'r') as zip_src:
        zip_src.extractall(ds_name)

@click.command()
@click.option('--force', is_flag=True, help='Force re-download and preprocess the dataset')
def main(force: bool = False):
    for ds in ["Anderson2016", "Brown2019"]:
        if not os.path.exists(ds) or force:
            extract_zipfile(ds)
        cgm_dataset = DatsetPreprocessor(ds)
        cgm_dataset.interpolate_data()
        cgm_dataset.save_data()
        print(f"Preprocessed {ds} dataset")

if __name__ == "__main__":
    main()