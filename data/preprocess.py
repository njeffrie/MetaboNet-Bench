import os
from tqdm import tqdm
from datasets import Dataset
import pandas as pd
import numpy as np

import dataset_info

class DatsetPreprocessor:
    """
    A class to convert CGM data from text file format to HuggingFace Dataset.
    
    The CGM data file contains pipe-separated values with a header line containing column names.
    Column names are automatically read from the first line of the file.
    """
    
    def __init__(self, ds_name: str):
        self.dataset_info = dataset_info.get(ds_name)
        self.dataset_dir = ds_name
        self.load_data()
    
    def load_data(self):
        """
        Load the CGM data from the text file.
        
        Args:
            chunk_size (Optional[int]): If provided, load data in chunks to handle large files
        
        Returns:
            CGMDataSet: Self for method chaining
        """

        print(f"Loading CGM data from {self.dataset_info['data_file']}")
        
        # Load entire file at once, skipping the header
        df = pd.read_csv(self.dataset_info['data_file'], sep=self.dataset_info['sep'], skiprows=0)#self.dataset_info['skiprows'])
        
        # Convert data types
        df['PtID'] = df[self.dataset_info['columns']['ID']].astype(int)
        df['CGM'] = pd.to_numeric(df[self.dataset_info['columns']['CGM']], errors='coerce')
        
        # Convert datetime column
        datetime_names = self.dataset_info['columns']['DateTime']
        datetime_format = self.dataset_info['datetime_format']
        if isinstance(datetime_names, list): # TODO: this does not work for all datasets
            date_col = pd.to_datetime(df[datetime_names[0]], format=datetime_format[0])
            time_col = pd.to_timedelta(df[datetime_names[1]])
            
            df['DataDtTm'] = date_col + time_col
        else: # Handle case where date and time are combined.
            df['DataDtTm'] = pd.to_datetime(df[datetime_names].astype(str).str.strip(), format=datetime_format, errors='coerce')
        
        self.dataset = df[['PtID', 'DataDtTm', 'CGM']]
        
        print(f"Successfully loaded {len(df)} CGM records")
    
    def preprocess(self, min_sequence_length: int = 192, 
                                      max_gap_minutes: int = 6):
        """
        This method identifies sequences of consecutive CGM readings where the time gap
        between consecutive readings is within the specified threshold.
        
        Args:
            min_sequence_length (int): Minimum number of consecutive readings to include
            max_gap_minutes (int): Maximum allowed gap between consecutive readings in minutes
        """
        # Get all data and sort by patient ID and timestamp at once
        df = self.dataset.sort_values(['PtID', 'DataDtTm']).reset_index(drop=True)
        
        patient_ids = sorted(df['PtID'].unique().tolist())
        
        sequences = []

        for pid in tqdm(patient_ids):
            # Filter data for the specific patient (already sorted by timestamp)
            patient_df = df[df['PtID'] == pid].copy()
            
            if len(patient_df) == 0:
                continue
            
            # Calculate time differences between consecutive readings
            patient_df['time_diff'] = patient_df['DataDtTm'].diff().dt.total_seconds() / 60  # Convert to minutes
            
            # Identify sequence breaks (gaps larger than max_gap_minutes)
            # First row will have NaN time_diff, so we mark it as False (not a break)
            sequence_breaks = patient_df['time_diff'] > max_gap_minutes
            sequence_breaks.iloc[0] = False  # First reading starts a new sequence
            
            # Create sequence IDs
            patient_df['sequence_id'] = sequence_breaks.cumsum()
            
            # Group by sequence and yield each sequence that meets minimum length
            for _, group in patient_df.groupby('sequence_id'):
                for i in range(0, len(group) - min_sequence_length + 1, 12): # increment by one hour.
                    # Create a dataset for this sequence
                    sequence = group.drop(['time_diff', 'sequence_id'], axis=1)[i:i+min_sequence_length]
                    sequence_array = np.array([
                        sequence['PtID'].values,
                        sequence['DataDtTm'].astype(np.int64).values,  # Convert datetime to unix timestamp
                        sequence['CGM'].values
                    ])
                    sequences.append(sequence_array)
        print(f"Found {len(sequences)} sequences")
        
        # Stack all sequences into a single array of shape (-1, 3, 192)
        all_sequences = np.stack(sequences)
        del sequences

        # If the dataset is too large, use 20k samples rather than 10% to speed up the benchmark.
        test_size = min(len(all_sequences) // 10 , 20000)
        
        # Convert to datasets
        def create_dataset(array):
            return Dataset.from_dict({
                'sequences': array.tolist()
            })
        
        ds_all = create_dataset(all_sequences)

        # Use a fixed seed to ensure reproducibility and avoid test/train overlap.
        ds_all.shuffle(seed=42)

        ds_all.select(range(test_size)).to_parquet(f'{self.dataset_dir}/dataset_test.parquet')
        ds_all.select(range(test_size, len(ds_all))).to_parquet(f'{self.dataset_dir}/dataset_train.parquet')


if __name__ == "__main__":
    for dir in os.listdir("."):
        if os.path.isdir(dir) and dir in ["Anderson2016", "Brown2019", "Lynch2022"]:
            cgm_dataset = DatsetPreprocessor(dir)
            cgm_dataset.preprocess()
