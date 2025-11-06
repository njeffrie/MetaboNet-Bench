import os
from tqdm import tqdm
from datasets import Dataset
import pandas as pd
import numpy as np
import zipfile
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
        df = pd.read_csv(self.dataset_info['data_file'], sep=self.dataset_info['sep'])
        
        # Convert data types
        df['PtID'] = df[self.dataset_info['columns']['ID']].astype(int)
        df['CGM'] = pd.to_numeric(df[self.dataset_info['columns']['CGM']], errors='coerce')
        
        # Convert datetime column
        datetime_names = self.dataset_info['columns']['DateTime']
        datetime_format = self.dataset_info['datetime_format']
        if isinstance(datetime_names, list): # TODO: this does not work for all datasets
            date_col = pd.to_datetime(df[datetime_names[0]], format=datetime_format[0])
            time_col = pd.to_timedelta(df[datetime_names[1]])
            
            df['DataDtTm'] = (date_col + time_col).round('5min')
        else: # Handle case where date and time are combined.
            df['DataDtTm'] = pd.to_datetime(df[datetime_names].astype(str), format=datetime_format, errors='coerce')
            df['DataDtTm'] = df['DataDtTm'].dt.floor('5min')
        #insulin_info = pd.DataFrame(columns=['PtID', 'DataDtTm', 'Insulin'])
        dataset_output = {'PtID': [], 'DataDtTm': [], 'CGM': [], 'Insulin': []}
        # Load insulin data if available
        basal_info = pd.read_csv(f"{self.dataset_dir}/Data Files/InsulinPumpSettings_a.txt", sep='|', encoding='utf-16')
        basal_changes = pd.read_csv(f"{self.dataset_dir}/Data Files/Pump_BasalRateChange.txt", sep='|')
        bolus_info = pd.read_csv(f"{self.dataset_dir}/Data Files/Pump_BolusDelivered.txt", sep='|')
        
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

        self.dataset = pd.DataFrame(dataset_output)
        print(self.dataset)
        
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
        
        patient_id_list = []
        date_times = []
        glucose_values = []
        insulin_values = []

        def create_split(patient_ids_split, df):
            for pid in tqdm(patient_ids_split):
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
                        patient_id_list.append(np.array(sequence['PtID'].values[:1], dtype=np.int32))
                        date_times.append(np.array(sequence['DataDtTm'].astype(np.int64).values, dtype=np.int64))
                        glucose_values.append(np.array(sequence['CGM'].values, dtype=np.float32))
                        insulin_values.append(np.array(sequence['Insulin'].values, dtype=np.float32))
                        #print(insulin_values)
            print(f"Found {len(patient_id_list)} sequences")
            
            # Stack all sequences into arrays
            patient_ids_np = np.stack(patient_id_list)
            date_times_np = np.stack(date_times)
            glucose_values_np = np.stack(glucose_values)
            insulin_values_np = np.stack(insulin_values)

            # If the dataset is too large, use 20k samples rather than 10% to speed up the benchmark.
            test_size = min(len(patient_id_list) // 10 , 20000)
            
            # Shuffle with a predetermined seed to produce a reproducible split.
            ds = Dataset.from_dict({
                'PtID': patient_ids_np, 
                'DataDtTm': date_times_np, 
                'CGM': glucose_values_np,
                'Insulin': insulin_values_np
            })
            #ds = ds.train_test_split(test_size=test_size, shuffle=False)
            return ds

        test_size = max(len(patient_ids) // 10 , 1)
        train_size = len(patient_ids) - test_size
        print(f'creating split with {test_size} test samples and {train_size} train samples')
        ds_train = create_split(patient_ids[:train_size], df)
        ds_test = create_split(patient_ids[-test_size:], df)
        ds_train.save_to_disk(f'{self.dataset_dir}/dataset/train')
        ds_test.save_to_disk(f'{self.dataset_dir}/dataset/test')


def extract_zipfile(ds_name):
    os.makedirs(ds_name, exist_ok=True)
    with zipfile.ZipFile(f'{ds_name}.zip', 'r') as zip_ref:
        zip_ref.extractall(ds_name)

if __name__ == "__main__":
    for ds in ["Brown2019"]:#, "Anderson2016", "Lynch2022"]:
        if not os.path.exists(ds):
            extract_zipfile(ds)
        cgm_dataset = DatsetPreprocessor(ds)
        cgm_dataset.preprocess()
