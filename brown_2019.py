import pandas as pd
from datasets import Dataset
from typing import Optional, Dict, Any, Iterator
import logging

class CGMDataSet:
    """
    A class to convert CGM data from text file format to HuggingFace Dataset.
    
    The CGM data file contains pipe-separated values with a header line containing column names.
    Column names are automatically read from the first line of the file.
    """
    
    def __init__(self, file_path: str = "brown_dataset/Data Files/cgm.txt"):
        """
        Initialize the CGMDataSet.
        
        Args:
            file_path (str): Path to the CGM data text file
        """
        self.file_path = file_path
        self.dataset = None
        self.logger = logging.getLogger(__name__)
    
    def load_data(self, chunk_size: Optional[int] = None) -> 'CGMDataSet':
        """
        Load the CGM data from the text file.
        
        Args:
            chunk_size (Optional[int]): If provided, load data in chunks to handle large files
        
        Returns:
            CGMDataSet: Self for method chaining
        """
        try:
            self.logger.info(f"Loading CGM data from {self.file_path}")
            
            columns = ['PtID', 'Period', 'DataDtTm', 'CGM']
            
            if chunk_size:
                # Load data in chunks for large files, skipping the header
                self.logger.info(f"Loading data in chunks of {chunk_size} rows")
                chunks = []
                for chunk in pd.read_csv(self.file_path, 
                                       sep='|', 
                                       names=columns,
                                       skiprows=1,  # Skip the header line
                                       chunksize=chunk_size):
                    chunks.append(chunk)
                df = pd.concat(chunks, ignore_index=True)
            else:
                # Load entire file at once, skipping the header
                df = pd.read_csv(self.file_path, sep='|', names=columns, skiprows=1)
            
            # Convert data types
            df['PtID'] = df['PtID'].astype(int)
            df['CGM'] = pd.to_numeric(df['CGM'], errors='coerce')
            
            # Convert datetime column
            df['DataDtTm'] = pd.to_datetime(df['DataDtTm'], format='%d%b%y:%H:%M:%S', errors='coerce')
            
            # Create HuggingFace Dataset
            self.dataset = Dataset.from_pandas(df)
            
            self.logger.info(f"Successfully loaded {len(df)} CGM records")
            self.logger.info(f"Dataset features: {self.dataset.features}")
            
        except Exception as e:
            self.logger.error(f"Error loading CGM data: {str(e)}")
            raise
        
        return self
    
    def get_dataset(self) -> Dataset:
        """
        Get the HuggingFace Dataset.
        
        Returns:
            Dataset: The loaded HuggingFace Dataset
        """
        if self.dataset is None:
            self.load_data()
        return self.dataset
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics of the dataset.
        
        Returns:
            Dict[str, Any]: Summary statistics including counts, ranges, etc.
        """
        if self.dataset is None:
            self.load_data()
        
        df = self.dataset.to_pandas()
        
        stats = {
            'total_records': len(df),
            'unique_patients': df['PtID'].nunique(),
            'unique_periods': df['Period'].nunique(),
            'periods': df['Period'].value_counts().to_dict(),
            'date_range': {
                'start': df['DataDtTm'].min().isoformat() if not df['DataDtTm'].isna().all() else None,
                'end': df['DataDtTm'].max().isoformat() if not df['DataDtTm'].isna().all() else None
            },
            'cgm_stats': {
                'mean': df['CGM'].mean(),
                'std': df['CGM'].std(),
                'min': df['CGM'].min(),
                'max': df['CGM'].max(),
                'missing_values': df['CGM'].isna().sum()
            }
        }
        
        return stats
    
    def get_patient_ids(self) -> list:
        """
        Get a list of all unique patient IDs in the dataset.
        
        Returns:
            list: List of unique patient IDs sorted in ascending order
        """
        if self.dataset is None:
            self.load_data()
        
        df = self.dataset.to_pandas()
        patient_ids = sorted(df['PtID'].unique().tolist())
        
        self.logger.info(f"Found {len(patient_ids)} unique patients")
        return patient_ids
    
    def filter_by_patient(self, patient_ids: list) -> Dataset:
        """
        Filter the dataset by patient IDs.
        
        Args:
            patient_ids (list): List of patient IDs to include
        
        Returns:
            Dataset: Filtered dataset
        """
        if self.dataset is None:
            self.load_data()
        
        return self.dataset.filter(lambda x: x['PtID'] in patient_ids)
    
    def filter_by_period(self, periods: list) -> Dataset:
        """
        Filter the dataset by study periods.
        
        Args:
            periods (list): List of periods to include
        
        Returns:
            Dataset: Filtered dataset
        """
        if self.dataset is None:
            self.load_data()
        
        return self.dataset.filter(lambda x: x['Period'] in periods)
    
    def filter_by_date_range(self, start_date: str, end_date: str) -> Dataset:
        """
        Filter the dataset by date range.
        
        Args:
            start_date (str): Start date in ISO format
            end_date (str): End date in ISO format
        
        Returns:
            Dataset: Filtered dataset
        """
        if self.dataset is None:
            self.load_data()
        
        return self.dataset.filter(
            lambda x: start_date <= x['DataDtTm'] <= end_date
        )
    
    def yield_consecutive_cgm_sequences(self, patient_id: int = None, min_sequence_length: int = 2, 
                                      max_gap_minutes: int = 6) -> Iterator[Dataset]:
        """
        Yield datasets containing consecutive CGM values for a specific patient.
        
        This method identifies sequences of consecutive CGM readings where the time gap
        between consecutive readings is within the specified threshold.
        
        Args:
            patient_id (int, optional): The patient ID to filter for. If None, processes all patients.
            min_sequence_length (int): Minimum number of consecutive readings to include
            max_gap_minutes (int): Maximum allowed gap between consecutive readings in minutes
        
        Yields:
            Dataset: A dataset containing a sequence of consecutive CGM values
        """
        if self.dataset is None:
            self.load_data()
        
        if patient_id is None:
            # Process all patients
            patient_ids = self.get_patient_ids()
        else:
            patient_ids = [patient_id]
        
        for pid in patient_ids:
            # Filter data for the specific patient and sort by timestamp
            patient_data = self.filter_by_patient([pid])
            df = patient_data.to_pandas()
            
            if len(df) == 0:
                self.logger.warning(f"No data found for patient {pid}")
                continue
            
            # Sort by timestamp
            df = df.sort_values('DataDtTm').reset_index(drop=True)
            
            # Calculate time differences between consecutive readings
            df['time_diff'] = df['DataDtTm'].diff().dt.total_seconds() / 60  # Convert to minutes
            
            # Identify sequence breaks (gaps larger than max_gap_minutes)
            # First row will have NaN time_diff, so we mark it as False (not a break)
            sequence_breaks = df['time_diff'] > max_gap_minutes
            sequence_breaks.iloc[0] = False  # First reading starts a new sequence
            
            # Create sequence IDs
            df['sequence_id'] = sequence_breaks.cumsum()
            
            # Group by sequence and yield each sequence that meets minimum length
            for seq_id, group in df.groupby('sequence_id'):
                if len(group) >= min_sequence_length:
                    # Create a dataset for this sequence
                    sequence_dataset = Dataset.from_pandas(group.drop(['time_diff', 'sequence_id'], axis=1))
                    yield sequence_dataset
    
    def save_dataset(self, output_path: str) -> None:
        """
        Save the dataset to disk.
        
        Args:
            output_path (str): Path where to save the dataset
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")
        
        self.dataset.save_to_disk(output_path)
        self.logger.info(f"Dataset saved to {output_path}")