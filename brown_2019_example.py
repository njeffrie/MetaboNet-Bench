#!/usr/bin/env python3
"""
Example usage of the consecutive CGM sequences functionality.

This script demonstrates how to:
1. Load CGM data
2. Extract consecutive CGM sequences for a specific patient
3. Analyze the sequences
"""

import logging
from brown_2019 import CGMDataSet

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """Main function demonstrating consecutive CGM sequences."""
    
    print("=== Consecutive CGM Sequences Example ===\n")
    
    # Initialize and load the dataset
    print("1. Loading CGM data...")
    cgm_dataset = CGMDataSet()
    cgm_dataset.load_data(chunk_size=None)
    
    # Get patient data summary
    print(cgm_dataset.get_summary_stats())
    # Extract and analyze sequences
    sequences = []
    total_readings = 0
    
    for i, sequence in enumerate(cgm_dataset.yield_consecutive_cgm_sequences()):
        sequences.append(sequence)
        total_readings += len(sequence)
        
        # Print details for first few sequences
        if i < 5:
            df = sequence.to_pandas()
            start_time = df['DataDtTm'].min()
            end_time = df['DataDtTm'].max()
            duration_minutes = (end_time - start_time).total_seconds() / 60
            cgm_range = f"{df['CGM'].min():.0f}-{df['CGM'].max():.0f}"
            
            print(f"  Sequence {i+1}:")
            print(f"    Length: {len(sequence)} readings")
            print(f"    Duration: {duration_minutes:.1f} minutes")
            print(f"    Time range: {start_time} to {end_time}")
            print(f"    CGM range: {cgm_range}")
            print(f"    Period: {df['Period'].iloc[0]}")
            print()
        
        # Limit output for demo
        if i >= 9:
            break

if __name__ == "__main__":
    main() 