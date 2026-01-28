from models.models import get_model
import numpy as np
import pandas as pd
from tqdm import tqdm
import click
import os
import re

def load_dataset(subset_size=None):
    """Load dataset from parquet file using pandas."""
    df = pd.read_parquet('data/metabonet_test.parquet')
    if subset_size is not None:
        df = df.head(subset_size)
    return df

def run_batch(model_runner, input_batch):
    """Run model prediction on a batch of inputs."""
    input_array = np.stack(input_batch, axis=0)
    ts, cgm, insulin, carbs = np.split(input_array, 4, axis=1)
    preds = model_runner.predict(
        ts.squeeze(1), 
        cgm.squeeze(1), 
        insulin.squeeze(1), 
        carbs.squeeze(1)
    )
    return preds

def extract_patient_id(patient_id):
    """Extract numeric patient ID from string."""
    return int(re.findall(r'\d+', str(patient_id))[-1])

def run_benchmark(model, df, batch_size=1, device='cpu'):
    """Run benchmark on a model and save results to parquet."""
    model_runner = get_model(model.lower(), device=device)
    min_sequence_length = 192
    pred_len = 12
    step_size = 12  # 1 hour increments
    
    # Collect all results
    all_results = []
    
    for ds_name, dataset_group in df.groupby('DatasetName'):
        print(f'Processing dataset: {ds_name} ({len(dataset_group)} rows)')
        
        input_batch = []
        batch_metadata = []
        batch_labels = []
        batch_pred_timestamps = []
        
        for patient_id, patient_data in tqdm(dataset_group.groupby('PtID'), 
                                            desc=f'{ds_name} patients'):
            patient_id_num = extract_patient_id(patient_id)
            
            for seq_id, sequence_data in patient_data.groupby('SequenceID'):
                seq_len = len(sequence_data)
                if seq_len < min_sequence_length:
                    continue
                
                # Convert to numpy arrays once
                cgm_values = sequence_data['CGM'].values
                timestamps = sequence_data['DataDtTm'].values.astype(np.int64)
                insulin_values = sequence_data['Insulin'].values
                carbs_values = sequence_data['Carbs'].values
                
                # Generate sliding windows
                for i in range(0, seq_len - min_sequence_length + 1, step_size):
                    end_idx = i + min_sequence_length
                    cgm_window = cgm_values[i:end_idx]
                    
                    # Model input: last 180 timesteps (192 - 12)
                    model_input_cgm = cgm_window[-192:-12]
                    label = cgm_window[-12:]
                    
                    # Corresponding timestamps, insulin, carbs (180 timesteps)
                    ts_window = timestamps[i:i+180]
                    insulin_window = insulin_values[i:i+180]
                    carbs_window = carbs_values[i:i+180]
                    
                    # Prediction timestamps: last 12 timesteps (corresponding to label)
                    pred_timestamps = timestamps[i+180:i+min_sequence_length]
                    
                    # Store batch item
                    input_batch.append(np.stack([
                        ts_window, 
                        model_input_cgm, 
                        insulin_window, 
                        carbs_window
                    ], axis=0))
                    batch_metadata.append({
                        'dataset': ds_name,
                        'patient_id': patient_id_num
                    })
                    batch_labels.append(label)
                    batch_pred_timestamps.append(pred_timestamps)
                    
                    # Process batch when full
                    if len(input_batch) == batch_size:
                        preds = run_batch(model_runner, input_batch)
                        preds = np.clip(preds, 40, 600)
                        
                        # Store results
                        for pred, metadata, label, pred_ts in zip(preds, batch_metadata, batch_labels, batch_pred_timestamps):
                            for step in range(pred_len):
                                all_results.append({
                                    'model': model,
                                    'dataset': metadata['dataset'],
                                    'patient_id': metadata['patient_id'],
                                    'timestamp': pred_ts[step],
                                    'prediction': pred[step],
                                    'label': label[step]
                                })
                        
                        input_batch = []
                        batch_metadata = []
                        batch_labels = []
                        batch_pred_timestamps = []
        
        # Process remaining batch
        if len(input_batch) > 0:
            preds = run_batch(model_runner, input_batch)
            preds = np.clip(preds, 40, 600)
            
            for pred, metadata, label, pred_ts in zip(preds, batch_metadata, batch_labels, batch_pred_timestamps):
                for step in range(pred_len):
                    all_results.append({
                        'model': model,
                        'dataset': metadata['dataset'],
                        'patient_id': metadata['patient_id'],
                        'timestamp': pred_ts[step],
                        'prediction': pred[step],
                        'label': label[step]
                    })
    
    # Convert to DataFrame and save
    results_df = pd.DataFrame(all_results)
    
    # Calculate and print metrics
    if len(results_df) > 0:
        # Overall metrics
        overall_rmse = np.sqrt(np.mean((results_df['label'] - results_df['prediction'])**2))
        overall_ape = np.mean(np.abs(results_df['label'] - results_df['prediction']) / 
                             np.abs(results_df['label'])) * 100
        print(f'Overall: RMSE={overall_rmse:.2f}, APE={overall_ape:.2f}%')
        
        # Save to parquet
        os.makedirs('results', exist_ok=True)
        output_path = f'results/{model}_results.parquet'
        results_df.to_parquet(output_path, index=False, engine='pyarrow')
        print(f'Saved results to {output_path}')
    else:
        print('No results to save')

@click.command()
@click.option('--model',
              type=str,
              default='zoh',
              help='Model name(s) to run (comma-separated)')
@click.option('--subset_size',
              type=int,
              default=None,
              help='Subset size to run the benchmark on')
@click.option('--batch_size',
              type=int,
              default=1,
              help='Batch size for model inference')
@click.option('--device',
              type=str,
              default='cpu',
              help='Device to run the benchmark on')
def main(model, subset_size=None, batch_size=1, device='cpu'):
    """Run benchmark on specified models."""
    df = load_dataset(subset_size=subset_size)
    models = [m.strip() for m in model.split(',')]
    
    for model_name in models:
        print(f'\n{"="*60}')
        print(f'Running benchmark for model: {model_name}')
        print(f'{"="*60}')
        run_benchmark(model_name, df, batch_size=batch_size, device=device)

if __name__ == "__main__":
    main()
