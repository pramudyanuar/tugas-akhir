import os
import glob
from pathlib import Path
import csv
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import sys

# Add root directory to python path
sys.path.append(str(Path(__file__).parent.parent))
from scripts.train import generate_training_plots

def extract_metrics():
    tb_dir = Path("logs/tensorboard")
    worker_dirs = list(tb_dir.glob("train_async_worker_*"))
    if not worker_dirs:
        print("No worker directories found under logs/tensorboard.")
        return
        
    print(f"Found {len(worker_dirs)} worker logs. Reading TensorBoard events...")
    
    # Store metrics: tag -> list of (step, value)
    all_metrics = {
        'episode/reward': [],
        'episode/utilization': [],
        'episode/success_rate': [],
        'loss/a3c_policy': [],
        'loss/a3c_value': [],
        'loss/a3c_entropy': [],
        'loss/a3c_total': [],
        'loss/high_level_policy': []
    }
    
    for w_dir in worker_dirs:
        event_files = list(w_dir.glob("events.out.tfevents.*"))
        if not event_files:
            continue
        # Use the latest event file
        event_file = max(event_files, key=os.path.getmtime)
        try:
            acc = EventAccumulator(str(event_file))
            acc.Reload()
            for tag in all_metrics.keys():
                if tag in acc.Tags()['scalars']:
                    for event in acc.Scalars(tag):
                        all_metrics[tag].append((event.step, event.value))
        except Exception as e:
            print(f"Error reading {event_file}: {e}")
            
    print("TensorBoard events loaded. Processing epoch records...")
    
    num_epochs = 500
    local_steps_per_epoch = 256 # 4096 total / 16 workers
    
    epoch_records = []
    for epoch in range(1, num_epochs + 1):
        start_step = (epoch - 1) * local_steps_per_epoch
        end_step = epoch * local_steps_per_epoch
        
        row = {'epoch': epoch}
        for tag, data in all_metrics.items():
            # Filter values in the current step range
            vals = [val for step, val in data if start_step <= step < end_step]
            
            # Map tags to CSV field names
            field_name = tag.replace('episode/', '').replace('loss/', '')
            if 'reward' in field_name:
                field_name = 'reward_mean_last100'
            elif 'utilization' in field_name:
                field_name = 'utilization_mean_last100'
            elif 'success_rate' in field_name:
                field_name = 'success_rate_mean_last100'
            elif 'a3c_policy' in field_name:
                field_name = 'a3c_policy_loss_mean'
            elif 'a3c_value' in field_name:
                field_name = 'a3c_value_loss_mean'
            elif 'a3c_entropy' in field_name:
                field_name = 'a3c_entropy_mean'
            elif 'a3c_total' in field_name:
                field_name = 'a3c_total_loss_mean'
            elif 'high_level_policy' in field_name:
                field_name = 'high_level_policy_loss_mean'
                
            if vals:
                row[field_name] = float(np.mean(vals))
            else:
                # Fallback to previous epoch value if no events in this range
                row[field_name] = epoch_records[-1][field_name] if epoch_records else 0.0
                
        row['deadlocks'] = 0
        row['rearrange_success_rate'] = 0.0
        epoch_records.append(row)
        
    # Write to CSV
    csv_path = Path("logs/training/training_epoch_metrics.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(epoch_records[0].keys()))
        writer.writeheader()
        writer.writerows(epoch_records)
        
    print(f"CSV saved to {csv_path}")
    print("Generating plots...")
    generate_training_plots(csv_path)
    print("Plots generated successfully!")

if __name__ == '__main__':
    extract_metrics()
