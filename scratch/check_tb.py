import os
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

tb_dir = "logs/tensorboard/train_async_worker_0"
files = sorted([os.path.join(tb_dir, f) for f in os.listdir(tb_dir) if "tfevents" in f])
if not files:
    print("No events files found.")
    exit(0)

target_file = files[-1]
print(f"Reading file: {target_file}")

try:
    ea = event_accumulator.EventAccumulator(target_file)
    ea.Reload()
    
    tags = ea.Tags().get('scalars', [])
    print("Tags:", tags)
    
    for tag in ['loss/a3c_total', 'loss/a3c_policy', 'loss/a3c_value', 'loss/a3c_entropy', 'loss/high_level_policy']:
        if tag in tags:
            events = ea.Scalars(tag)
            vals = [ev.value for ev in events]
            if len(vals) > 0:
                print(f"{tag:25s} | First 10: {np.mean(vals[:10]):.4f} | Last 10: {np.mean(vals[-10:]):.4f} | Mean: {np.mean(vals):.4f}")
except Exception as e:
    print("Error:", e)
