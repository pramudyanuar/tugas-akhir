import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

tb_dir = "logs/tensorboard/train_async_worker_0"
files = sorted([os.path.join(tb_dir, f) for f in os.listdir(tb_dir) if "events.out.tfevents" in f])
if not files:
    print("No events files found.")
    exit(1)

target_file = files[-1]
print(f"Reading file: {target_file}")

ea = event_accumulator.EventAccumulator(target_file)
ea.Reload()

# Get metrics
metrics = {}
tags_to_extract = {
    'episode/utilization': 'Utilization (%)',
    'episode/reward': 'Episode Reward',
    'loss/a3c_total': 'A3C Total Loss'
}

for tag, label in tags_to_extract.items():
    if tag in ea.Tags().get('scalars', []):
        events = ea.Scalars(tag)
        steps = [ev.step for ev in events]
        values = [ev.value for ev in events]
        metrics[label] = (steps, values)
    else:
        print(f"Tag {tag} not found.")

# Plotting
fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=False)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

for idx, (label, (steps, values)) in enumerate(metrics.items()):
    ax = axes[idx]
    
    # Smooth the curve for better readability
    if len(values) > 100:
        # Simple moving average
        window_size = 50
        smoothed_values = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
        smoothed_steps = steps[window_size-1:]
        ax.plot(steps, values, alpha=0.2, color='#1f77b4', label='Raw')
        ax.plot(smoothed_steps, smoothed_values, color='#1f77b4', linewidth=2, label='Smoothed (MA=50)')
        ax.legend()
    else:
        ax.plot(steps, values, color='#1f77b4', linewidth=2)
        
    ax.set_title(f"{label} over Steps", fontsize=14, fontweight='bold')
    ax.set_xlabel("Steps", fontsize=11)
    ax.set_ylabel(label, fontsize=11)
    ax.tick_params(axis='both', which='major', labelsize=10)

plt.tight_layout()

# Save path
out_dir = "/home/user/.gemini/antigravity/brain/ceca81aa-c015-4b44-a8f1-aa9954761514"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "training_progress_chart.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Plot saved to: {out_path}")
