import numpy as np
from src.core.container_env import ContainerEnv

env = ContainerEnv(
    container_length=60,
    container_width=24,
    container_height=26,
    dataset_type='perfect_pack_layered',
    layered_min_height=2,
    layered_max_height=6,
    perfect_pack_sigma=4,
    perfect_pack_size_bias=3.0,
    perfect_pack_mean_ratio=0.25,
)

state, mask = env.reset(seed=42)
print(f"Number of items generated in episode: {len(env.items)}")
print("Item dimensions:")
for i, item in enumerate(env.items[:15]):
    print(f"  Item {i}: {item}")
if len(env.items) > 15:
    print(f"  ... and {len(env.items) - 15} more items")

total_vol = sum(item['l'] * item['w'] * item['h'] for item in env.items)
container_vol = 60 * 24 * 26
print(f"Total volume of all items: {total_vol} / {container_vol} ({total_vol/container_vol*100:.2f}%)")
