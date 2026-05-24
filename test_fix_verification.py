#!/usr/bin/env python3
"""Quick test to verify fix works."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.core.container_env import ContainerEnv

env = ContainerEnv(
    dataset_type='perfect_pack_layered',
    perfect_pack_sigma=4,
    perfect_pack_size_bias=3.0,
    perfect_pack_mean_ratio=0.25,
    layered_min_height=2,
    layered_max_height=6,
    fast_stability_mask=True,
    use_structural_validation=False,
)

state, action_mask = env.reset(seed=42)

print(f"Initial action_mask valid count: {action_mask.sum()}")

# Try to place items
items_placed = 0
for step_idx in range(20):  # Try 20 steps
    valid_actions = np.where(action_mask > 0)[0]
    if len(valid_actions) == 0:
        print(f"Step {step_idx}: No valid actions")
        break
    
    action = valid_actions[0]  # Take first valid action
    (next_state, next_mask), reward, done, info = env.step(action)
    
    if info['success']:
        items_placed += 1
        print(f"Step {step_idx}: SUCCESS! Reward={reward:.4f}, Items={items_placed}")
    else:
        print(f"Step {step_idx}: FAILED - {info['action_type']}")
    
    action_mask = next_mask
    if done:
        break

print(f"\n{'='*50}")
print(f"Final result: {items_placed} items placed out of {len(env.items)}")
print(f"Utilization: {env.get_utilization():.2f}%")
print(f"Test: {'PASS' if items_placed > 0 else 'FAIL'}")
