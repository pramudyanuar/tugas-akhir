#!/usr/bin/env python3
"""Debug test untuk cek apakah action mask semua 0."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.core.container_env import ContainerEnv

# Create environment
env = ContainerEnv(
    dataset_type='perfect_pack_layered',
    perfect_pack_sigma=4,
    perfect_pack_size_bias=3.0,
    perfect_pack_mean_ratio=0.25,
    layered_min_height=2,
    layered_max_height=6,
    fast_stability_mask=True,
    use_structural_validation=False,  # Disable untuk first test
)

# Reset
state, action_mask = env.reset(seed=42)

print(f"Initial state shape: {state.shape}")
print(f"Initial action_mask shape: {action_mask.shape}")
print(f"Action mask dtype: {action_mask.dtype}")
print(f"Action mask sum (valid actions): {action_mask.sum()}")
print(f"Action mask all zeros?: {(action_mask == 0).all()}")
print(f"Number of items: {len(env.items)}")
print(f"Current item index: {env.current_index}")

# Get first item info
if len(env.items) > 0:
    from src.utils.item_utils import get_item_dims
    item_l, item_w, item_h = get_item_dims(env.items[0])
    print(f"First item dims: ({item_l}, {item_w}, {item_h})")

print("\n=== FIRST 5 STEPS ===")
from src.core.lbcp import is_stable

for step_idx in range(5):
    print(f"\n--- Step {step_idx} ---")
    
    # Get action mask
    state, action_mask = env._get_state_and_mask()
    valid_actions = np.where(action_mask > 0)[0]
    skip_action_valid = bool(action_mask[-1])
    
    print(f"Valid placement positions: {len(valid_actions[:-1]) if len(valid_actions) > 0 else 0}")
    print(f"Skip action valid: {skip_action_valid}")
    print(f"Total valid actions: {len(valid_actions)}")
    
    if len(valid_actions) == 0:
        print("ERROR: No valid actions available!")
        break
    
    # Take first valid action (not random) for debugging
    action = valid_actions[0]
    x = action % env.L
    y = action // env.L
    
    # Get current item
    current_item = env.items[env.current_index]
    from src.utils.item_utils import get_item_dims
    item_l, item_w, item_h = get_item_dims(current_item)
    
    print(f"Taking action: {action} (x={x}, y={y}, skip={action == env.L * env.W})")
    print(f"Item dims: ({item_l}, {item_w}, {item_h})")
    
    # Check if _is_valid_position would accept it
    base_height = env.height_map.max_height_in_region(x, y, item_l, item_w)
    stability = is_stable(env.height_map.map, x, y, item_l, item_w, item_h, env.H)
    print(f"Base height: {base_height}, New height: {base_height + item_h}, Max: {env.H}")
    print(f"is_stable() result: {stability}")
    
    is_valid = env._is_valid_position(x, y, item_l, item_w, item_h)
    print(f"_is_valid_position() result: {is_valid}")
    
    (next_state, next_mask), reward, done, info = env.step(action)
    print(f"Reward: {reward:.4f} | Done: {done} | Success: {info['success']}")
    print(f"Items placed: {len(env.placed_items)}")
    
    if done:
        print("Episode done!")
        break

print(f"\nFinal items placed: {len(env.placed_items)}")
print(f"Final utilization: {env.get_utilization():.2f}%")
