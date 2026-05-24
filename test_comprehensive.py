#!/usr/bin/env python3
"""Comprehensive test untuk debug placement issue."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.core.container_env import ContainerEnv
from src.core.lbcp import is_stable
from src.utils.item_utils import get_item_dims

print("="*70)
print("TESTING FIX FOR PLACEMENT ISSUE")
print("="*70)

# Config: MINIMAL to isolate issue
env = ContainerEnv(
    dataset_type='perfect_pack_layered',
    perfect_pack_sigma=4,
    perfect_pack_size_bias=3.0,
    perfect_pack_mean_ratio=0.25,
    layered_min_height=2,
    layered_max_height=6,
    fast_stability_mask=True,
    use_structural_validation=False,  # Disable LBCP for isolation
    max_episode_length=500,
)

state, action_mask = env.reset(seed=42)
print(f"\nEnvironment initialized:")
print(f"  Container: {env.L}x{env.W}x{env.H}")
print(f"  Items: {len(env.items)}")
print(f"  Action mask valid: {action_mask.sum():.0f}/{len(action_mask)}")

# TEST: Can we place first item?
print(f"\n{'='*70}")
print("TEST 1: Try to place first item")
print(f"{'='*70}")

current_item = env.items[0]
item_l, item_w, item_h = get_item_dims(current_item)
print(f"Item 0: {item_l}x{item_w}x{item_h}")

valid_actions = np.where(action_mask > 0)[0]
print(f"Valid actions: {len(valid_actions)}")

if len(valid_actions) > 0:
    action = valid_actions[0]
    x = action % env.L
    y = action // env.L
    
    print(f"Trying position: ({x}, {y})")
    
    # Check validations
    print("\nValidation checks:")
    
    # Boundary
    if x + item_l > env.L or y + item_w > env.W:
        print(f"  Boundary: FAIL (would exceed bounds)")
    else:
        print(f"  Boundary: PASS")
    
    # Overflow
    base_height = env.height_map.max_height_in_region(x, y, item_l, item_w)
    if base_height + item_h > env.H:
        print(f"  Overflow: FAIL (height {base_height}+{item_h}>{env.H})")
    else:
        print(f"  Overflow: PASS (height {base_height}+{item_h}={base_height+item_h}<={env.H})")
    
    # Stability
    stable = is_stable(env.height_map.map, x, y, item_l, item_w, item_h, env.H, strict_mode=False)
    print(f"  Stability (strict_mode=False): {'PASS' if stable else 'FAIL'}")
    
    # Full validation
    is_valid = env._is_valid_position(x, y, item_l, item_w, item_h)
    print(f"  Overall _is_valid_position(): {is_valid}")
    
    # Try to step
    print(f"\nExecuting step with action={action}...")
    (next_state, next_mask), reward, done, info = env.step(action)
    print(f"  Success: {info['success']}")
    print(f"  Reward: {reward:.4f}")
    print(f"  Items placed: {len(env.placed_items)}")

# TEST: Collect stats from multiple placements
print(f"\n{'='*70}")
print("TEST 2: Collect rollout (up to 50 steps)")
print(f"{'='*70}")

env.reset(seed=42)
items_placed = 0
steps_taken = 0
episode_reward = 0.0

while steps_taken < 50 and env.current_index < len(env.items):
    state, action_mask = env._get_state_and_mask()
    valid_actions = np.where(action_mask > 0)[0]
    
    if len(valid_actions) == 0:
        print(f"Step {steps_taken}: No valid actions available!")
        break
    
    action = valid_actions[0]
    (next_state, next_mask), reward, done, info = env.step(action)
    
    if info['success']:
        items_placed += 1
        print(f"Step {steps_taken}: ✓ Item placed (reward={reward:.4f})")
    else:
        print(f"Step {steps_taken}: ✗ {info['action_type']} (reward={reward:.4f})")
    
    episode_reward += reward
    action_mask = next_mask
    steps_taken += 1
    
    if done:
        break

print(f"\n{'='*70}")
print("RESULTS")
print(f"{'='*70}")
print(f"Steps taken: {steps_taken}")
print(f"Items placed: {items_placed}")
print(f"Episode reward: {episode_reward:.4f}")
print(f"Utilization: {env.get_utilization():.2f}%")
print(f"\nFix status: {'✓ WORKING' if items_placed > 0 else '✗ STILL BROKEN'}")
