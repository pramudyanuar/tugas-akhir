#!/usr/bin/env python3
"""Debug reward calculation"""
import numpy as np
import sys
from src.core.container_env import ContainerEnv
from src.utils.item_utils import get_item_dims

# Initialize environment
env = ContainerEnv(
    container_length=59,
    container_width=23,
    container_height=23,
    dataset_type='perfect_pack_layered',
    perfect_pack_sigma=4,
    perfect_pack_size_bias=3.0,
    perfect_pack_mean_ratio=0.25,
    layered_min_height=2,
    layered_max_height=6,
    fast_stability_mask=False,
)

# Reset
print("=== Resetting environment ===")
state, action_mask = env.reset(seed=0)
print(f"Items generated: {len(env.items)}")
print(f"Sample items: {[get_item_dims(item) for item in env.items[:5]]}")
print(f"Current index: {env.current_index}")
print(f"Episode reward: {env.episode_reward}")
print(f"Episode length: {env.episode_length}")

if len(env.items) == 0:
    print("ERROR: No items generated!")
    sys.exit(1)

print("\n=== Running 10 steps ===")
for step_idx in range(10):
    # Get current item
    if env.current_index < len(env.items):
        current_item = env.items[env.current_index]
        item_dims = get_item_dims(current_item)
        print(f"\nStep {step_idx}: current_index={env.current_index}, item={item_dims}")
    
    # Random action (position or skip)
    valid_actions = np.where(action_mask > 0)[0]
    if len(valid_actions) > 0:
        action = valid_actions[0]  # Pick first valid action
    else:
        action = env.L * env.W  # Skip
    
    print(f"  Action: {action} (valid={action_mask[action]:.4f})")
    
    # Take step
    (next_state, next_mask), reward, done, info = env.step(action)
    
    print(f"  Reward: {reward:.6f}")
    print(f"  Done: {done}")
    print(f"  Info: {info}")
    print(f"  Episode reward so far: {env.episode_reward:.6f}")
    print(f"  Episode length: {env.episode_length}")
    
    state = next_state
    action_mask = next_mask
    
    if done:
        print(f"\n✓ Episode complete! Final reward: {env.episode_reward:.6f}")
        break
else:
    print(f"\n✗ Episode not complete after 10 steps")
    print(f"  Total items: {len(env.items)}")
    print(f"  Current index: {env.current_index}")
    print(f"  Episode reward: {env.episode_reward:.6f}")
