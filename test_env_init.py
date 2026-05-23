#!/usr/bin/env python3
"""Simple test to check if environment initializes properly"""
import numpy as np
from src.core.container_env import ContainerEnv

env = ContainerEnv(
    container_length=60,
    container_width=24,
    container_height=26,
    dataset_type='perfect_pack_layered',
    perfect_pack_sigma=4,
    perfect_pack_size_bias=3.0,
    perfect_pack_mean_ratio=0.25,
    layered_min_height=2,
    layered_max_height=6,
)

print("Resetting environment...")
state, mask = env.reset(seed=0)
print(f"Items generated: {len(env.items)}")
print(f"Container dimensions: {env.L}x{env.W}x{env.H} = {env.container_volume}")
print(f"Current index: {env.current_index}")
print(f"Episode reward: {env.episode_reward}")
print(f"Episode length: {env.episode_length}")

if len(env.items) > 0:
    print(f"First 5 items: {[str(item) for item in env.items[:5]]}")
else:
    print("ERROR: No items generated!")

print("\nRunning 5 steps...")
for i in range(5):
    valid_actions = np.where(mask > 0)[0]
    action = valid_actions[0] if len(valid_actions) > 0 else env.L * env.W
    
    (next_state, next_mask), reward, done, info = env.step(action)
    
    print(f"Step {i+1}: reward={reward:.6f}, done={done}, episode_reward={env.episode_reward:.6f}, "
          f"current_index={env.current_index}, episode_length={env.episode_length}")
    
    mask = next_mask
    if done:
        print("Episode complete!")
        break
