#!/usr/bin/env python3
"""Mini training run to verify placement fix works during actual training."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.core.container_env import ContainerEnv
from src.learning.agents.a3c import A3C
from src.learning.models.actor_critic import ActorCriticNetwork

print("="*70)
print("MINI TRAINING TEST - Verify Placement Works")
print("="*70)

# Create environment matching training config
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
    fast_stability_mask=True,
    use_structural_validation=False,
    max_episode_length=500,
)

# Create A3C agent
network = ActorCriticNetwork(L=env.L, W=env.W, action_size=env.action_size)
a3c = A3C(
    state_size=env.state_size,
    action_size=env.action_size,
    L=env.L,
    W=env.W,
    device='cpu',
    network=network,
)

print(f"\nEnvironment:")
print(f"  Container: {env.L}x{env.W}x{env.H}")
print(f"  State size: {env.state_size}")
print(f"  Action size: {env.action_size}")

# Simulate one episode
print(f"\n{'='*70}")
print("Simulating one episode with random policy")
print(f"{'='*70}\n")

state, action_mask = env.reset(seed=42)
episode_reward = 0.0
items_placed = 0
step_count = 0
successful_placements = []

while step_count < 100 and env.current_index < len(env.items):
    # Get action mask for current state
    state, action_mask = env._get_state_and_mask()
    
    # Count valid actions
    valid_actions_count = int(action_mask.sum())
    
    if valid_actions_count == 0:
        print(f"Step {step_count}: No valid actions available!")
        break
    
    # Random action from valid ones
    valid_action_indices = np.where(action_mask > 0)[0]
    action = np.random.choice(valid_action_indices)
    
    # Execute action
    (next_state, next_mask), reward, done, info = env.step(action)
    
    episode_reward += reward
    step_count += 1
    
    # Print status
    action_type = info['action_type']
    success_str = "✓" if info['success'] else "✗"
    
    if info['success']:
        items_placed += 1
        successful_placements.append({
            'step': step_count,
            'action': action,
            'reward': reward
        })
        print(f"Step {step_count}: {success_str} Placement | reward={reward:7.4f} | items={items_placed}")
    else:
        print(f"Step {step_count}: {success_str} {action_type:7s}    | reward={reward:7.4f} | items={items_placed}")
    
    state = next_state
    action_mask = next_mask
    
    if done:
        break

# Final results
print(f"\n{'='*70}")
print("FINAL RESULTS")
print(f"{'='*70}")
print(f"Total steps: {step_count}")
print(f"Items placed: {items_placed}/{len(env.items)}")
print(f"Episode reward: {episode_reward:.4f}")
print(f"Container utilization: {env.get_utilization():.2f}%")
print(f"Max height used: {env.get_max_height()}/{env.H}")

if items_placed > 0:
    print(f"\n✓ FIX VERIFIED: Items are being placed successfully!")
    print(f"  First placement at step {successful_placements[0]['step']}")
    print(f"  Average reward per placement: {sum(p['reward'] for p in successful_placements) / len(successful_placements):.4f}")
else:
    print(f"\n✗ FIX FAILED: No items placed despite valid actions available!")
    print(f"  Action mask still reports valid actions but all are rejected")

print(f"\nTest {'PASSED' if items_placed > 0 else 'FAILED'}")
