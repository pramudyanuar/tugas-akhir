import torch
import numpy as np
from src.core.container_env import ContainerEnv
from src.learning.models.actor_critic import ActorCriticNetwork
from src.learning.models.high_level_agent import HighLevelAgent

# Load checkpoints
checkpoint_low = "logs/training/async_interrupted.pt"
checkpoint_high = "logs/training/async_high_level_interrupted.pt"

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

# Load networks
model_high = HighLevelAgent(input_dim=60 * 24 + 4, num_strategies=8)
try:
    model_high.load_state_dict(torch.load(checkpoint_high, map_location='cpu'))
    print("Loaded high-level model.")
except Exception as e:
    print("High-level load error:", e)

model_low = ActorCriticNetwork(L=60, W=24, action_size=60 * 24 + 1)
try:
    model_low.load_state_dict(torch.load(checkpoint_low, map_location='cpu'))
    print("Loaded low-level model.")
except Exception as e:
    print("Low-level load error:", e)

# Run one episode
state, action_mask = env.reset(seed=42)
done = False
step_count = 0

print(f"\nStarting episode with {len(env.items)} items.")
print(f"Initial state: height map max = {np.max(env.height_map.map)}")

total_skips = 0
total_invalids = 0
total_placements = 0

while not done:
    step_count += 1
    item = env.items[env.current_index]
    
    # Run high-level decision
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    high_output = model_high(state_tensor)
    strategy_logits = high_output['strategy_logits']
    strategy = torch.argmax(strategy_logits, dim=-1).item()
    macro_decision = model_high.decode_macro_decision(strategy)
    orientation = macro_decision.get('orientation', 0)
    zone_priority = macro_decision.get('zone_priority', 'center')
    
    # Recompute state and mask for orientation
    policy_state, policy_action_mask = env._get_state_and_mask(orientation=orientation)
    
    # Count valid actions in policy_action_mask (excluding skip)
    valid_indices = np.where(policy_action_mask[:-1] > 0)[0]
    num_valid = len(valid_indices)
    
    # Select low-level action using A3C
    state_t = torch.FloatTensor(policy_state).unsqueeze(0)
    mask_t = torch.FloatTensor(policy_action_mask).unsqueeze(0)
    
    policy_logits, _ = model_low(state_t)
    # Apply mask
    masked_logits = policy_logits.clone()
    masked_logits[mask_t == 0] = -1e9
    action = torch.argmax(masked_logits, dim=-1).item()
    
    # Take step
    (next_state, next_mask), reward, done, info = env.step((action, orientation))
    state = next_state
    action_mask = next_mask
    
    # Log step
    action_type = info.get('action_type', 'unknown')
    if action == env.L * env.W:
        action_desc = "SKIP"
        total_skips += 1
    elif action_type == 'invalid':
        action_desc = f"INVALID placement at ({action % env.L}, {action // env.L})"
        total_invalids += 1
    else:
        action_desc = f"PLACED at {info.get('position')} with orientation {orientation}"
        total_placements += 1
        
    print(f"Step {step_count:02d} | Item {env.current_index-1:02d}: l={item['l']}, w={item['w']}, h={item['h']}, stack={item['stacking']} | Strategy: {strategy} (zone={zone_priority}) | Valid actions: {num_valid} | Action: {action_desc} | Util: {env.get_utilization():.2f}%")

print(f"\nEpisode finished. Steps: {step_count}, Placed: {total_placements}, Skips: {total_skips}, Invalids: {total_invalids}, Final Util: {env.get_utilization():.2f}%")
