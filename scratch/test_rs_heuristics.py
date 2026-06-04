import numpy as np
from src.core.container_env import ContainerEnv
from src.learning.agents.oracle_policy import OraclePolicy, RandomPolicy

# Initialize environment using the custom 'rs' dataset
env = ContainerEnv(
    container_length=10,
    container_width=10,
    container_height=10,
    dataset_type='rs',
    fast_stability_mask=True,
)

print(f"Loaded 'rs' dataset. Total episodes: {len(env.rs_data)}")
print(f"Container dimensions: {env.L}x{env.W}x{env.H}")

policies = {
    'DBLF': OraclePolicy(env, priority='dblf'),
    'Load Balance': OraclePolicy(env, priority='load_balance'),
    'Random': RandomPolicy(env),
}

num_episodes = 20

for name, policy in policies.items():
    utils = []
    success_rates = []
    rewards = []
    item_counts = []
    
    for ep in range(num_episodes):
        state, action_mask = env.reset(seed=42 + ep)
        done = False
        
        placed_count = 0
        episode_reward = 0.0
        while not done:
            action = policy.select_action(state, action_mask)
            (next_state, next_mask), reward, done, info = env.step(action)
            state = next_state
            action_mask = next_mask
            episode_reward += reward
            if info.get('success', False):
                placed_count += 1
                
        utils.append(env.get_utilization())
        success_rates.append(placed_count / len(env.items) * 100.0)
        rewards.append(episode_reward)
        item_counts.append(len(env.items))
        
    print(f"\nPolicy: {name}")
    print(f"  Mean Utilization: {np.mean(utils):.2f}% ± {np.std(utils):.2f}%")
    print(f"  Mean Success Rate (items placed): {np.mean(success_rates):.2f}%")
    print(f"  Mean Reward: {np.mean(rewards):.4f}")
    print(f"  Average Sequence/Item Count per Episode: {np.mean(item_counts):.1f}")
