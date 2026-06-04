import numpy as np
from src.core.container_env import ContainerEnv
from src.learning.agents.oracle_policy import OraclePolicy, RandomPolicy

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
    fast_stability_mask=True,  # MUCH faster
)

policies = {
    'DBLF': OraclePolicy(env, priority='dblf'),
    'Load Balance': OraclePolicy(env, priority='load_balance'),
    'Random': RandomPolicy(env),
}

num_episodes = 5

for name, policy in policies.items():
    utils = []
    success_rates = []
    
    for ep in range(num_episodes):
        state, action_mask = env.reset(seed=42 + ep)
        done = False
        
        placed_count = 0
        while not done:
            action = policy.select_action(state, action_mask)
            (next_state, next_mask), reward, done, info = env.step(action)
            state = next_state
            action_mask = next_mask
            if info.get('success', False):
                placed_count += 1
                
        utils.append(env.get_utilization())
        success_rates.append(placed_count / len(env.items) * 100.0)
        
    print(f"Policy: {name}")
    print(f"  Mean Utilization: {np.mean(utils):.2f}% ± {np.std(utils):.2f}%")
    print(f"  Mean Success Rate (items placed): {np.mean(success_rates):.2f}%")
