"""Policy comparison evaluation script.

Compares multiple policies on the same benchmark:
- A3C learned low-level policy
- Oracle DBLF-like greedy heuristic
- Oracle load-balance greedy heuristic
- Oracle height greedy heuristic
- Oracle center-priority greedy heuristic
- Random baseline
"""

import argparse
import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.container_env import ContainerEnv
from evaluate import compare_policies, evaluate
from src.learning.agents.a3c import A3C
from src.learning.models.high_level_agent import HighLevelAgent
from src.learning.agents.oracle_policy import OraclePolicy, RandomPolicy


def main():
    parser = argparse.ArgumentParser(description='Compare different bin packing policies')
    parser.add_argument('--episodes', type=int, default=50, help='Number of evaluation episodes')
    parser.add_argument('--max-items', type=int, default=20, help='Max items per episode')
    parser.add_argument('--device', type=str, default='cpu', help='Device: cpu or cuda')
    parser.add_argument('--dataset', type=str, default='random', help='Dataset type: random or cutting_stock')
    parser.add_argument('--a3c-checkpoint', type=str, default=None, help='Path to A3C checkpoint')
    parser.add_argument('--high-level-checkpoint', type=str, default=None, help='Path to HighLevelAgent checkpoint')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("Policy Comparison for 3D Bin Packing")
    print("="*70 + "\n")
    
    # Initialize environment for oracle policies
    env = ContainerEnv(max_items=args.max_items, seed=42, dataset_type=args.dataset)
    
    policies = {}
    
    # Add Oracle policies
    print("Loading heuristic baseline policies...")
    policies['oracle_dblf'] = OraclePolicy(env, priority='dblf')
    policies['oracle_load_balance'] = OraclePolicy(env, priority='load_balance')
    policies['oracle_height'] = OraclePolicy(env, priority='height')
    policies['oracle_center'] = OraclePolicy(env, priority='nearest_center')
    policies['random'] = RandomPolicy(env)
    
    # Add A3C if checkpoint provided
    if args.a3c_checkpoint and os.path.exists(args.a3c_checkpoint):
        print(f"Loading A3C from {args.a3c_checkpoint}...")
        a3c = A3C(
            state_size=env.state_size,
            action_size=env.action_size,
            L=env.L,
            W=env.W,
            learning_rate=3e-4,
            gamma=0.99,
            entropy_coef=0.01,
            value_coef=0.5,
            device=args.device
        )
        checkpoint = torch.load(args.a3c_checkpoint, map_location=args.device)
        if isinstance(checkpoint, dict) and 'network' in checkpoint:
            a3c.network.load_state_dict(checkpoint['network'])
        else:
            a3c.network.load_state_dict(checkpoint)
        a3c.network.eval()
        policies['a3c'] = a3c
    
    # Add HighLevelAgent if checkpoint provided. The current comparison script
    # keeps baselines and the learned low-level policy separate; hierarchical
    # evaluation is handled by scripts/evaluate.py.
    if args.high_level_checkpoint and os.path.exists(args.high_level_checkpoint):
        print(f"Loading HighLevelAgent from {args.high_level_checkpoint}...")
        high_level = HighLevelAgent(input_dim=env.state_size)
        checkpoint = torch.load(args.high_level_checkpoint, map_location=args.device)
        high_level.load_state_dict(checkpoint)
        high_level.eval()
        # For now, just use the high-level agent for forward pass
        # A more sophisticated evaluation would integrate both
    
    if len(policies) == 0:
        print("ERROR: No policies available. Please provide checkpoints or use oracle policies.")
        return
    
    print(f"\nEvaluating {len(policies)} policies...\n")
    
    # Run comparison
    results = compare_policies(
        policies,
        num_episodes=args.episodes,
        max_items=args.max_items,
        device=args.device,
        dataset_type=args.dataset
    )
    
    # Save results to CSV without requiring pandas.
    output_file = 'policy_comparison_results.csv'
    import csv
    metric_names = sorted(next(iter(results.values())).keys()) if results else []
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['policy'] + metric_names)
        for policy_name, metrics in results.items():
            writer.writerow([policy_name] + [metrics[name] for name in metric_names])
    print(f"Results saved to {output_file}")


if __name__ == '__main__':
    main()
