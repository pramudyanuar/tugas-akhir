import numpy as np
import torch
import torch.nn.functional as F
import sys
import os
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from src.core.container_env import ContainerEnv
from src.core.lbcp import is_stable
from src.core.candidate_generator import CandidateGenerator
from src.learning.models.high_level_agent import HighLevelAgent
from src.learning.models.actor_critic import ActorCriticNetwork
from src.planning.mcts import MCTS
from src.planning.high_level_search import HighLevelSearcher
from src.learning.agents.oracle_policy import OraclePolicy, RandomPolicy
from visualization import ContainerVisualizer
from src.utils.metrics import Metrics
from src.utils.item_utils import get_item_dims, make_item


class EvaluationMetrics:
    """
    Comprehensive evaluation module untuk 3D Bin Packing dengan LBCP.
    
    Metrics:
    1. Volume Utilization Efficiency: Percentage ruang container yang terisi
    2. Load Distribution Balance: Seberapa seimbang load di dalam kontainer
    3. Stability Rate: Percentage items yang stable
    4. Success Rate: Percentage items yang berhasil ditempatkan
    5. Center of Gravity: Tracking penyimpangan CoG dari ideal center
    """
    
    def __init__(self, container_dims=(59, 23, 23)):
        """
        Initialize evaluation metrics.
        
        Args:
            container_dims: Tuple of (length, width, height)
        """
        self.L, self.W, self.H = container_dims
        self.container_volume = self.L * self.W * self.H
        
        # Track metrics across episodes
        self.episodes_utilization = []
        self.episodes_load_balance = []
        self.episodes_stability_rate = []
        self.episodes_success_rate = []
        self.episodes_cog_deviation = []
        self.episodes_reward = []
        self.episodes_rearrange_attempts = []
        self.episodes_rearrange_success_rate = []
        self.episodes_rearrange_apply_rate = []
        self.episodes_avg_rearrange_value = []
        self.episodes_avg_unpack_depth = []
    
    def compute_volume_utilization(self, placed_items):
        """
        Compute volume utilization efficiency.
        
        Args:
            placed_items: List of item dicts atau tuples
            
        Returns:
            float: Utilization percentage (0-100)
        """
        if len(placed_items) == 0:
            return 0.0
        
        total_volume = sum(
            get_item_dims(item)[0] * get_item_dims(item)[1] * get_item_dims(item)[2]
            for item in placed_items
        )
        utilization = (total_volume / self.container_volume) * 100.0
        
        return min(utilization, 100.0)
    
    def compute_load_distribution_balance(self, placed_positions, placed_items, height_map):
        """
        Compute load distribution balance menggunakan center of gravity analysis.
        
        Strategy:
        1. Divide container menjadi 4 quadrants (2x2)
        2. Compute total weight di setiap quadrant
        3. Compute coefficient of variation dari quadrant weights
        4. Load balance = 1 / (1 + CV)
        
        Args:
            placed_positions: List of tuples (x, y, base_height)
            placed_items: List of item dicts atau tuples
            height_map: 2D array of heights
            
        Returns:
            dict: {
                'load_balance': float (0-1),
                'quadrant_weights': list of 4 weights,
                'cog_deviation': float
            }
        """
        if len(placed_items) == 0:
            return {
                'load_balance': 1.0,
                'quadrant_weights': [0.0, 0.0, 0.0, 0.0],
                'cog_deviation': 0.0
            }
        
        # Divide container menjadi 4 quadrants
        mid_x = self.L / 2.0
        mid_y = self.W / 2.0
        
        quadrant_weights = [0.0, 0.0, 0.0, 0.0]
        total_weight = 0.0
        
        for pos, item in zip(placed_positions, placed_items):
            x, y, _ = pos
            l, w, h = get_item_dims(item)
            
            # Compute center dari item
            item_center_x = x + l / 2.0
            item_center_y = y + w / 2.0
            
            # Item volume sebagai proxy untuk weight
            item_weight = l * w * h
            total_weight += item_weight
            
            # Assign ke quadrant
            if item_center_x < mid_x and item_center_y < mid_y:
                quadrant_weights[0] += item_weight  # Bottom-left
            elif item_center_x >= mid_x and item_center_y < mid_y:
                quadrant_weights[1] += item_weight  # Bottom-right
            elif item_center_x < mid_x and item_center_y >= mid_y:
                quadrant_weights[2] += item_weight  # Top-left
            else:
                quadrant_weights[3] += item_weight  # Top-right
        
        # Compute load balance
        if total_weight > 0:
            avg_weight = total_weight / 4.0
            variance = sum((w - avg_weight) ** 2 for w in quadrant_weights) / 4.0
            std_dev = np.sqrt(variance)
            cv = std_dev / avg_weight if avg_weight > 0 else 0.0
            load_balance = 1.0 / (1.0 + cv)
        else:
            load_balance = 1.0
        
        # Compute CoG deviation
        total_cog_x = 0.0
        total_cog_y = 0.0
        
        for pos, item in zip(placed_positions, placed_items):
            x, y, _ = pos
            l, w, h = get_item_dims(item)
            item_weight = l * w * h
            
            item_cog_x = x + l / 2.0
            item_cog_y = y + w / 2.0
            
            total_cog_x += item_cog_x * item_weight
            total_cog_y += item_cog_y * item_weight
        
        # Ideal CoG adalah center dari container
        ideal_cog_x = self.L / 2.0
        ideal_cog_y = self.W / 2.0
        
        if total_weight > 0:
            actual_cog_x = total_cog_x / total_weight
            actual_cog_y = total_cog_y / total_weight
            
            cog_deviation = np.sqrt((actual_cog_x - ideal_cog_x) ** 2 + 
                                   (actual_cog_y - ideal_cog_y) ** 2)
        else:
            cog_deviation = 0.0
        
        return {
            'load_balance': load_balance,
            'quadrant_weights': quadrant_weights,
            'cog_deviation': cog_deviation
        }
    
    def compute_stability_rate(self, height_map, placed_positions, placed_items):
        """
        Compute stability rate menggunakan LBCP validation.
        
        Args:
            height_map: 2D array of heights
            placed_positions: List of tuples (x, y, base_height)
            placed_items: List of item dicts atau tuples
            
        Returns:
            dict: {
                'stability_rate': float (0-1),
                'stable_count': int,
                'total_count': int
            }
        """
        if len(placed_items) == 0:
            return {
                'stability_rate': 1.0,
                'stable_count': 0,
                'total_count': 0
            }
        
        stable_count = 0
        
        for pos, item in zip(placed_positions, placed_items):
            x, y, _ = pos
            l, w, h = get_item_dims(item)
            
            try:
                if is_stable(height_map, x, y, l, w, h, self.H):
                    stable_count += 1
            except Exception:
                # If stability check fails, consider unstable
                pass
        
        stability_rate = stable_count / len(placed_items) if len(placed_items) > 0 else 1.0
        
        return {
            'stability_rate': stability_rate,
            'stable_count': stable_count,
            'total_count': len(placed_items)
        }
    
    def compute_success_rate(self, total_items, placed_items):
        """
        Compute success rate (items placed / total items).
        
        Args:
            total_items: Total items dalam episode
            placed_items: Items yang berhasil ditempatkan
            
        Returns:
            float: Success rate (0-1)
        """
        if total_items == 0:
            return 1.0
        
        return len(placed_items) / total_items
    
    def evaluate_episode(self, env, model_high, model_low, use_mcts=True, num_simulations=50):
        """
        Evaluate single episode dengan comprehensive metrics.
        
        Args:
            env: ContainerEnv instance
            model_high: High-level agent
            model_low: Low-level agent
            use_mcts: Whether to use MCTS
            num_simulations: MCTS simulation budget
            
        Returns:
            dict: Complete evaluation metrics untuk episode
        """
        state, action_mask = env.reset()
        candidate_generator = CandidateGenerator(env.L, env.W)
        
        total_items = len(env.items)
        episode_reward = 0.0
        step_count = 0
        rearrange_attempts = 0
        rearrange_success = 0
        rearrange_applied = 0
        rearrange_value_sum = 0.0
        rearrange_depth_sum = 0.0
        
        while step_count < total_items * 2:  # Max steps = 2x items
            if env.current_index >= total_items:
                break

            # High-level decision
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            high_output = model_high(state_tensor)
            strategy = model_high.select_strategy(high_output['strategy_logits'])

            macro_decision = model_high.decode_macro_decision(strategy)
            orientation = macro_decision.get('orientation', 0)

            policy_state, policy_action_mask = env._get_state_and_mask(
                orientation=orientation
            )

            # Candidate generation + feasibility masking
            candidate_actions = candidate_generator.generate_from_macro(
                policy_action_mask,
                macro_decision=macro_decision,
                top_k=128
            )

            if len(candidate_actions) > 0:
                # Low-level policy picks an action from candidate set
                action = self._select_low_level_action(
                    env,
                    model_low,
                    policy_action_mask,
                    candidate_actions
                )
            else:
                # Deadlock handling: try repack first (if allowed), then MCTS.
                action = env.L * env.W  # default: skip action

                if macro_decision.get('allow_repacking', False) and len(env.placed_items) > 0:
                    repack_result = env.perform_repack(strategy='auto')
                    if repack_result.get('success', False):
                        policy_state, policy_action_mask = env._get_state_and_mask(
                            orientation=orientation
                        )
                    candidate_actions = candidate_generator.generate_from_macro(
                        policy_action_mask,
                        macro_decision=macro_decision,
                        top_k=128
                    )

                    if len(candidate_actions) > 0:
                        action = self._select_low_level_action(
                            env,
                            model_low,
                            policy_action_mask,
                            candidate_actions
                        )

                if action == env.L * env.W and use_mcts:
                    mcts = MCTS(env, budget=num_simulations)
                    rearrange_attempts += 1
                    rearr_result = mcts.search_rearrangement(
                        failed_item=env.items[env.current_index]
                        if env.current_index < len(env.items) else None,
                        max_unpack=3,
                        apply_to_env=True,
                    )

                    if rearr_result.get('success', False):
                        rearrange_success += 1
                    if rearr_result.get('applied', False):
                        rearrange_applied += 1
                    rearrange_value_sum += float(rearr_result.get('best_value', 0.0))
                    rearrange_depth_sum += float(len(rearr_result.get('best_sequence', [])))

                    if rearr_result.get('applied', False):
                        policy_state, policy_action_mask = env._get_state_and_mask(
                            orientation=orientation
                        )
                        candidate_actions = candidate_generator.generate_from_macro(
                            policy_action_mask,
                            macro_decision=macro_decision,
                            top_k=128
                        )

                        if len(candidate_actions) > 0:
                            action = self._select_low_level_action(
                                env,
                                model_low,
                                policy_action_mask,
                                candidate_actions
                            )

                    if action == env.L * env.W:
                        mcts_result = mcts.search(state, action_mask, depth_limit=10)
                        mcts_action = int(mcts_result['best_action'])

                        if 0 <= mcts_action < len(action_mask) and action_mask[mcts_action] > 0:
                            action = mcts_action
            
            # Execute action
            (next_state, next_mask), reward, done, info = env.step(
                (action, orientation)
            )
            episode_reward += reward
            
            state = next_state
            action_mask = next_mask
            step_count += 1
            
            if done:
                break

        # Compute all metrics
        utilization = self.compute_volume_utilization(env.placed_items)
        
        load_dist = self.compute_load_distribution_balance(
            env.placed_positions,
            env.placed_items,
            env.height_map.map
        )
        
        stability = self.compute_stability_rate(
            env.height_map.map,
            env.placed_positions,
            env.placed_items
        )
        
        success_rate = self.compute_success_rate(total_items, env.placed_items)
        
        # Store metrics
        self.episodes_utilization.append(utilization)
        self.episodes_load_balance.append(load_dist['load_balance'])
        self.episodes_stability_rate.append(stability['stability_rate'])
        self.episodes_success_rate.append(success_rate)
        self.episodes_cog_deviation.append(load_dist['cog_deviation'])
        self.episodes_reward.append(episode_reward)

        denom = max(rearrange_attempts, 1)
        rearrange_success_rate = rearrange_success / denom
        rearrange_apply_rate = rearrange_applied / denom
        avg_rearrange_value = rearrange_value_sum / denom
        avg_unpack_depth = rearrange_depth_sum / denom

        self.episodes_rearrange_attempts.append(rearrange_attempts)
        self.episodes_rearrange_success_rate.append(rearrange_success_rate)
        self.episodes_rearrange_apply_rate.append(rearrange_apply_rate)
        self.episodes_avg_rearrange_value.append(avg_rearrange_value)
        self.episodes_avg_unpack_depth.append(avg_unpack_depth)
        
        return {
            'utilization': utilization,
            'load_balance': load_dist['load_balance'],
            'cog_deviation': load_dist['cog_deviation'],
            'quadrant_weights': load_dist['quadrant_weights'],
            'stability_rate': stability['stability_rate'],
            'stable_count': stability['stable_count'],
            'total_items': stability['total_count'],
            'success_rate': success_rate,
            'episode_reward': episode_reward,
            'steps': step_count,
            'items_placed': len(env.placed_items),
            'max_height': env.get_max_height(),
            'rearrange_attempts': rearrange_attempts,
            'rearrange_success_rate': rearrange_success_rate,
            'rearrange_apply_rate': rearrange_apply_rate,
            'avg_rearrange_value': avg_rearrange_value,
            'avg_unpack_depth': avg_unpack_depth,
        }

    def _select_low_level_action(self, env, model_low, action_mask, candidate_actions):
        """
        Select low-level action from candidate set using masked policy sampling.
        """
        height_map_tensor = torch.FloatTensor(env.height_map.map).unsqueeze(0).unsqueeze(0)
        item = env.items[env.current_index] if env.current_index < len(env.items) else make_item(1, 1, 1)
        item_l, item_w, item_h = get_item_dims(item)
        item_dim = torch.FloatTensor([
            item_l / env.L,
            item_w / env.W,
            item_h / env.H,
        ]).unsqueeze(0)

        with torch.no_grad():
            logits = model_low(height_map_tensor, item_dim)

        env_mask = torch.BoolTensor(np.asarray(action_mask) > 0).unsqueeze(0)
        candidate_mask = torch.zeros_like(env_mask)
        for action in candidate_actions:
            if 0 <= action < env.L * env.W:
                candidate_mask[0, action] = True

        combined_mask = env_mask & candidate_mask

        if not torch.any(combined_mask):
            # Fallback: if no candidate survives, choose skip if valid.
            skip_idx = env.L * env.W
            if skip_idx < len(action_mask) and action_mask[skip_idx] > 0:
                return skip_idx

            valid_actions = np.where(np.asarray(action_mask) > 0)[0]
            return int(valid_actions[0]) if len(valid_actions) > 0 else skip_idx

        masked_logits = logits.clone()
        masked_logits[~combined_mask] = float('-inf')

        probs = F.softmax(masked_logits, dim=-1)
        probs = torch.where(torch.isnan(probs), torch.zeros_like(probs), probs)

        probs_sum = probs.sum(dim=-1, keepdim=True)
        probs = torch.where(probs_sum > 0, probs / probs_sum, probs)

        if torch.sum(probs) <= 0:
            valid_idx = torch.where(combined_mask[0])[0]
            return int(valid_idx[0].item()) if len(valid_idx) > 0 else env.L * env.W

        action = torch.distributions.Categorical(probs).sample()
        return int(action.item())
    
    def get_summary_statistics(self):
        """
        Get summary statistics across all evaluated episodes.
        
        Returns:
            dict: Summary statistics
        """
        if len(self.episodes_utilization) == 0:
            return {}
        
        return {
            'avg_utilization': np.mean(self.episodes_utilization),
            'std_utilization': np.std(self.episodes_utilization),
            'avg_load_balance': np.mean(self.episodes_load_balance),
            'std_load_balance': np.std(self.episodes_load_balance),
            'avg_stability_rate': np.mean(self.episodes_stability_rate),
            'std_stability_rate': np.std(self.episodes_stability_rate),
            'avg_success_rate': np.mean(self.episodes_success_rate),
            'std_success_rate': np.std(self.episodes_success_rate),
            'avg_cog_deviation': np.mean(self.episodes_cog_deviation),
            'std_cog_deviation': np.std(self.episodes_cog_deviation),
            'avg_reward': np.mean(self.episodes_reward),
            'std_reward': np.std(self.episodes_reward),
            'avg_rearrange_attempts': np.mean(self.episodes_rearrange_attempts),
            'avg_rearrange_success_rate': np.mean(self.episodes_rearrange_success_rate),
            'avg_rearrange_apply_rate': np.mean(self.episodes_rearrange_apply_rate),
            'avg_rearrange_value': np.mean(self.episodes_avg_rearrange_value),
            'avg_unpack_depth': np.mean(self.episodes_avg_unpack_depth),
            'num_episodes': len(self.episodes_utilization)
        }
    
    def print_summary(self):
        """Print formatted summary statistics."""
        stats = self.get_summary_statistics()
        
        if not stats:
            print("No evaluation data available")
            return
        
        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY STATISTICS")
        print("=" * 70)
        print(f"Number of episodes: {stats['num_episodes']}")
        print()
        print("Volume Utilization Efficiency:")
        print(f"  Mean: {stats['avg_utilization']:.2f}% ± {stats['std_utilization']:.2f}%")
        print()
        print("Load Distribution Balance:")
        print(f"  Mean: {stats['avg_load_balance']:.4f} ± {stats['std_load_balance']:.4f}")
        print(f"  (1.0 = perfectly balanced, 0.0 = completely imbalanced)")
        print()
        print("Stability Rate:")
        print(f"  Mean: {stats['avg_stability_rate']:.2%} ± {stats['std_stability_rate']:.2%}")
        print()
        print("Success Rate (items placed / total items):")
        print(f"  Mean: {stats['avg_success_rate']:.2%} ± {stats['std_success_rate']:.2%}")
        print()
        print("Center of Gravity Deviation:")
        print(f"  Mean: {stats['avg_cog_deviation']:.4f} ± {stats['std_cog_deviation']:.4f}")
        print(f"  (Lower is better, ideal < 1.0)")
        print()
        print("Episode Reward:")
        print(f"  Mean: {stats['avg_reward']:.4f} ± {stats['std_reward']:.4f}")
        print()
        print("Rearrangement (MCTS):")
        print(f"  Avg Attempts/Episode: {stats['avg_rearrange_attempts']:.2f}")
        print(f"  Success Rate:         {stats['avg_rearrange_success_rate']:.2%}")
        print(f"  Apply Rate:           {stats['avg_rearrange_apply_rate']:.2%}")
        print(f"  Avg Rearrange Value:  {stats['avg_rearrange_value']:.4f}")
        print(f"  Avg Unpack Depth:     {stats['avg_unpack_depth']:.2f}")
        print("=" * 70 + "\n")

    def export_episode_metrics_csv(self, filepath):
        """
        Export per-episode metrics to CSV for reporting.
        """
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            'episode',
            'utilization',
            'load_balance',
            'stability_rate',
            'success_rate',
            'cog_deviation',
            'episode_reward',
            'rearrange_attempts',
            'rearrange_success_rate',
            'rearrange_apply_rate',
            'avg_rearrange_value',
            'avg_unpack_depth',
        ]

        with output_path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            num_episodes = len(self.episodes_utilization)
            for i in range(num_episodes):
                writer.writerow({
                    'episode': i + 1,
                    'utilization': self.episodes_utilization[i],
                    'load_balance': self.episodes_load_balance[i],
                    'stability_rate': self.episodes_stability_rate[i],
                    'success_rate': self.episodes_success_rate[i],
                    'cog_deviation': self.episodes_cog_deviation[i],
                    'episode_reward': self.episodes_reward[i],
                    'rearrange_attempts': self.episodes_rearrange_attempts[i],
                    'rearrange_success_rate': self.episodes_rearrange_success_rate[i],
                    'rearrange_apply_rate': self.episodes_rearrange_apply_rate[i],
                    'avg_rearrange_value': self.episodes_avg_rearrange_value[i],
                    'avg_unpack_depth': self.episodes_avg_unpack_depth[i],
                })

        return str(output_path)


def evaluate(model_high=None, model_low=None, num_episodes=5, use_mcts=True,
             output_csv='logs/evaluation/evaluation_episode_metrics.csv',
             dataset_type='random', save_visualizations=True):
    """
    Run evaluation dengan multiple episodes.
    
    Args:
        model_high: High-level agent (if None, create default)
        model_low: Low-level agent (if None, create default)
        num_episodes: Number of episodes to evaluate
        use_mcts: Whether to use MCTS planning
        dataset_type: 'random', 'cutting_stock', 'perfect_pack', or 'perfect_pack_layered'
        save_visualizations: Whether to save visualizations during eval
        
    Returns:
        dict: Evaluation results
    """
    # Create models if not provided
    if model_high is None:
        model_high = HighLevelAgent()
    if model_low is None:
        model_low = ActorCriticNetwork(L=59, W=23, action_size=59*23+1)
    
    # Create evaluator
    evaluator = EvaluationMetrics(container_dims=(59, 23, 23))
    
    # Setup visualization
    visualizer = ContainerVisualizer(container_dims=(59, 23, 23)) if save_visualizations else None
    vis_dir = Path('outputs/evaluation_visualizations') if save_visualizations else None
    if save_visualizations:
        vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Run evaluation
    for episode in range(num_episodes):
        env = ContainerEnv(max_items=20, seed=episode, dataset_type=dataset_type)
        
        print(f"Evaluating episode {episode + 1}/{num_episodes}...", end=" ")
        result = evaluator.evaluate_episode(env, model_high, model_low, 
                                           use_mcts=use_mcts, num_simulations=30)
        
        print(f"Util: {result['utilization']:.1f}% | "
              f"LB: {result['load_balance']:.3f} | "
              f"SR: {result['success_rate']:.1%} | "
              f"Reward: {result['episode_reward']:.2f}")
        
        # Save visualization for this episode
        if save_visualizations and len(env.placed_items) > 0:
            try:
                vis_file = vis_dir / f"eval_episode_{episode:03d}.png"
                visualizer.visualize_packing_2d(
                    env.placed_items,
                    env.placed_positions,
                    env.height_map,
                    title=f"Eval Episode {episode}: util={result['utilization']:.1f}%, "
                          f"bal={result['load_balance']:.3f}"
                )
                plt.savefig(str(vis_file), dpi=100, bbox_inches='tight')
                plt.close()
            except Exception as e:
                pass  # Silently skip visualization errors
    
    # Print summary
    evaluator.print_summary()

    csv_path = evaluator.export_episode_metrics_csv(output_csv)
    print(f"Episode metrics CSV saved: {csv_path}")
    
    if save_visualizations:
        print(f"Visualizations saved to: {vis_dir}")
    
    return evaluator.get_summary_statistics()


if __name__ == "__main__":
    """Test evaluation metrics"""
    
    print("=" * 70)
    print("Running Evaluation Test")
    print("=" * 70)
    
    # Import necessary modules
    from rl.high_level_agent import HighLevelAgent
    from src.learning.models.actor_critic import ActorCriticNetwork
    
    # Create models
    model_high = HighLevelAgent()
    model_low = ActorCriticNetwork(L=59, W=23, action_size=59*23+1)
    
    # Run evaluation
    results = evaluate(model_high, model_low, num_episodes=3, use_mcts=False)


def compare_policies(policies_dict, num_episodes=50, max_items=20, device='cpu', dataset_type='random'):
    """
    Compare multiple policies on the same environment and dataset.
    
    Args:
        policies_dict: Dict mapping policy_name -> policy_instance
            e.g., {
                'a3c': a3c_policy,
                'oracle_dblf': OraclePolicy(env, priority='dblf'),
                'oracle_load_balance': OraclePolicy(env, priority='load_balance'),
                'oracle_height': OraclePolicy(env, priority='height'),
                'random': RandomPolicy(env)
            }
        num_episodes (int): Number of episodes to run
        max_items (int): Max items per episode
        device (str): 'cpu' or 'cuda'
        dataset_type (str): 'random' or 'cutting_stock'
        
    Returns:
        dict: Comparison results with metrics for each policy
    """
    print("\n" + "="*70)
    print("POLICY COMPARISON EVALUATION")
    print("="*70 + "\n")
    
    # Initialize environment
    env = ContainerEnv(max_items=max_items, seed=42, dataset_type=dataset_type)
    metrics = EvaluationMetrics()
    
    results = {}
    
    for policy_name, policy in policies_dict.items():
        print(f"\nEvaluating {policy_name}...")
        print(f"{'='*50}")
        
        policy_metrics = {
            'utilization': [],
            'load_balance': [],
            'stability_rate': [],
            'success_rate': [],
            'total_reward': [],
        }
        
        for episode in range(num_episodes):
            state, action_mask = env.reset()
            episode_reward = 0.0
            done = False
            
            while not done:
                # Select action using policy
                if hasattr(policy, 'select_action'):
                    selected = policy.select_action(state, action_mask)
                    action = selected[0] if isinstance(selected, tuple) else selected
                else:
                    # For raw actor-critic models: convert to tensor and pass through network
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    mask_tensor = torch.FloatTensor(action_mask).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        logits, value = policy.network(state_tensor)
                    
                    masked_logits = policy.mask_logits(logits, mask_tensor)
                    action = torch.argmax(masked_logits[0]).item()
                
                (state, action_mask), reward, done, info = env.step(action)
                episode_reward += reward
            
            # Compute metrics for this episode
            utilization = metrics.compute_volume_utilization(env.placed_items)
            load_balance_info = metrics.compute_load_distribution_balance(
                env.placed_positions,
                env.placed_items,
                env.height_map
            )
            stability_info = metrics.compute_stability_rate(
                env.height_map.map,
                env.placed_positions,
                env.placed_items
            )
            stability_rate = 100.0 * stability_info['stability_rate']
            success_rate = 100.0 * len(env.placed_items) / len(env.items)
            
            policy_metrics['utilization'].append(utilization)
            policy_metrics['load_balance'].append(load_balance_info['load_balance'])
            policy_metrics['stability_rate'].append(stability_rate)
            policy_metrics['success_rate'].append(success_rate)
            policy_metrics['total_reward'].append(episode_reward)
            
            if (episode + 1) % max(1, num_episodes // 5) == 0:
                print(f"  Episode {episode+1}/{num_episodes}: "
                      f"Util={np.mean(policy_metrics['utilization'][-5:]):.1f}%, "
                      f"Balance={np.mean(policy_metrics['load_balance'][-5:]):.3f}, "
                      f"Success={np.mean(policy_metrics['success_rate'][-5:]):.1f}%")
        
        # Compute statistics
        results[policy_name] = {
            'utilization_mean': np.mean(policy_metrics['utilization']),
            'utilization_std': np.std(policy_metrics['utilization']),
            'load_balance_mean': np.mean(policy_metrics['load_balance']),
            'load_balance_std': np.std(policy_metrics['load_balance']),
            'stability_rate_mean': np.mean(policy_metrics['stability_rate']),
            'stability_rate_std': np.std(policy_metrics['stability_rate']),
            'success_rate_mean': np.mean(policy_metrics['success_rate']),
            'success_rate_std': np.std(policy_metrics['success_rate']),
            'total_reward_mean': np.mean(policy_metrics['total_reward']),
            'total_reward_std': np.std(policy_metrics['total_reward']),
        }
    
    # Print comparison table
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70 + "\n")
    
    metric_names = [
        'utilization_mean',
        'utilization_std',
        'load_balance_mean',
        'load_balance_std',
        'stability_rate_mean',
        'stability_rate_std',
        'success_rate_mean',
        'success_rate_std',
        'total_reward_mean',
        'total_reward_std',
    ]
    header = ["policy"] + metric_names
    rows = []
    for policy_name, values in results.items():
        rows.append([policy_name] + [values[name] for name in metric_names])

    col_widths = [
        max(len(str(row[i])) for row in ([header] + rows))
        for i in range(len(header))
    ]
    print(" | ".join(str(value).ljust(col_widths[i]) for i, value in enumerate(header)))
    print("-+-".join("-" * width for width in col_widths))
    for row in rows:
        formatted = [row[0]] + [f"{float(value):.4f}" for value in row[1:]]
        print(" | ".join(str(value).ljust(col_widths[i]) for i, value in enumerate(formatted)))
    
    print("\n" + "="*70 + "\n")
    
    return results

    
    print("✓ Evaluation completed successfully!")
