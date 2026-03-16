import numpy as np
import torch
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from env.container_env import ContainerEnv
from env.lbcp import is_stable
from rl.high_level_agent import HighLevelAgent
from rl.low_level_agent import LowLevelAgent
from planning.mcts import MCTS
from utils.metrics import Metrics


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
    
    def compute_volume_utilization(self, placed_items):
        """
        Compute volume utilization efficiency.
        
        Args:
            placed_items: List of tuples (length, width, height)
            
        Returns:
            float: Utilization percentage (0-100)
        """
        if len(placed_items) == 0:
            return 0.0
        
        total_volume = sum(l * w * h for l, w, h in placed_items)
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
            placed_items: List of tuples (length, width, height)
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
            l, w, h = item
            
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
            l, w, h = item
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
            placed_items: List of tuples (length, width, height)
            
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
            l, w, h = item
            
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
        
        total_items = len(env.items)
        episode_reward = 0.0
        step_count = 0
        
        while step_count < total_items * 2:  # Max steps = 2x items
            # High-level decision
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            high_output = model_high(state_tensor)
            strategy = model_high.select_strategy(high_output['strategy_logits'])
            
            # Optional: Use MCTS untuk planning
            if use_mcts:
                mcts = MCTS(env, budget=num_simulations)
                mcts_result = mcts.search(state, action_mask, depth_limit=10)
                action = mcts_result['best_action']
            else:
                # Low-level decision
                height_map_tensor = torch.FloatTensor(
                    env.height_map.map
                ).unsqueeze(0).unsqueeze(0)
                
                item = env.items[env.current_index] if env.current_index < len(env.items) else (1, 1, 1)
                item_dim = torch.FloatTensor([item[0] / env.L, item[1] / env.W, item[2] / env.H]).unsqueeze(0)
                
                low_output = model_low(height_map_tensor, item_dim)
                action_logits = low_output
                
                # Mask invalid actions
                masked_logits = action_logits.clone()
                masked_logits[0, action_mask == 0] = -float('inf')
                
                action = torch.argmax(masked_logits).item()
            
            # Execute action
            (next_state, next_mask), reward, done, info = env.step(action)
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
            'max_height': env.get_max_height()
        }
    
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
        print("=" * 70 + "\n")


def evaluate(model_high=None, model_low=None, num_episodes=5, use_mcts=True):
    """
    Run evaluation dengan multiple episodes.
    
    Args:
        model_high: High-level agent (if None, create default)
        model_low: Low-level agent (if None, create default)
        num_episodes: Number of episodes to evaluate
        use_mcts: Whether to use MCTS planning
        
    Returns:
        dict: Evaluation results
    """
    # Create models if not provided
    if model_high is None:
        model_high = HighLevelAgent()
    if model_low is None:
        model_low = LowLevelAgent()
    
    # Create evaluator
    evaluator = EvaluationMetrics(container_dims=(59, 23, 23))
    
    # Run evaluation
    for episode in range(num_episodes):
        env = ContainerEnv(max_items=20, seed=episode)
        
        print(f"Evaluating episode {episode + 1}/{num_episodes}...", end=" ")
        result = evaluator.evaluate_episode(env, model_high, model_low, 
                                           use_mcts=use_mcts, num_simulations=30)
        
        print(f"Util: {result['utilization']:.1f}% | "
              f"LB: {result['load_balance']:.3f} | "
              f"SR: {result['success_rate']:.1%} | "
              f"Reward: {result['episode_reward']:.2f}")
    
    # Print summary
    evaluator.print_summary()
    
    return evaluator.get_summary_statistics()


if __name__ == "__main__":
    """Test evaluation metrics"""
    
    print("=" * 70)
    print("Running Evaluation Test")
    print("=" * 70)
    
    # Import necessary modules
    from rl.high_level_agent import HighLevelAgent
    from rl.low_level_agent import LowLevelAgent
    
    # Create models
    model_high = HighLevelAgent()
    model_low = LowLevelAgent()
    
    # Run evaluation
    results = evaluate(model_high, model_low, num_episodes=3, use_mcts=False)
    
    print("✓ Evaluation completed successfully!")