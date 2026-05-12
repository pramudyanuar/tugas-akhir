import numpy as np
import torch
import torch.nn.functional as F
import sys
import os
import csv
import time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt

# Add workspace to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.container_env import ContainerEnv
from src.core.candidate_generator import CandidateGenerator
from src.learning.models.high_level_agent import HighLevelAgent
from src.learning.agents.ppo import PPO
from src.planning.mcts import MCTS
from src.planning.high_level_search import HighLevelSearcher
from visualization import ContainerVisualizer
from src.utils.metrics import Metrics
from src.utils.item_utils import get_item_dims


class TrainingLogger:
    """
    Logger untuk training metrics.
    """
    
    def __init__(self, log_dir='logs/training'):
        """
        Initialize logger.
        
        Args:
            log_dir (str): Directory untuk save logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_utilizations = []
        self.episode_success_rates = []
        
    def log_episode(self, episode_reward, episode_length, utilization, 
                   items_placed, total_items):
        """Log episode metrics."""
        success_rate = (items_placed / total_items * 100) if total_items > 0 else 0.0
        
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.episode_utilizations.append(utilization)
        self.episode_success_rates.append(success_rate)
    
    def get_stats(self):
        """Get training statistics."""
        if not self.episode_rewards:
            return {}
        
        return {
            'reward_mean': np.mean(self.episode_rewards[-100:]),
            'reward_std': np.std(self.episode_rewards[-100:]),
            'length_mean': np.mean(self.episode_lengths[-100:]),
            'utilization_mean': np.mean(self.episode_utilizations[-100:]),
            'success_rate_mean': np.mean(self.episode_success_rates[-100:]),
        }
    
    def print_episode_summary(self, episode, reward, length, utilization, items_placed, total_items):
        """Print episode summary."""
        success_rate = (items_placed / total_items * 100) if total_items > 0 else 0.0
        
        print(f"Episode {episode:4d} | "
              f"Reward: {reward:8.4f} | "
              f"Length: {length:3d} | "
              f"Util: {utilization:6.2f}% | "
              f"Success: {success_rate:5.1f}% | "
              f"Items: {items_placed}/{total_items}")


class TrainingLoop:
    """
    Training loop untuk PPO agent dengan 3D bin packing environment.
    
    Features:
    - Collect N steps dari environment
    - Compute returns menggunakan GAE
    - Update network dengan PPO
    - Log metrics: reward, utilization, episode length
    """
    
    def __init__(self, env, ppo_agent, n_steps=2048, 
                 device='cpu', seed=None, debug_actions=False,
                 vis_interval=10, vis_dir='outputs/visualizations',
                 blf_only=False):
        """
        Initialize training loop.
        
        Args:
            env (ContainerEnv): Environment
            ppo_agent (PPO): PPO agent
            n_steps (int): Number of steps to collect per update
            device (str): 'cpu' atau 'cuda'
            seed (int): Random seed
        """
        self.env = env
        self.ppo = ppo_agent
        self.n_steps = n_steps
        self.device = device
        self.seed = seed
        self.debug_actions = bool(debug_actions)

        # Hierarchical components for macro decision + candidate selection.
        self.high_level_agent = HighLevelAgent(input_dim=env.state_size).to(device)
        self.high_level_agent.train()  # Enable training mode
        
        # Optimizer for HighLevelAgent
        self.high_level_optimizer = torch.optim.Adam(
            self.high_level_agent.parameters(), 
            lr=1e-4
        )
        
        self.candidate_generator = CandidateGenerator(env.L, env.W)
        self.mcts_budget = 20
        
        self.logger = TrainingLogger()
        self.rearrange_stats = {
            'deadlocks': 0,
            'rearrange_attempts': 0,
            'rearrange_success': 0,
            'rearrange_applied': 0,
            'rearrange_best_value_sum': 0.0,
            'rearrange_unpack_depth_sum': 0.0,
            'mcts_fallback_used': 0,
            'mcts_active_used': 0,  # MCTS used during regular action selection
        }
        
        self.total_steps = 0
        self.episode_count = 0
        self.blf_only = bool(blf_only)
        
        # Visualization setup with correct container dimensions
        self.visualizer = ContainerVisualizer(container_dims=(env.L, env.W, env.H))
        self.vis_dir = Path(vis_dir)
        self.vis_dir.mkdir(parents=True, exist_ok=True)
        self.vis_interval = max(1, int(vis_interval))  # Save visualization every N episodes
        self.last_vis_episode = 0

    def _select_hierarchical_action(self, state, action_mask, sample_strategy=True):
        """Select action with TWO-LEVEL HIERARCHICAL RL:
        
        LEVEL 1 (HIGH-LEVEL: STRATEGY SELECTION):
            - HighLevelAgent selects WHAT strategy (orientation, repack, skip, etc.)
            - Outputs strategy + strategy_log_prob for training
        
        LEVEL 2 (LOW-LEVEL: POSITION SELECTION):
            - Given strategy, PPO agent selects WHERE to place (position)
            - Uses ActorCriticNetwork + action masking
            - Outputs position + log_prob + value estimate
        
        COMPLETE DECISION FLOW:
            State -> HighLevelAgent -> Strategy (e.g., "rotate 90°")
                  -> CandidateGenerator -> Filter valid positions
                  -> PPO Agent -> Select position (e.g., "place at x=10, y=5")
                  -> Environment.step() -> Execute & get reward
        
        DEADLOCK HANDLING:
            - If PPO selects invalid move (deadlock), trigger fallback:
              * Try attempt_repack() first
              * If still stuck, use MCTS search for escape
        
        TRAINING UPDATES:
            - PPO loss: Policy + value loss on collected trajectories
            - HighLevelAgent: Ready for future hierarchical training
        
        Args:
            state: Current state (height_map + item dimensions)
            action_mask: Legal actions mask (1 = valid, 0 = invalid)
            sample_strategy (bool): If True, sample strategies (training mode)
                                  If False, use greedy selection (eval mode)
        
        Returns:
            tuple: (action, log_prob, value, effective_action_mask, strategy_info)
                   - action: Position index selected by PPO
                   - log_prob: PPO's log probability of action
                   - value: PPO's value estimate of current state
                   - effective_action_mask: Mask after filtering by strategy
                   - strategy_info: {'strategy', 'strategy_log_prob', 'strategy_logits'}
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # Forward pass in training mode to enable strategy gradients
        high_output = self.high_level_agent(state_tensor)
        strategy_logits = high_output['strategy_logits']
        
        # Select strategy with gradient tracking
        strategy, strategy_log_prob = self.high_level_agent.select_strategy(
            strategy_logits, 
            sample=sample_strategy
        )
        
        macro_decision = self.high_level_agent.decode_macro_decision(strategy)
        orientation = macro_decision.get('orientation', 0)

        # Recompute state/mask for the selected orientation.
        policy_state, policy_action_mask = self.env._get_state_and_mask(
            orientation=orientation
        )

        # Candidate generation from macro decision.
        candidate_actions = self.candidate_generator.generate_from_macro(
            policy_action_mask,
            macro_decision=macro_decision,
            top_k=128
        )

        effective_mask = np.zeros_like(policy_action_mask, dtype=np.float32)
        if len(candidate_actions) > 0:
            for idx in candidate_actions:
                if 0 <= idx < self.env.L * self.env.W and policy_action_mask[idx] > 0:
                    effective_mask[idx] = 1.0

        strategy_info = {
            'strategy': strategy,
            'strategy_log_prob': strategy_log_prob,
            'strategy_logits': strategy_logits,
            'load_balance': high_output['load_balance'],
            'orientation': orientation
        }

        if np.sum(effective_mask[:-1]) > 0:
            if self.blf_only:
                valid_actions = np.where(np.asarray(effective_mask[:-1]) > 0)[0]
                if len(valid_actions) > 0:
                    action = int(valid_actions.min())
                else:
                    action = self.env.L * self.env.W
                log_prob, value = self._compute_logprob_value_for_action(
                    policy_state, effective_mask, action
                )
                return action, log_prob, value, effective_mask, strategy_info, policy_state

            # Use PPO agent to select position with masked action space
            ppo_action, ppo_log_prob, ppo_value = self.ppo.select_action(
                policy_state, effective_mask
            )
            
            # Return action from PPO - let the policy learn good placement
            return ppo_action, ppo_log_prob, ppo_value, effective_mask, strategy_info, policy_state

        self.rearrange_stats['deadlocks'] += 1

        # Deadlock handling: optional repack if no candidate position exists.
        if macro_decision.get('allow_repacking', False) and len(self.env.placed_items) > 0:
            repack_result = self.env.perform_repack(strategy='auto')
            if repack_result.get('success', False):
                # Recompute state and mask after repacking.
                policy_state, policy_action_mask = self.env._get_state_and_mask(
                    orientation=orientation
                )
                candidate_actions = self.candidate_generator.generate_from_macro(
                    policy_action_mask,
                    macro_decision=macro_decision,
                    top_k=128
                )
                effective_mask = np.zeros_like(policy_action_mask, dtype=np.float32)
                for idx in candidate_actions:
                    if 0 <= idx < self.env.L * self.env.W and policy_action_mask[idx] > 0:
                        effective_mask[idx] = 1.0

                if np.sum(effective_mask[:-1]) > 0:
                    action, log_prob, value = self.ppo.select_action(policy_state, effective_mask)
                    return action, log_prob, value, effective_mask, strategy_info, policy_state

        # If still deadlocked, use MCTS as planner fallback.
        mcts = MCTS(self.env, budget=self.mcts_budget)
        self.rearrange_stats['rearrange_attempts'] += 1
        rearr_result = mcts.search_rearrangement(
            failed_item=self.env.items[self.env.current_index]
            if self.env.current_index < len(self.env.items) else None,
            max_unpack=3,
            apply_to_env=True,
        )

        if rearr_result.get('success', False):
            self.rearrange_stats['rearrange_success'] += 1
        if rearr_result.get('applied', False):
            self.rearrange_stats['rearrange_applied'] += 1
        self.rearrange_stats['rearrange_best_value_sum'] += float(rearr_result.get('best_value', 0.0))
        self.rearrange_stats['rearrange_unpack_depth_sum'] += float(len(rearr_result.get('best_sequence', [])))

        if rearr_result.get('applied', False):
            policy_state, policy_action_mask = self.env._get_state_and_mask(
                orientation=orientation
            )
            candidate_actions = self.candidate_generator.generate_from_macro(
                policy_action_mask,
                macro_decision=macro_decision,
                top_k=128
            )
            effective_mask = np.zeros_like(policy_action_mask, dtype=np.float32)
            for idx in candidate_actions:
                if 0 <= idx < self.env.L * self.env.W and policy_action_mask[idx] > 0:
                    effective_mask[idx] = 1.0

            if np.sum(effective_mask[:-1]) > 0:
                action, log_prob, value = self.ppo.select_action(policy_state, effective_mask)
                return action, log_prob, value, effective_mask, strategy_info, policy_state

        mcts_result = mcts.search(policy_state, policy_action_mask, depth_limit=5)
        self.rearrange_stats['mcts_fallback_used'] += 1
        mcts_action = int(mcts_result['best_action'])

        # Ensure selected action is legal.
        if 0 <= mcts_action < len(policy_action_mask) and policy_action_mask[mcts_action] > 0:
            action = mcts_action
        else:
            valid_actions = np.where(np.asarray(policy_action_mask) > 0)[0]
            action = int(valid_actions[0]) if len(valid_actions) > 0 else self.env.L * self.env.W

        # Compute log_prob and value for the chosen fallback action.
        log_prob, value = self._compute_logprob_value_for_action(policy_state, policy_action_mask, action)
        return action, log_prob, value, np.asarray(policy_action_mask, dtype=np.float32), strategy_info, policy_state

    def _compute_logprob_value_for_action(self, state, action_mask, action):
        """Compute log probability and value for a fixed action under current PPO policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mask_tensor = torch.FloatTensor(action_mask).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, value = self.ppo.network(state_tensor)

        masked_logits = self.ppo.mask_logits(logits, mask_tensor)
        probs = F.softmax(masked_logits, dim=-1)
        probs = torch.where(torch.isnan(probs), torch.zeros_like(probs), probs)

        probs_sum = probs.sum(dim=-1, keepdim=True)
        probs = torch.where(probs_sum > 0, probs / probs_sum, probs)

        if probs[0, action] <= 0:
            log_prob = torch.tensor(-20.0, device=self.device)
        else:
            log_prob = torch.log(probs[0, action] + 1e-12)

        return log_prob.item(), value.item()
    
    def _blend_ppo_mcts_decision(self, state, action_mask, effective_mask, ppo_action, ppo_log_prob, ppo_value, use_mcts_prob=0.2):
        """
        Blend PPO action with MCTS search results.
        
        Args:
            state: Current state
            action_mask: Full action mask
            effective_mask: Filtered action mask from candidates
            ppo_action: Action selected by PPO
            ppo_log_prob: Log probability from PPO
            ppo_value: Value estimate from PPO
            use_mcts_prob: Probability of consulting MCTS (default 0.2 = 20%)
            
        Returns:
            tuple: (final_action, final_log_prob, final_value, mcts_used)
        """
        # Decide whether to use MCTS
        use_mcts = np.random.random() < use_mcts_prob
        mcts_used = False
        
        if not use_mcts:
            return ppo_action, ppo_log_prob, ppo_value, False
        
        try:
            # Run MCTS search with limited budget
            mcts = MCTS(self.env, budget=self.mcts_budget // 2)  # Use half budget for faster computation
            mcts_result = mcts.search(state, action_mask, depth_limit=5)
            mcts_action = int(mcts_result['best_action'])
            
            # Check if MCTS action is valid
            if 0 <= mcts_action < len(action_mask) and action_mask[mcts_action] > 0:
                # MCTS found a good action - prefer MCTS in ~70% of the time, blend with PPO otherwise
                if np.random.random() < 0.7:
                    # Use MCTS action
                    final_action = mcts_action
                    log_prob, value = self._compute_logprob_value_for_action(state, action_mask, mcts_action)
                    mcts_used = True
                else:
                    # Blend: use PPO action but let PPO know about MCTS preference
                    final_action = ppo_action
                    log_prob, value = ppo_log_prob, ppo_value
                    mcts_used = False
            else:
                # MCTS action was invalid, stick with PPO
                final_action = ppo_action
                log_prob, value = ppo_log_prob, ppo_value
                mcts_used = False
                
            return final_action, log_prob, value, mcts_used
            
        except Exception as e:
            # If MCTS fails, fall back to PPO
            return ppo_action, ppo_log_prob, ppo_value, False
    
    def _update_high_level_agent(self, strategy_buffer, num_epochs=3):
        """
        Update HighLevelAgent using strategy buffer with policy gradient.
        
        DISABLED: Currently has gradient computation issues (in-place modifications).
                 Will be re-enabled after proper gradient graph management.
        
        Args:
            strategy_buffer (list): List of dicts with strategy info
            num_epochs (int): Number of update passes
        """
        # TODO: Fix and re-enable high-level agent training
        pass
    
    def _save_visualization(self, episode_num, suffix=""):
        """
        Save visualization of current container state.
        
        Args:
            episode_num (int): Episode number
            suffix (str): Optional suffix for filename
        """
        try:
            if len(self.env.placed_items) == 0:
                return  # Skip if no items placed
            
            title = f"Episode {episode_num}: n_items={len(self.env.placed_items)}, util={self.env.get_utilization():.1f}%"
            
            # Save 2D visualization
            fig_2d = self.visualizer.visualize_packing_2d(
                self.env.placed_items,
                self.env.placed_positions,
                self.env.height_map,
                title=title
            )
            filename_2d = self.vis_dir / f"episode_{episode_num:04d}_2d{suffix}.png"
            fig_2d.savefig(str(filename_2d), dpi=100, bbox_inches='tight')
            plt.close(fig_2d)
            
            # Save 3D visualization
            fig_3d = self.visualizer.visualize_packing_3d(
                self.env.placed_items,
                self.env.placed_positions,
                title=title
            )
            filename_3d = self.vis_dir / f"episode_{episode_num:04d}_3d{suffix}.png"
            fig_3d.savefig(str(filename_3d), dpi=100, bbox_inches='tight')
            plt.close(fig_3d)
            
            # Save cross-sections
            fig_cross = self.visualizer.visualize_cross_sections(
                self.env.height_map,
                title=f"Episode {episode_num}: Cross-Sections"
            )
            filename_cross = self.vis_dir / f"episode_{episode_num:04d}_cross{suffix}.png"
            fig_cross.savefig(str(filename_cross), dpi=100, bbox_inches='tight')
            plt.close(fig_cross)
            
        except Exception as e:
            import traceback
            print(f"\n⚠️  Visualization error at episode {episode_num}: {e}")
            traceback.print_exc()
    
    def collect_steps(self, n_steps):
        """
        Collect N steps dari environment.
        
        Args:
            n_steps (int): Number of steps to collect
            
        Returns:
            dict: Collected trajectories
        """
        steps_collected = 0
        timing = {
            'action_select_s': 0.0,
            'env_step_s': 0.0,
            'store_transition_s': 0.0,
            'episode_reset_s': 0.0,
            'visualization_s': 0.0,
        }
        episode_info = {
            'rewards': [],
            'lengths': [],
            'utilizations': [],
            'placed_items': [],
            'total_items': []
        }
        strategy_buffer = []  # Store strategy info for HighLevelAgent update
        
        # Reset environment
        reset_start = time.perf_counter()
        state, action_mask = self.env.reset(seed=self.seed)
        timing['episode_reset_s'] += time.perf_counter() - reset_start
        if self.seed is not None:
            self.seed += 1  # Increment seed untuk variety
        
        while steps_collected < n_steps:
            # Select action with hierarchical control (sample strategies for training).
            select_start = time.perf_counter()
            action, log_prob, value, effective_mask, strategy_info, policy_state = self._select_hierarchical_action(
                state, action_mask, sample_strategy=True
            )
            timing['action_select_s'] += time.perf_counter() - select_start
            
            if self.debug_actions:
                valid_count = int(np.sum(effective_mask[:-1] > 0))
                action_valid = (
                    0 <= action < len(effective_mask)
                    and effective_mask[action] > 0
                )
                print(
                    f"Action debug | step={self.total_steps} action={action} "
                    f"valid_count={valid_count} action_valid={action_valid}"
                )

            # Take step in environment
            step_start = time.perf_counter()
            (next_state, next_mask), reward, done, info = self.env.step(
                (action, strategy_info.get('orientation', 0))
            )
            timing['env_step_s'] += time.perf_counter() - step_start
            
            # Store PPO transition
            store_start = time.perf_counter()
            self.ppo.store_transition(
                state=policy_state,
                action=action,
                reward=reward,
                value=value,
                log_prob=log_prob,
                action_mask=effective_mask,
                done=1.0 if done else 0.0
            )
            timing['store_transition_s'] += time.perf_counter() - store_start
            
            # Store strategy info for HighLevelAgent update
            strategy_buffer.append({
                'strategy_logits': strategy_info['strategy_logits'],
                'strategy_log_prob': strategy_info['strategy_log_prob'],
                'reward': reward,
                'load_balance': strategy_info['load_balance']
            })
            
            steps_collected += 1
            self.total_steps += 1
            
            state = next_state
            action_mask = next_mask
            
            # Episode done
            if done:
                episode_reward = self.env.episode_reward
                episode_length = self.env.episode_length
                utilization = self.env.get_utilization()
                items_placed = len(self.env.placed_items)
                total_items = len(self.env.items)
                
                episode_info['rewards'].append(episode_reward)
                episode_info['lengths'].append(episode_length)
                episode_info['utilizations'].append(utilization)
                episode_info['placed_items'].append(items_placed)
                episode_info['total_items'].append(total_items)
                
                self.logger.log_episode(episode_reward, episode_length, 
                                       utilization, items_placed, total_items)
                self.logger.print_episode_summary(self.episode_count, episode_reward, 
                                                 episode_length, utilization, 
                                                 items_placed, total_items)
                
                self.episode_count += 1
                
                # Save visualization every vis_interval episodes
                if self.episode_count % self.vis_interval == 0 and self.episode_count > self.last_vis_episode:
                    vis_start = time.perf_counter()
                    self._save_visualization(self.episode_count)
                    timing['visualization_s'] += time.perf_counter() - vis_start
                    self.last_vis_episode = self.episode_count
                
                # Reset environment untuk episode baru
                reset_start = time.perf_counter()
                state, action_mask = self.env.reset(seed=self.seed)
                timing['episode_reset_s'] += time.perf_counter() - reset_start
                if self.seed is not None:
                    self.seed += 1
        
        # Get next value untuk GAE bootstrap
        if not done:
            _, _, next_value, _, _, _ = self._select_hierarchical_action(
                state, action_mask, sample_strategy=False
            )
        else:
            next_value = 0.0
        
        return episode_info, next_value, strategy_buffer, timing
    
    def train_epoch(self, num_epochs=2, log_frequency=10):
        """
        Train untuk satu epoch (collect steps + update).
        
        Args:
            num_epochs (int): Jumlah PPO update epochs (reduced to 2 for faster training)
            log_frequency (int): Frequency untuk print summary
        """
        print(f"\n{'='*70}")
        print(f"Training Epoch {self.episode_count // log_frequency + 1}")
        print(f"{'='*70}\n")
        
        epoch_start = time.perf_counter()

        # Collect trajectories
        episode_info, next_value, strategy_buffer, timing = self.collect_steps(self.n_steps)
        
        # Update HighLevelAgent
        print(f"\nUpdating HighLevelAgent... ", end='', flush=True)
        self._update_high_level_agent(strategy_buffer)
        print("Done!")
        
        # Update PPO network (reduced epochs and batch size for speed)
        print(f"Updating PPO network... ", end='', flush=True)
        self.ppo.update(next_value=next_value, num_epochs=num_epochs, batch_size=32)
        print("Done!\n")

        epoch_elapsed = time.perf_counter() - epoch_start
        steps_per_sec = self.n_steps / epoch_elapsed if epoch_elapsed > 0 else 0.0

        timing_total = sum(timing.values())
        if timing_total > 0:
            def pct(value):
                return (value / timing_total) * 100.0
        else:
            def pct(value):
                return 0.0
        
        # Print statistics
        stats = self.logger.get_stats()
        print(f"{'='*70}")
        print("Training Statistics (last 100 episodes):")
        print(f"{'='*70}")
        print(f"Episode Reward:      {stats.get('reward_mean', 0):.4f} ± {stats.get('reward_std', 0):.4f}")
        print(f"Episode Length:      {stats.get('length_mean', 0):.1f} steps")
        print(f"Container Util:      {stats.get('utilization_mean', 0):.2f}%")
        print(f"Success Rate:        {stats.get('success_rate_mean', 0):.1f}%")

        rearr_attempts = max(self.rearrange_stats['rearrange_attempts'], 1)
        rearr_success_rate = 100.0 * self.rearrange_stats['rearrange_success'] / rearr_attempts
        rearr_apply_rate = 100.0 * self.rearrange_stats['rearrange_applied'] / rearr_attempts
        avg_rearr_value = self.rearrange_stats['rearrange_best_value_sum'] / rearr_attempts
        avg_unpack_depth = self.rearrange_stats['rearrange_unpack_depth_sum'] / rearr_attempts

        print(f"Deadlocks:            {self.rearrange_stats['deadlocks']}")
        print(f"Rearrange Attempts:   {self.rearrange_stats['rearrange_attempts']}")
        print(f"Rearrange Success:    {rearr_success_rate:.1f}%")
        print(f"Rearrange Applied:    {rearr_apply_rate:.1f}%")
        print(f"Avg Rearrange Value:  {avg_rearr_value:.4f}")
        print(f"Avg Unpack Depth:     {avg_unpack_depth:.2f}")
        print(f"MCTS Active Used:     {self.rearrange_stats['mcts_active_used']}")
        print(f"MCTS Fallback Used:   {self.rearrange_stats['mcts_fallback_used']}")
        print(f"Total Steps:         {self.total_steps}")
        print(f"Total Episodes:      {self.episode_count}")
        print(f"Epoch Time:          {epoch_elapsed:.2f}s ({steps_per_sec:.1f} steps/s)")
        print("Timing Breakdown (collect_steps):")
        print(f"  Action Select:     {timing['action_select_s']:.2f}s ({pct(timing['action_select_s']):.1f}%)")
        print(f"  Env Step:          {timing['env_step_s']:.2f}s ({pct(timing['env_step_s']):.1f}%)")
        print(f"  Store Transition:  {timing['store_transition_s']:.2f}s ({pct(timing['store_transition_s']):.1f}%)")
        print(f"  Episode Reset:     {timing['episode_reset_s']:.2f}s ({pct(timing['episode_reset_s']):.1f}%)")
        print(f"  Visualization:     {timing['visualization_s']:.2f}s ({pct(timing['visualization_s']):.1f}%)")
        print(f"{'='*70}\n")


def train(num_epochs=10, n_steps=2048, seed=42, device='cpu', dataset_type='random',
          debug_mask=False, debug_actions=False, vis_interval=10, vis_dir='outputs/visualizations',
          blf_only=False):
    """
    Main training function.
    
    Args:
        num_epochs (int): Number of training epochs
        n_steps (int): Steps per epoch
        max_items (int): Max items per episode
        seed (int): Random seed
        device (str): 'cpu' atau 'cuda'
        dataset_type (str): 'random' atau 'cutting_stock'
    """
    print("\n" + "="*70)
    print("3D BIN PACKING WITH PPO TRAINING")
    print("="*70 + "\n")
    
    # Initialize environment
    print("Initializing environment...")
    # 20-foot container: 60×24×26 (decimeter units ≈ 6m × 2.4m × 2.6m)
    env = ContainerEnv(
        seed=seed,
        dataset_type=dataset_type,
        container_length=60,    # 20 feet ≈ 6 meters
        container_width=24,     # 8 feet ≈ 2.4 meters
        container_height=26,    # 8.5 feet ≈ 2.6 meters
        layered_min_height=args.layered_min_height,
        layered_max_height=args.layered_max_height,
        perfect_pack_sigma=args.pp_sigma,
        perfect_pack_size_bias=args.pp_size_bias,
        perfect_pack_mean_ratio=args.pp_mean_ratio,
    )
    env.debug_mask_stats = bool(debug_mask)
    state, action_mask = env.reset()
    
    state_size = env.state_size
    action_size = env.action_size
    
    print(f"  State size: {state_size}")
    print(f"  Action size: {action_size}")
    print(f"  Container: {env.L} x {env.W} x {env.H}")
    print("")
    print(f"  Dataset type: {dataset_type}\n")

    # Dataset summary (after initial reset)
    if env.items:
        dims = np.array([get_item_dims(item) for item in env.items], dtype=np.float32)
        volumes = dims[:, 0] * dims[:, 1] * dims[:, 2]
        total_volume = float(np.sum(volumes))
        container_volume = float(env.L * env.W * env.H)
        util_percent = (total_volume / container_volume * 100.0) if container_volume > 0 else 0.0

        print("Dataset summary:")
        print(f"  Items generated: {len(env.items)}")
        print(f"  Volume utilization (items/box): {util_percent:6.2f}%")
        print(f"  L min/mean/max: {dims[:, 0].min():.0f}/{dims[:, 0].mean():.2f}/{dims[:, 0].max():.0f}")
        print(f"  W min/mean/max: {dims[:, 1].min():.0f}/{dims[:, 1].mean():.2f}/{dims[:, 1].max():.0f}")
        print(f"  H min/mean/max: {dims[:, 2].min():.0f}/{dims[:, 2].mean():.2f}/{dims[:, 2].max():.0f}")

        if env.dataset_type == 'perfect_pack_layered' and env.ground_truth_positions:
            layer_map = {}
            for item, (x, y, z) in zip(env.items, env.ground_truth_positions):
                _, _, h = get_item_dims(item)
                layer_map[z] = max(layer_map.get(z, 0), int(h))
            layer_heights = np.array(list(layer_map.values()), dtype=np.float32)
            if layer_heights.size > 0:
                print(f"  Layers: {len(layer_heights)}")
                print(
                    "  Layer height min/mean/max: "
                    f"{layer_heights.min():.0f}/{layer_heights.mean():.2f}/{layer_heights.max():.0f}"
                )
        print("")
    
    # Initialize PPO agent
    print("Initializing PPO agent...")
    ppo = PPO(
        state_size=state_size,
        action_size=action_size,
        L=env.L,  # Pass actual container dimensions
        W=env.W,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        device=device
    )
    print("  Done!\n")
    
    # Initialize training loop
    training_loop = TrainingLoop(
        env=env,
        ppo_agent=ppo,
        n_steps=n_steps,
        device=device,
        seed=seed,
        debug_actions=debug_actions,
        vis_interval=vis_interval,
        vis_dir=vis_dir,
        blf_only=blf_only,
    )
    
    # Training loop
    epoch_records = []
    try:
        for epoch in range(num_epochs):
            training_loop.train_epoch(num_epochs=3, log_frequency=1)

            # Record per-epoch summary for CSV reporting.
            stats = training_loop.logger.get_stats()
            rearr_attempts = max(training_loop.rearrange_stats['rearrange_attempts'], 1)
            epoch_records.append({
                'epoch': epoch + 1,
                'reward_mean_last100': stats.get('reward_mean', 0.0),
                'reward_std_last100': stats.get('reward_std', 0.0),
                'length_mean_last100': stats.get('length_mean', 0.0),
                'utilization_mean_last100': stats.get('utilization_mean', 0.0),
                'success_rate_mean_last100': stats.get('success_rate_mean', 0.0),
                'deadlocks': training_loop.rearrange_stats['deadlocks'],
                'rearrange_attempts': training_loop.rearrange_stats['rearrange_attempts'],
                'rearrange_success_rate': training_loop.rearrange_stats['rearrange_success'] / rearr_attempts,
                'rearrange_apply_rate': training_loop.rearrange_stats['rearrange_applied'] / rearr_attempts,
                'avg_rearrange_value': training_loop.rearrange_stats['rearrange_best_value_sum'] / rearr_attempts,
                'avg_unpack_depth': training_loop.rearrange_stats['rearrange_unpack_depth_sum'] / rearr_attempts,
                'mcts_fallback_used': training_loop.rearrange_stats['mcts_fallback_used'],
                'total_steps': training_loop.total_steps,
                'total_episodes': training_loop.episode_count,
            })
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                checkpoint_path = f"logs/training/checkpoint_epoch_{epoch+1}.pt"
                ppo.save_checkpoint(checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}\n")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70 + "\n")
    
    # Final statistics
    stats = training_loop.logger.get_stats()
    print("Final Statistics:")
    print(f"  Average Reward:    {stats.get('reward_mean', 0):.4f}")
    print(f"  Average Length:    {stats.get('length_mean', 0):.1f} steps")
    print(f"  Average Util:      {stats.get('utilization_mean', 0):.2f}%")
    print(f"  Average Success:   {stats.get('success_rate_mean', 0):.1f}%")
    print(f"  Total Steps:       {training_loop.total_steps}")
    print(f"  Total Episodes:    {training_loop.episode_count}\n")

    # Export epoch summary metrics to CSV.
    csv_path = Path('logs/training/training_epoch_metrics.csv')
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if epoch_records:
        with csv_path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(epoch_records[0].keys()))
            writer.writeheader()
            writer.writerows(epoch_records)
        print(f"Training metrics CSV saved: {csv_path}")
    
    return training_loop, ppo


if __name__ == "__main__":
    """Train PPO agent for 3D bin packing"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Train PPO agent for 3D bin packing')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--n_steps', type=int, default=2048, help='Steps per epoch')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cpu', help='Device: cpu or cuda')
    parser.add_argument('--dataset', type=str, default='perfect_pack_layered', help='Dataset: perfect_pack_layered only')
    parser.add_argument('--layered-min-height', type=int, default=2, help='Min layer height for perfect_pack_layered')
    parser.add_argument('--layered-max-height', type=int, default=6, help='Max layer height for perfect_pack_layered')
    parser.add_argument('--pp-sigma', type=float, default=4, help='Perfect pack Gaussian sigma')
    parser.add_argument('--pp-size-bias', type=float, default=3.0, help='Perfect pack size bias')
    parser.add_argument('--pp-mean-ratio', type=float, default=0.25, help='Perfect pack mean ratio')
    parser.add_argument('--debug-mask', action='store_true', help='Print action mask stats per step')
    parser.add_argument('--debug-actions', action='store_true', help='Print chosen action stats per step')
    parser.add_argument('--vis-interval', type=int, default=10, help='Save visualization every N episodes')
    parser.add_argument('--vis-dir', type=str, default='outputs/visualizations', help='Visualization output directory')
    parser.add_argument('--blf-only', action='store_true', help='Use Bottom-Left-Fill position selection')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("3D BIN PACKING WITH PPO TRAINING")
    print("="*70)
    if args.dataset != 'perfect_pack_layered':
        raise ValueError("Training hanya mendukung dataset 'perfect_pack_layered'.")

    print(
        f"Config: epochs={args.num_epochs}, steps={args.n_steps}, "
        f"dataset={args.dataset}, sigma={args.pp_sigma}, size_bias={args.pp_size_bias}, "
        f"mean_ratio={args.pp_mean_ratio}, layers={args.layered_min_height}-{args.layered_max_height}, "
        f"vis_interval={args.vis_interval}, blf_only={args.blf_only}\n"
    )
    
    # Train with parsed arguments
    training_loop, ppo = train(
        num_epochs=args.num_epochs,
        n_steps=args.n_steps,
        seed=args.seed,
        device=args.device,
        dataset_type=args.dataset,
        debug_mask=args.debug_mask,
        debug_actions=args.debug_actions,
        vis_interval=args.vis_interval,
        vis_dir=args.vis_dir,
        blf_only=args.blf_only,
    )
    
    print("\nTraining completed successfully!")