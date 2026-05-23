import numpy as np
import torch
import torch.nn.functional as F
import sys
import os
import csv
import multiprocessing as mp
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
from src.learning.agents.a3c import A3C
from src.learning.agents.shared_optim import SharedAdam
from src.learning.models.actor_critic import ActorCriticNetwork
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
    Training loop untuk A3C agent dengan 3D bin packing environment.
    
    Features:
    - Collect N steps dari environment
    - Compute returns menggunakan advantage
    - Update network dengan A3C
    - Log metrics: reward, utilization, episode length
    """
    
    def __init__(self, env, a3c_agent, n_steps=2048,
                 device='cpu', seed=None, debug_actions=False,
                 vis_interval=10, vis_dir='outputs/visualizations',
                 blf_only=False, repack_cooldown=1,
                 high_level_agent=None, high_level_optimizer=None,
                 candidate_top_k=128, mcts_budget=20, use_mcts_prob=0.0,
                 progress_interval=25, loop_id=None):
        """
        Initialize training loop.
        
        Args:
            env (ContainerEnv): Environment
            a3c_agent (A3C): A3C agent
            n_steps (int): Number of steps to collect per update
            device (str): 'cpu' atau 'cuda'
            seed (int): Random seed
        """
        self.env = env
        self.a3c = a3c_agent
        self.n_steps = n_steps
        self.device = device
        self.seed = seed
        self.debug_actions = bool(debug_actions)
        self.loop_id = loop_id

        # Hierarchical components for macro decision + candidate selection.
        if high_level_agent is None:
            self.high_level_agent = HighLevelAgent(input_dim=env.state_size).to(device)
        else:
            self.high_level_agent = high_level_agent.to(device)
        self.high_level_agent.train()  # Enable training mode

        if high_level_optimizer is None:
            self.high_level_optimizer = torch.optim.Adam(
                self.high_level_agent.parameters(),
                lr=1e-4
            )
        else:
            self.high_level_optimizer = high_level_optimizer
        
        self.candidate_generator = CandidateGenerator(env.L, env.W)
        self.mcts_budget = int(mcts_budget)
        if candidate_top_k is None:
            self.candidate_top_k = None
        else:
            candidate_top_k = int(candidate_top_k)
            self.candidate_top_k = None if candidate_top_k <= 0 else candidate_top_k
        self.use_mcts_prob = float(use_mcts_prob)
        
        self.logger = TrainingLogger()
        self.progress_interval = max(1, int(progress_interval))
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
        self.deadlock_streak = 0
        self.repack_cooldown = max(1, int(repack_cooldown))
        
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
            - Given strategy, A3C agent selects WHERE to place (position)
            - Uses ActorCriticNetwork + action masking
            - Outputs position + log_prob + value estimate
        
        COMPLETE DECISION FLOW:
            State -> HighLevelAgent -> Strategy (e.g., "rotate 90°")
                  -> CandidateGenerator -> Filter valid positions
                  -> A3C Agent -> Select position (e.g., "place at x=10, y=5")
                  -> Environment.step() -> Execute & get reward
        
        DEADLOCK HANDLING:
            - If A3C selects invalid move (deadlock), trigger fallback:
              * Try attempt_repack() first
              * If still stuck, use MCTS search for escape
        
        TRAINING UPDATES:
            - A3C loss: Policy + value loss on collected trajectories
            - HighLevelAgent: Ready for future hierarchical training
        
        Args:
            state: Current state (height_map + item dimensions)
            action_mask: Legal actions mask (1 = valid, 0 = invalid)
            sample_strategy (bool): If True, sample strategies (training mode)
                                  If False, use greedy selection (eval mode)
        
        Returns:
            tuple: (action, log_prob, value, effective_action_mask, strategy_info)
                   - action: Position index selected by A3C
                   - log_prob: A3C's log probability of action
                   - value: A3C's value estimate of current state
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
            top_k=self.candidate_top_k
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
            self.deadlock_streak = 0
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

            # Use A3C agent to select position with masked action space
            a3c_action, a3c_log_prob, a3c_value = self.a3c.select_action(
                policy_state, effective_mask
            )

            if self.use_mcts_prob > 0.0:
                final_action, log_prob, value, mcts_used = self._blend_a3c_mcts_decision(
                    policy_state,
                    policy_action_mask,
                    effective_mask,
                    a3c_action,
                    a3c_log_prob,
                    a3c_value,
                    use_mcts_prob=self.use_mcts_prob,
                )
                if mcts_used:
                    self.rearrange_stats['mcts_active_used'] += 1
                return final_action, log_prob, value, effective_mask, strategy_info, policy_state

            # Return action from A3C - let the policy learn good placement
            return a3c_action, a3c_log_prob, a3c_value, effective_mask, strategy_info, policy_state

        self.rearrange_stats['deadlocks'] += 1
        self.deadlock_streak += 1

        if self.repack_cooldown > 1 and (self.deadlock_streak % self.repack_cooldown) != 0:
            valid_actions = np.where(np.asarray(policy_action_mask[:-1]) > 0)[0]
            if len(valid_actions) > 0:
                action = int(valid_actions.min())
            else:
                action = self.env.L * self.env.W
            log_prob, value = self._compute_logprob_value_for_action(
                policy_state, policy_action_mask, action
            )
            return action, log_prob, value, np.asarray(policy_action_mask, dtype=np.float32), strategy_info, policy_state

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
                    top_k=self.candidate_top_k
                )
                effective_mask = np.zeros_like(policy_action_mask, dtype=np.float32)
                for idx in candidate_actions:
                    if 0 <= idx < self.env.L * self.env.W and policy_action_mask[idx] > 0:
                        effective_mask[idx] = 1.0

                if np.sum(effective_mask[:-1]) > 0:
                    action, log_prob, value = self.a3c.select_action(policy_state, effective_mask)
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
                top_k=self.candidate_top_k
            )
            effective_mask = np.zeros_like(policy_action_mask, dtype=np.float32)
            for idx in candidate_actions:
                if 0 <= idx < self.env.L * self.env.W and policy_action_mask[idx] > 0:
                    effective_mask[idx] = 1.0

            if np.sum(effective_mask[:-1]) > 0:
                action, log_prob, value = self.a3c.select_action(policy_state, effective_mask)
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
        """Compute log probability and value for a fixed action under current A3C policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mask_tensor = torch.FloatTensor(action_mask).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, value = self.a3c.network(state_tensor)

        masked_logits = self.a3c.mask_logits(logits, mask_tensor)
        probs = F.softmax(masked_logits, dim=-1)
        probs = torch.where(torch.isnan(probs), torch.zeros_like(probs), probs)

        probs_sum = probs.sum(dim=-1, keepdim=True)
        probs = torch.where(probs_sum > 0, probs / probs_sum, probs)

        if probs[0, action] <= 0:
            log_prob = torch.tensor(-20.0, device=self.device)
        else:
            log_prob = torch.log(probs[0, action] + 1e-12)

        return log_prob.item(), value.item()
    
    def _blend_a3c_mcts_decision(self, state, action_mask, effective_mask, a3c_action, a3c_log_prob, a3c_value, use_mcts_prob=0.2):
        """
        Blend A3C action with MCTS search results.
        
        Args:
            state: Current state
            action_mask: Full action mask
            effective_mask: Filtered action mask from candidates
            a3c_action: Action selected by A3C
            a3c_log_prob: Log probability from A3C
            a3c_value: Value estimate from A3C
            use_mcts_prob: Probability of consulting MCTS (default 0.2 = 20%)
            
        Returns:
            tuple: (final_action, final_log_prob, final_value, mcts_used)
        """
        # Decide whether to use MCTS
        use_mcts = np.random.random() < use_mcts_prob
        mcts_used = False
        
        if not use_mcts:
            return a3c_action, a3c_log_prob, a3c_value, False
        
        try:
            # Run MCTS search with limited budget
            mcts = MCTS(self.env, budget=self.mcts_budget // 2)  # Use half budget for faster computation
            mcts_result = mcts.search(state, action_mask, depth_limit=5)
            mcts_action = int(mcts_result['best_action'])
            
            # Check if MCTS action is valid
            if 0 <= mcts_action < len(action_mask) and action_mask[mcts_action] > 0:
                # MCTS found a good action - prefer MCTS in ~70% of the time, blend with A3C otherwise
                if np.random.random() < 0.7:
                    # Use MCTS action
                    final_action = mcts_action
                    log_prob, value = self._compute_logprob_value_for_action(state, action_mask, mcts_action)
                    mcts_used = True
                else:
                    # Blend: use A3C action but let A3C know about MCTS preference
                    final_action = a3c_action
                    log_prob, value = a3c_log_prob, a3c_value
                    mcts_used = False
            else:
                # MCTS action was invalid, stick with A3C
                final_action = a3c_action
                log_prob, value = a3c_log_prob, a3c_value
                mcts_used = False
                
            return final_action, log_prob, value, mcts_used
            
        except Exception as e:
            # If MCTS fails, fall back to A3C
            return a3c_action, a3c_log_prob, a3c_value, False
    
    def _update_high_level_agent(self, strategy_buffer, num_epochs=3):
        """
        Update HighLevelAgent using strategy buffer with policy gradient.
        
        DISABLED: Currently has gradient computation issues (in-place modifications).
                 Will be re-enabled after proper gradient graph management.
        
        Args:
            strategy_buffer (list): List of dicts with strategy info
            num_epochs (int): Number of update passes
        """
        if not strategy_buffer:
            return

        states = torch.FloatTensor(
            np.stack([entry['state'] for entry in strategy_buffer])
        ).to(self.device)
        actions = torch.LongTensor(
            [int(entry['strategy']) for entry in strategy_buffer]
        ).to(self.device)
        rewards = torch.FloatTensor(
            [float(entry['reward']) for entry in strategy_buffer]
        ).to(self.device)

        if rewards.numel() > 1:
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        else:
            advantages = rewards

        for _ in range(int(num_epochs)):
            high_output = self.high_level_agent(states)
            logits = high_output['strategy_logits']
            log_probs = F.log_softmax(logits, dim=-1)
            selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

            policy_loss = -(selected_log_probs * advantages.detach()).mean()

            self.high_level_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.high_level_agent.parameters(), 0.5)
            self.high_level_optimizer.step()
    
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

        print(
            f"Collect rollout | loop={self.loop_id if self.loop_id is not None else 'main'} | "
            f"target_steps={n_steps} | "
            f"progress_interval={self.progress_interval} | "
            f"episode={self.episode_count} | total_steps={self.total_steps}",
            flush=True,
        )
        
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
            
            # Store low-level transition
            store_start = time.perf_counter()
            self.a3c.store_transition(
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
                'state': np.asarray(state, dtype=np.float32),
                'strategy': strategy_info['strategy'],
                'reward': reward,
            })
            
            steps_collected += 1
            self.total_steps += 1

            if self.total_steps % self.progress_interval == 0:
                print(
                    f"Step progress | loop={self.loop_id if self.loop_id is not None else 'main'} | "
                    f"total_steps={self.total_steps} | "
                    f"rollout={steps_collected}/{n_steps} | "
                    f"episodes={self.episode_count} | "
                    f"current_episode_steps={self.env.episode_length} | "
                    f"current_episode_reward={self.env.episode_reward:.4f}",
                    flush=True,
                )
            
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
            num_epochs (int): Jumlah A3C update epochs (unused)
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
        
        # Update A3C network
        print(f"Updating A3C network... ", end='', flush=True)
        self.a3c.update(next_value=next_value)
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
          debug_mask=False, debug_actions=False, vis_interval=50, vis_dir='outputs/visualizations',
          blf_only=False, fast_stability_mask=False, repack_cooldown=1,
          layered_min_height=2, layered_max_height=6,
          pp_sigma=4, pp_size_bias=3.0, pp_mean_ratio=0.25,
          checkpoint_steps=0, candidate_top_k=128, mcts_budget=20, use_mcts_prob=0.0,
          early_stop=True, early_stop_metric='utilization_mean',
          early_stop_patience=10, early_stop_min_delta=0.1, early_stop_min_epochs=10):
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
    print("3D BIN PACKING WITH A3C TRAINING")
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
        layered_min_height=layered_min_height,
        layered_max_height=layered_max_height,
        perfect_pack_sigma=pp_sigma,
        perfect_pack_size_bias=pp_size_bias,
        perfect_pack_mean_ratio=pp_mean_ratio,
        fast_stability_mask=fast_stability_mask,
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
    
    # Initialize A3C agent
    print("Initializing A3C agent...")
    a3c = A3C(
        state_size=state_size,
        action_size=action_size,
        L=env.L,  # Pass actual container dimensions
        W=env.W,
        learning_rate=3e-4,
        gamma=0.99,
        entropy_coef=0.01,
        value_coef=0.5,
        device=device
    )
    print("  Done!\n")
    
    # Initialize training loop
    training_loop = TrainingLoop(
        env=env,
        a3c_agent=a3c,
        n_steps=n_steps,
        device=device,
        seed=seed,
        debug_actions=debug_actions,
        vis_interval=vis_interval,
        vis_dir=vis_dir,
        blf_only=blf_only,
        repack_cooldown=repack_cooldown,
        candidate_top_k=candidate_top_k,
        mcts_budget=mcts_budget,
        use_mcts_prob=use_mcts_prob,
    )
    
    # Training loop
    epoch_records = []
    next_checkpoint = checkpoint_steps if checkpoint_steps > 0 else None
    checkpoint_dir = Path('logs/training')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_metric = None
    no_improve_epochs = 0

    def get_metric(stats, metric_name):
        value = stats.get(metric_name)
        return None if value is None else float(value)

    try:
        for epoch in range(num_epochs):
            training_loop.train_epoch(num_epochs=3, log_frequency=1)

            # Record per-epoch summary for CSV reporting.
            stats = training_loop.logger.get_stats()
            current_metric = get_metric(stats, early_stop_metric)
            if early_stop and current_metric is not None:
                if best_metric is None or current_metric > (best_metric + early_stop_min_delta):
                    best_metric = current_metric
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1

                if (epoch + 1) >= int(early_stop_min_epochs) and no_improve_epochs >= int(early_stop_patience):
                    print(
                        f"Early stop: no improvement in {early_stop_patience} epochs "
                        f"(metric={early_stop_metric})."
                    )
                    break
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
                a3c.save_checkpoint(checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}\n")

            if next_checkpoint is not None:
                while training_loop.total_steps >= next_checkpoint:
                    ckpt_path = checkpoint_dir / f"checkpoint_step_{next_checkpoint}.pt"
                    a3c.save_checkpoint(str(ckpt_path))
                    hl_path = checkpoint_dir / f"checkpoint_high_level_step_{next_checkpoint}.pt"
                    torch.save(training_loop.high_level_agent.state_dict(), hl_path)
                    print(f"Checkpoint saved: {ckpt_path}")
                    next_checkpoint += checkpoint_steps
    
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
    
    return training_loop, a3c


def _a3c_worker(worker_id, shared_model, shared_optimizer,
                shared_high_level, shared_high_optimizer,
                config, global_step, global_episode, total_steps,
                last_checkpoint_step, checkpoint_lock):
    torch.set_num_threads(1)
    worker_seed = config['seed'] + worker_id * 1000

    env = ContainerEnv(
        seed=worker_seed,
        dataset_type=config['dataset_type'],
        container_length=60,
        container_width=24,
        container_height=26,
        layered_min_height=config['layered_min_height'],
        layered_max_height=config['layered_max_height'],
        perfect_pack_sigma=config['pp_sigma'],
        perfect_pack_size_bias=config['pp_size_bias'],
        perfect_pack_mean_ratio=config['pp_mean_ratio'],
        fast_stability_mask=config['fast_stability_mask'],
    )
    env.debug_mask_stats = bool(config['debug_mask'])

    a3c = A3C(
        state_size=env.state_size,
        action_size=env.action_size,
        L=env.L,
        W=env.W,
        learning_rate=config['learning_rate'],
        gamma=config['gamma'],
        entropy_coef=config['entropy_coef'],
        value_coef=config['value_coef'],
        device=config['device'],
        network=shared_model,
        optimizer=shared_optimizer,
    )

    loop = TrainingLoop(
        env=env,
        a3c_agent=a3c,
        n_steps=config['rollout_steps'],
        device=config['device'],
        seed=worker_seed,
        debug_actions=config['debug_actions'],
        vis_interval=10**9,
        vis_dir=config['vis_dir'],
        blf_only=config['blf_only'],
        repack_cooldown=config['repack_cooldown'],
        high_level_agent=shared_high_level,
        high_level_optimizer=shared_high_optimizer,
        candidate_top_k=config['candidate_top_k'],
        mcts_budget=config['mcts_budget'],
        use_mcts_prob=config['use_mcts_prob'],
    )

    while True:
        with global_step.get_lock():
            if global_step.value >= total_steps:
                break

        episode_info, next_value, strategy_buffer, _ = loop.collect_steps(config['rollout_steps'])

        loop._update_high_level_agent(strategy_buffer)
        a3c.update(next_value=next_value)

        with global_step.get_lock():
            global_step.value += config['rollout_steps']

        if worker_id == 0 and config['checkpoint_steps'] > 0:
            with checkpoint_lock:
                if global_step.value - last_checkpoint_step.value >= config['checkpoint_steps']:
                    ckpt_step = global_step.value
                    last_checkpoint_step.value = ckpt_step
                    ckpt_path = Path(config['checkpoint_dir']) / f"a3c_async_step_{ckpt_step}.pt"
                    torch.save(shared_model.state_dict(), ckpt_path)
                    hl_path = Path(config['checkpoint_dir']) / f"a3c_async_high_level_step_{ckpt_step}.pt"
                    torch.save(shared_high_level.state_dict(), hl_path)

        if episode_info.get('lengths'):
            with global_episode.get_lock():
                global_episode.value += len(episode_info['lengths'])


def train_async(num_epochs=10, n_steps=2048, seed=42, device='cpu', dataset_type='random',
                debug_mask=False, debug_actions=False, vis_interval=50, vis_dir='outputs/visualizations',
                blf_only=False, fast_stability_mask=False, repack_cooldown=1,
                async_workers=4, rollout_steps=32, checkpoint_steps=0,
                learning_rate=3e-4, gamma=0.99, entropy_coef=0.01, value_coef=0.5,
                layered_min_height=2, layered_max_height=6,
                pp_sigma=4, pp_size_bias=3.0, pp_mean_ratio=0.25,
                candidate_top_k=128, mcts_budget=20, use_mcts_prob=0.0,
                early_stop=True, early_stop_metric='utilization_mean',
                early_stop_patience=10, early_stop_min_delta=0.1, early_stop_min_epochs=10):
    if device != 'cpu':
        print("Async A3C uses CPU workers; switching device to cpu.")
        device = 'cpu'

    # Build a temporary env to infer sizes
    env = ContainerEnv(
        seed=seed,
        dataset_type=dataset_type,
        container_length=60,
        container_width=24,
        container_height=26,
        layered_min_height=layered_min_height,
        layered_max_height=layered_max_height,
        perfect_pack_sigma=pp_sigma,
        perfect_pack_size_bias=pp_size_bias,
        perfect_pack_mean_ratio=pp_mean_ratio,
        fast_stability_mask=fast_stability_mask,
    )
    state_size = env.state_size
    action_size = env.action_size

    shared_model = ActorCriticNetwork(L=env.L, W=env.W, action_size=action_size).to(device)
    shared_model.share_memory()
    shared_optimizer = SharedAdam(shared_model.parameters(), lr=learning_rate)

    shared_high_level = HighLevelAgent(input_dim=state_size).to(device)
    shared_high_level.share_memory()
    shared_high_optimizer = SharedAdam(shared_high_level.parameters(), lr=1e-4)

    total_steps = num_epochs * n_steps
    global_step = mp.Value('i', 0)
    global_episode = mp.Value('i', 0)

    checkpoint_dir = Path('logs/training')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config = {
        'seed': seed,
        'dataset_type': dataset_type,
        'layered_min_height': layered_min_height,
        'layered_max_height': layered_max_height,
        'pp_sigma': pp_sigma,
        'pp_size_bias': pp_size_bias,
        'pp_mean_ratio': pp_mean_ratio,
        'fast_stability_mask': fast_stability_mask,
        'debug_mask': debug_mask,
        'debug_actions': debug_actions,
        'vis_dir': vis_dir,
        'blf_only': blf_only,
        'repack_cooldown': repack_cooldown,
        'rollout_steps': int(rollout_steps),
        'learning_rate': learning_rate,
        'gamma': gamma,
        'entropy_coef': entropy_coef,
        'value_coef': value_coef,
        'device': device,
        'checkpoint_steps': int(checkpoint_steps),
        'checkpoint_dir': str(checkpoint_dir),
        'candidate_top_k': candidate_top_k,
        'mcts_budget': mcts_budget,
        'use_mcts_prob': use_mcts_prob,
    }

    ctx = mp.get_context('spawn')
    processes = []
    last_checkpoint_step = mp.Value('i', 0)
    checkpoint_lock = ctx.Lock()
    for wid in range(int(async_workers)):
        p = ctx.Process(
            target=_a3c_worker,
            args=(
                wid,
                shared_model,
                shared_optimizer,
                shared_high_level,
                shared_high_optimizer,
                config,
                global_step,
                global_episode,
                total_steps,
                last_checkpoint_step,
                checkpoint_lock,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    return shared_model


def train_batched(num_epochs=10, n_steps=2048, seed=42, device='cpu', dataset_type='random',
                  debug_mask=False, debug_actions=False, vis_interval=50, vis_dir='outputs/visualizations',
                  blf_only=False, fast_stability_mask=False, repack_cooldown=1,
                  batched_envs=4, rollout_steps=32,
                  learning_rate=3e-4, gamma=0.99, entropy_coef=0.01, value_coef=0.5,
                  layered_min_height=2, layered_max_height=6,
                  pp_sigma=4, pp_size_bias=3.0, pp_mean_ratio=0.25,
                  checkpoint_steps=0, candidate_top_k=128, mcts_budget=20, use_mcts_prob=0.0,
                  early_stop=True, early_stop_metric='utilization_mean',
                  early_stop_patience=10, early_stop_min_delta=0.1, early_stop_min_epochs=10):
    envs = []
    for idx in range(int(batched_envs)):
        env = ContainerEnv(
            seed=seed + idx * 1000,
            dataset_type=dataset_type,
            container_length=60,
            container_width=24,
            container_height=26,
            layered_min_height=layered_min_height,
            layered_max_height=layered_max_height,
            perfect_pack_sigma=pp_sigma,
            perfect_pack_size_bias=pp_size_bias,
            perfect_pack_mean_ratio=pp_mean_ratio,
            fast_stability_mask=fast_stability_mask,
        )
        env.debug_mask_stats = bool(debug_mask)
        envs.append(env)

    state_size = envs[0].state_size
    action_size = envs[0].action_size

    a3c = A3C(
        state_size=state_size,
        action_size=action_size,
        L=envs[0].L,
        W=envs[0].W,
        learning_rate=learning_rate,
        gamma=gamma,
        entropy_coef=entropy_coef,
        value_coef=value_coef,
        device=device,
    )

    shared_high_level = HighLevelAgent(input_dim=state_size).to(device)
    high_optimizer = torch.optim.Adam(shared_high_level.parameters(), lr=1e-4)

    loops = []
    for env_idx, env in enumerate(envs):
        loop = TrainingLoop(
            env=env,
            a3c_agent=a3c,
            n_steps=rollout_steps,
            device=device,
            seed=seed,
            debug_actions=debug_actions,
            vis_interval=10**9,
            vis_dir=vis_dir,
            blf_only=blf_only,
            repack_cooldown=repack_cooldown,
            high_level_agent=shared_high_level,
            high_level_optimizer=high_optimizer,
            candidate_top_k=candidate_top_k,
            mcts_budget=mcts_budget,
            use_mcts_prob=use_mcts_prob,
            loop_id=env_idx,
        )
        loops.append(loop)

    total_steps = num_epochs * n_steps
    steps_done = 0
    next_epoch_step = n_steps
    epoch_index = 0
    best_metric = None
    no_improve_epochs = 0
    next_checkpoint = checkpoint_steps if checkpoint_steps > 0 else None
    checkpoint_dir = Path('logs/training')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    while steps_done < total_steps:
        for loop in loops:
            if steps_done >= total_steps:
                break
            episode_info, next_value, strategy_buffer, _ = loop.collect_steps(rollout_steps)
            loop._update_high_level_agent(strategy_buffer)
            a3c.update(next_value=next_value)
            steps_done += rollout_steps
            print(
                f"Global progress | steps_done={steps_done}/{total_steps} | "
                f"last_loop={loop.loop_id} | episodes={[l.episode_count for l in loops]}",
                flush=True,
            )

            if steps_done >= next_epoch_step:
                epoch_index += 1
                next_epoch_step += n_steps

                stats_list = [l.logger.get_stats() for l in loops if l.logger.get_stats()]
                if stats_list:
                    avg_metric = float(
                        np.mean([s.get(early_stop_metric, 0.0) for s in stats_list])
                    )
                    if early_stop:
                        if best_metric is None or avg_metric > (best_metric + early_stop_min_delta):
                            best_metric = avg_metric
                            no_improve_epochs = 0
                        else:
                            no_improve_epochs += 1

                        if epoch_index >= int(early_stop_min_epochs) and no_improve_epochs >= int(early_stop_patience):
                            print(
                                f"Early stop: no improvement in {early_stop_patience} epochs "
                                f"(metric={early_stop_metric})."
                            )
                            steps_done = total_steps
                            break

            if next_checkpoint is not None:
                while steps_done >= next_checkpoint:
                    ckpt_path = checkpoint_dir / f"batched_step_{next_checkpoint}.pt"
                    a3c.save_checkpoint(str(ckpt_path))
                    hl_path = checkpoint_dir / f"batched_high_level_step_{next_checkpoint}.pt"
                    torch.save(shared_high_level.state_dict(), hl_path)
                    print(f"Checkpoint saved: {ckpt_path}")
                    next_checkpoint += checkpoint_steps

    return a3c


if __name__ == "__main__":
    """Train A3C agent for 3D bin packing"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Train A3C agent for 3D bin packing')
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
    parser.add_argument('--vis-interval', type=int, default=50, help='Save visualization every N episodes')
    parser.add_argument('--vis-dir', type=str, default='outputs/visualizations', help='Visualization output directory')
    parser.add_argument('--blf-only', action='store_true', help='Use Bottom-Left-Fill position selection')
    parser.add_argument('--fast-stability-mask', action='store_true', default=False,
                        help='Use fast stability mask (skip per-position LBCP)')
    parser.add_argument('--repack-cooldown', type=int, default=1,
                        help='Run repack/MCTS every N deadlocks (1 = always)')
    parser.add_argument('--checkpoint-steps', type=int, default=0,
                        help='Checkpoint interval (steps) for sync/batched modes')
    parser.add_argument('--async-workers', type=int, default=1,
                        help='Number of async A3C workers (1 = sync training)')
    parser.add_argument('--rollout-steps', type=int, default=32,
                        help='Rollout steps per async worker update')
    parser.add_argument('--batched-envs', type=int, default=1,
                        help='Number of envs for single-process batched training')
    parser.add_argument('--async-checkpoint-steps', type=int, default=0,
                        help='Checkpoint interval (steps) for async mode')
    parser.add_argument('--candidate-top-k', type=int, default=128,
                        help='Top-k candidates after macro filtering (<=0 for all)')
    parser.add_argument('--mcts-budget', type=int, default=20,
                        help='MCTS simulation budget for fallback/planning')
    parser.add_argument('--use-mcts-prob', type=float, default=0.0,
                        help='Probability to consult MCTS during regular action selection')
    parser.add_argument('--no-early-stop', action='store_true',
                        help='Disable early stopping')
    parser.add_argument('--early-stop-metric', type=str, default='utilization_mean',
                        help='Metric for early stopping (utilization_mean or reward_mean)')
    parser.add_argument('--early-stop-patience', type=int, default=10,
                        help='Epochs to wait without improvement before stopping')
    parser.add_argument('--early-stop-min-delta', type=float, default=0.1,
                        help='Minimum improvement to reset early stop counter')
    parser.add_argument('--early-stop-min-epochs', type=int, default=10,
                        help='Minimum epochs before early stopping can trigger')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("3D BIN PACKING WITH A3C TRAINING")
    print("="*70)
    if args.dataset != 'perfect_pack_layered':
        raise ValueError("Training hanya mendukung dataset 'perfect_pack_layered'.")

    print(
        f"Config: epochs={args.num_epochs}, steps={args.n_steps}, "
        f"dataset={args.dataset}, sigma={args.pp_sigma}, size_bias={args.pp_size_bias}, "
        f"mean_ratio={args.pp_mean_ratio}, layers={args.layered_min_height}-{args.layered_max_height}, "
        f"vis_interval={args.vis_interval}, blf_only={args.blf_only}, "
        f"fast_mask={args.fast_stability_mask}, repack_cooldown={args.repack_cooldown}, "
        f"async_workers={args.async_workers}, rollout_steps={args.rollout_steps}, "
        f"batched_envs={args.batched_envs}, async_ckpt_steps={args.async_checkpoint_steps}, "
        f"ckpt_steps={args.checkpoint_steps}, candidate_top_k={args.candidate_top_k}, "
        f"mcts_budget={args.mcts_budget}, use_mcts_prob={args.use_mcts_prob}, "
        f"early_stop={not args.no_early_stop}, early_stop_metric={args.early_stop_metric}, "
        f"early_stop_patience={args.early_stop_patience}\n"
    )
    
    # Train with parsed arguments
    if args.async_workers > 1:
        train_async(
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
            fast_stability_mask=args.fast_stability_mask,
            repack_cooldown=args.repack_cooldown,
            async_workers=args.async_workers,
            rollout_steps=args.rollout_steps,
            checkpoint_steps=args.async_checkpoint_steps,
            layered_min_height=args.layered_min_height,
            layered_max_height=args.layered_max_height,
            pp_sigma=args.pp_sigma,
            pp_size_bias=args.pp_size_bias,
            pp_mean_ratio=args.pp_mean_ratio,
            candidate_top_k=args.candidate_top_k,
            mcts_budget=args.mcts_budget,
            use_mcts_prob=args.use_mcts_prob,
            early_stop=not args.no_early_stop,
            early_stop_metric=args.early_stop_metric,
            early_stop_patience=args.early_stop_patience,
            early_stop_min_delta=args.early_stop_min_delta,
            early_stop_min_epochs=args.early_stop_min_epochs,
        )
    elif args.batched_envs > 1:
        train_batched(
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
            fast_stability_mask=args.fast_stability_mask,
            repack_cooldown=args.repack_cooldown,
            batched_envs=args.batched_envs,
            rollout_steps=args.rollout_steps,
            layered_min_height=args.layered_min_height,
            layered_max_height=args.layered_max_height,
            pp_sigma=args.pp_sigma,
            pp_size_bias=args.pp_size_bias,
            pp_mean_ratio=args.pp_mean_ratio,
            checkpoint_steps=args.checkpoint_steps,
            candidate_top_k=args.candidate_top_k,
            mcts_budget=args.mcts_budget,
            use_mcts_prob=args.use_mcts_prob,
            early_stop=not args.no_early_stop,
            early_stop_metric=args.early_stop_metric,
            early_stop_patience=args.early_stop_patience,
            early_stop_min_delta=args.early_stop_min_delta,
            early_stop_min_epochs=args.early_stop_min_epochs,
        )
    else:
        training_loop, a3c = train(
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
            fast_stability_mask=args.fast_stability_mask,
            repack_cooldown=args.repack_cooldown,
            layered_min_height=args.layered_min_height,
            layered_max_height=args.layered_max_height,
            pp_sigma=args.pp_sigma,
            pp_size_bias=args.pp_size_bias,
            pp_mean_ratio=args.pp_mean_ratio,
            checkpoint_steps=args.checkpoint_steps,
            candidate_top_k=args.candidate_top_k,
            mcts_budget=args.mcts_budget,
            use_mcts_prob=args.use_mcts_prob,
            early_stop=not args.no_early_stop,
            early_stop_metric=args.early_stop_metric,
            early_stop_patience=args.early_stop_patience,
            early_stop_min_delta=args.early_stop_min_delta,
            early_stop_min_epochs=args.early_stop_min_epochs,
        )
    
    print("\nTraining completed successfully!")
