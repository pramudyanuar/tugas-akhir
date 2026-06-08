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
from src.core.stability_validator import StabilityValidator
from src.learning.models.high_level_agent import HighLevelAgent
from src.learning.agents.a3c import A3C
from src.learning.agents.shared_optim import SharedAdam
from src.learning.models.actor_critic import ActorCriticNetwork
from src.planning.mcts import MCTS
from src.planning.high_level_search import HighLevelSearcher
from visualization import ContainerVisualizer
from src.utils.metrics import Metrics
from src.utils.item_utils import get_item_dims
from src.utils.logger import create_logger


class TrainingLogger:
    """
    Logger untuk training metrics.
    """
    
    def __init__(self, log_dir='logs/training', tb_log_dir='logs/tensorboard', tb_experiment=None):
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
        self.a3c_policy_losses = []
        self.a3c_value_losses = []
        self.a3c_entropies = []
        self.a3c_total_losses = []
        self.high_level_policy_losses = []
        self.tb_logger = None
        if tb_experiment:
            self.tb_logger = create_logger(log_dir=tb_log_dir, experiment_name=tb_experiment)

    def _log_scalar(self, tag, value, step):
        if self.tb_logger is not None:
            self.tb_logger.log_scalar(tag, value, step)
        
    def log_episode(self, episode_reward, episode_length, utilization,
                    items_placed, total_items, step=None, buffer_stats=None):
        """Log episode metrics."""
        success_rate = (items_placed / total_items * 100) if total_items > 0 else 0.0
        
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.episode_utilizations.append(utilization)
        self.episode_success_rates.append(success_rate)

        if self.tb_logger is not None:
            log_step = int(step) if step is not None else len(self.episode_rewards)
            self._log_scalar('episode/reward', episode_reward, log_step)
            self._log_scalar('episode/length', episode_length, log_step)
            self._log_scalar('episode/utilization', utilization, log_step)
            self._log_scalar('episode/success_rate', success_rate, log_step)
            self._log_scalar('episode/items_placed', items_placed, log_step)
            self._log_scalar('episode/total_items', total_items, log_step)
            if buffer_stats is not None:
                for k, v in buffer_stats.items():
                    self._log_scalar(f'buffer/{k}', v, log_step)

    def log_update(self, a3c_loss=None, high_level_loss=None, step=None):
        """Log loss values from update steps."""
        if a3c_loss:
            self.a3c_policy_losses.append(float(a3c_loss.get('policy_loss', 0.0)))
            self.a3c_value_losses.append(float(a3c_loss.get('value_loss', 0.0)))
            self.a3c_entropies.append(float(a3c_loss.get('entropy', 0.0)))
            self.a3c_total_losses.append(float(a3c_loss.get('total_loss', 0.0)))
        if high_level_loss:
            self.high_level_policy_losses.append(float(high_level_loss.get('policy_loss', 0.0)))

        if self.tb_logger is not None:
            log_step = int(step) if step is not None else len(self.a3c_total_losses)
            if a3c_loss:
                self._log_scalar('loss/a3c_policy', a3c_loss.get('policy_loss', 0.0), log_step)
                self._log_scalar('loss/a3c_value', a3c_loss.get('value_loss', 0.0), log_step)
                self._log_scalar('loss/a3c_entropy', a3c_loss.get('entropy', 0.0), log_step)
                self._log_scalar('loss/a3c_total', a3c_loss.get('total_loss', 0.0), log_step)
            if high_level_loss:
                self._log_scalar('loss/high_level_policy', high_level_loss.get('policy_loss', 0.0), log_step)

    def log_epoch_summary(self, stats, rearrange_stats, step):
        """Log epoch-level summary metrics to TensorBoard."""
        if self.tb_logger is None:
            return
        log_step = int(step)
        self._log_scalar('epoch/reward_mean', stats.get('reward_mean', 0.0), log_step)
        self._log_scalar('epoch/reward_std', stats.get('reward_std', 0.0), log_step)
        self._log_scalar('epoch/length_mean', stats.get('length_mean', 0.0), log_step)
        self._log_scalar('epoch/utilization_mean', stats.get('utilization_mean', 0.0), log_step)
        self._log_scalar('epoch/success_rate_mean', stats.get('success_rate_mean', 0.0), log_step)
        self._log_scalar('epoch/a3c_policy_loss_mean', stats.get('a3c_policy_loss_mean', 0.0), log_step)
        self._log_scalar('epoch/a3c_value_loss_mean', stats.get('a3c_value_loss_mean', 0.0), log_step)
        self._log_scalar('epoch/a3c_entropy_mean', stats.get('a3c_entropy_mean', 0.0), log_step)
        self._log_scalar('epoch/a3c_total_loss_mean', stats.get('a3c_total_loss_mean', 0.0), log_step)
        self._log_scalar('epoch/high_level_policy_loss_mean', stats.get('high_level_policy_loss_mean', 0.0), log_step)

        rearr_attempts = max(int(rearrange_stats.get('rearrange_attempts', 0)), 1)
        self._log_scalar('epoch/deadlocks', rearrange_stats.get('deadlocks', 0), log_step)
        self._log_scalar('epoch/rearrange_attempts', rearrange_stats.get('rearrange_attempts', 0), log_step)
        self._log_scalar(
            'epoch/rearrange_success_rate',
            float(rearrange_stats.get('rearrange_success', 0)) / rearr_attempts,
            log_step,
        )
        self._log_scalar(
            'epoch/rearrange_apply_rate',
            float(rearrange_stats.get('rearrange_applied', 0)) / rearr_attempts,
            log_step,
        )
        self._log_scalar(
            'epoch/avg_rearrange_value',
            float(rearrange_stats.get('rearrange_best_value_sum', 0.0)) / rearr_attempts,
            log_step,
        )
        self._log_scalar(
            'epoch/avg_unpack_depth',
            float(rearrange_stats.get('rearrange_unpack_depth_sum', 0.0)) / rearr_attempts,
            log_step,
        )
        self._log_scalar('epoch/mcts_active_used', rearrange_stats.get('mcts_active_used', 0), log_step)
        self._log_scalar('epoch/mcts_fallback_used', rearrange_stats.get('mcts_fallback_used', 0), log_step)
        self.tb_logger.flush()
    
    def get_stats(self, last_n=100):
        """Get training statistics over the last N episodes."""
        if not self.episode_rewards:
            return {}

        window = min(int(last_n), len(self.episode_rewards))
        if window <= 0:
            return {}

        stats = {
            'reward_mean': np.mean(self.episode_rewards[-window:]),
            'reward_std': np.std(self.episode_rewards[-window:]),
            'length_mean': np.mean(self.episode_lengths[-window:]),
            'utilization_mean': np.mean(self.episode_utilizations[-window:]),
            'success_rate_mean': np.mean(self.episode_success_rates[-window:]),
            'window': window,
        }
        if self.a3c_total_losses:
            stats.update({
                'a3c_policy_loss_mean': float(np.mean(self.a3c_policy_losses)),
                'a3c_value_loss_mean': float(np.mean(self.a3c_value_losses)),
                'a3c_entropy_mean': float(np.mean(self.a3c_entropies)),
                'a3c_total_loss_mean': float(np.mean(self.a3c_total_losses)),
            })
        if self.high_level_policy_losses:
            stats['high_level_policy_loss_mean'] = float(np.mean(self.high_level_policy_losses))
        return stats
    
    def print_episode_summary(self, episode, reward, length, utilization, items_placed, total_items):
        """Print episode summary."""
        success_rate = (items_placed / total_items * 100) if total_items > 0 else 0.0
        
        print(f"Episode {episode:4d} | "
              f"Reward: {reward:8.4f} | "
              f"Length: {length:3d} | "
              f"Util: {utilization:6.2f}% | "
              f"Success: {success_rate:5.1f}% | "
              f"Items: {items_placed}/{total_items}", flush=True)


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
                 progress_interval=25, loop_id=None,
                 tb_log_dir='logs/tensorboard', tb_experiment=None):
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
        
        self.logger = TrainingLogger(tb_log_dir=tb_log_dir, tb_experiment=tb_experiment)
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
        self.epoch_count = 0
        self.current_state = None
        self.current_action_mask = None
        self.blf_only = bool(blf_only)
        self.deadlock_streak = 0
        self.repack_cooldown = max(1, int(repack_cooldown))
        
        # Visualization setup with correct container dimensions
        self.visualizer = ContainerVisualizer(container_dims=(env.L, env.W, env.H))
        self.vis_dir = Path(vis_dir)
        self.vis_dir.mkdir(parents=True, exist_ok=True)
        self.vis_interval = max(1, int(vis_interval))  # Save visualization every N episodes
        self.last_vis_episode = 0
        self.best_utilization = 0.0


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
                    # Urutkan berdasarkan x (panjang/depth) terlebih dahulu, baru y (lebar 0-24).
                    # a % L adalah x, a // L adalah y.
                    sorted_actions = sorted(valid_actions, key=lambda a: (a % self.env.L, a // self.env.L))
                    action = int(sorted_actions[0])
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
            return None

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

        losses = []
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
            losses.append(float(policy_loss.detach().cpu().item()))

        if not losses:
            return None
        return {'policy_loss': float(np.mean(losses))}
    
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
            
            # Save 3D visualization from multiple angles
            view_angles = [
                (25, 45),
                (35, 135),
                (20, 225),
                (15, 315),
            ]
            for elev, azim in view_angles:
                fig_3d = self.visualizer.visualize_packing_3d(
                    self.env.placed_items,
                    self.env.placed_positions,
                    title=f"{title} | elev={elev}, azim={azim}",
                    view=(elev, azim),
                )
                filename_3d = self.vis_dir / (
                    f"episode_{episode_num:04d}_3d_e{int(elev)}_a{int(azim)}{suffix}.png"
                )
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
        # Clear all caches to free memory between rollouts
        StabilityValidator.clear_cache()
        if hasattr(self.a3c, 'clear_cache'):
            self.a3c.clear_cache()
        if hasattr(self.high_level_agent, 'clear_cache'):
            self.high_level_agent.clear_cache()
        if hasattr(self.candidate_generator, 'clear_cache'):
            self.candidate_generator.clear_cache()
        
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
        
        # Continue the active episode across short rollouts. Reset only when this
        # loop has no active state yet or after an episode has finished.
        if self.current_state is None or self.current_action_mask is None:
            print(f"[Worker {self.loop_id if self.loop_id is not None else 'main'}] Resetting environment...", flush=True)
            reset_start = time.perf_counter()
            state, action_mask = self.env.reset(seed=self.seed)
            print(f"[Worker {self.loop_id if self.loop_id is not None else 'main'}] Environment reset complete in {time.perf_counter() - reset_start:.2f}s", flush=True)
            timing['episode_reset_s'] += time.perf_counter() - reset_start
            if self.seed is not None:
                self.seed += 1  # Increment seed untuk variety
        else:
            state = self.current_state
            action_mask = self.current_action_mask

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
                
                self.logger.log_episode(
                    episode_reward,
                    episode_length,
                    utilization,
                    items_placed,
                    total_items,
                    step=self.total_steps,
                    buffer_stats=self.env.get_buffer_stats(),
                )
                self.logger.print_episode_summary(self.episode_count, episode_reward, 
                                                 episode_length, utilization, 
                                                 items_placed, total_items)
                
                self.episode_count += 1
                
                # Check for new best utilization (global across all workers/envs)
                global_best_util = 0.0
                best_dir = Path('logs/training/best')
                best_json_path = best_dir / "best_placement_data.json"
                if best_json_path.exists():
                    try:
                        import json
                        with open(best_json_path, 'r') as jf:
                            data = json.load(jf)
                            global_best_util = float(data.get('utilization', 0.0))
                    except Exception:
                        pass
                
                self.best_utilization = max(self.best_utilization, global_best_util)
                
                if utilization > self.best_utilization:
                    # Remove previously saved best visualization files from this env's vis_dir
                    try:
                        for old_best_file in self.vis_dir.glob("*_best.png"):
                            try:
                                old_best_file.unlink()
                            except Exception:
                                pass
                    except Exception:
                        pass
                        
                    self.best_utilization = utilization
                    print(
                        f"\n⭐ NEW GLOBAL BEST UTILIZATION: {utilization:.2f}% (Episode {self.episode_count})! "
                        f"Saving checkpoints, raw data, and 2D/3D/cross-section visualizations...\n",
                        flush=True
                    )
                    best_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save model checkpoints
                    self.a3c.save_checkpoint(str(best_dir / "best_a3c.pt"))
                    if self.high_level_agent is not None:
                        torch.save(self.high_level_agent.state_dict(), str(best_dir / "best_high_level.pt"))
                    
                    # Generate and copy visualizations
                    vis_start = time.perf_counter()
                    self._save_visualization(self.episode_count, suffix="_best")
                    try:
                        import shutil
                        src_2d = self.vis_dir / f"episode_{self.episode_count:04d}_2d_best.png"
                        src_cross = self.vis_dir / f"episode_{self.episode_count:04d}_cross_best.png"
                        if src_2d.exists():
                            shutil.copy(str(src_2d), str(best_dir / "best_2d.png"))
                        if src_cross.exists():
                            shutil.copy(str(src_cross), str(best_dir / "best_cross.png"))
                        view_angles = [(25, 45), (35, 135), (20, 225), (15, 315)]
                        for elev, azim in view_angles:
                            src_3d = self.vis_dir / f"episode_{self.episode_count:04d}_3d_e{int(elev)}_a{int(azim)}_best.png"
                            if src_3d.exists():
                                shutil.copy(str(src_3d), str(best_dir / f"best_3d_e{int(elev)}_a{int(azim)}.png"))
                    except Exception as copy_err:
                        print(f"Error copying best visualization files: {copy_err}", flush=True)
                    
                    # Save raw placement data
                    try:
                        import json
                        best_data = {
                            'episode': self.episode_count,
                            'utilization': float(utilization),
                            'reward': float(episode_reward),
                            'length': int(episode_length),
                            'placed_items': [
                                {
                                    'l': int(item.get('l', 0)),
                                    'w': int(item.get('w', 0)),
                                    'h': int(item.get('h', 0)),
                                    'stacking': str(item.get('stacking', 'stackable')),
                                    'fragile': bool(item.get('fragile', False))
                                }
                                for item in self.env.placed_items
                            ],
                            'placed_positions': [
                                {'x': int(pos[0]), 'y': int(pos[1]), 'z': int(pos[2])}
                                for pos in self.env.placed_positions
                            ]
                        }
                        with open(best_json_path, "w") as jf:
                            json.dump(best_data, jf, indent=4)
                    except Exception as json_err:
                        print(f"Error saving best placement JSON data: {json_err}", flush=True)
                    
                    timing['visualization_s'] += time.perf_counter() - vis_start
                
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

        self.current_state = state
        self.current_action_mask = action_mask
        
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
        self.epoch_count += 1
        print(f"\n{'='*70}")
        print(f"Training Epoch {self.epoch_count}")
        print(f"{'='*70}\n")
        
        epoch_start = time.perf_counter()

        # Collect trajectories
        episode_info, next_value, strategy_buffer, timing = self.collect_steps(self.n_steps)
        
        # Update HighLevelAgent
        print(f"\nUpdating HighLevelAgent... ", end='', flush=True)
        high_level_loss = self._update_high_level_agent(strategy_buffer)
        print("Done!")
        
        # Update A3C network
        print(f"Updating A3C network... ", end='', flush=True)
        a3c_loss = self.a3c.update(next_value=next_value)
        print("Done!\n")

        self.logger.log_update(
            a3c_loss=a3c_loss,
            high_level_loss=high_level_loss,
            step=self.total_steps,
        )

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
        stats = self.logger.get_stats(last_n=100)
        stats_window = int(stats.get('window', 0))
        print(f"{'='*70}")
        print(f"Training Statistics (last {stats_window} episodes):")
        print(f"{'='*70}")
        print(f"Episode Reward:      {stats.get('reward_mean', 0):.4f} ± {stats.get('reward_std', 0):.4f}")
        print(f"Episode Length:      {stats.get('length_mean', 0):.1f} steps")
        print(f"Container Util:      {stats.get('utilization_mean', 0):.2f}%")
        print(f"Success Rate:        {stats.get('success_rate_mean', 0):.1f}%")
        if stats.get('a3c_total_loss_mean') is not None:
            print("\nA3C Loss (avg over updates):")
            print(f"  Policy Loss:       {stats.get('a3c_policy_loss_mean', 0):.6f}")
            print(f"  Value Loss:        {stats.get('a3c_value_loss_mean', 0):.6f}")
            print(f"  Entropy:           {stats.get('a3c_entropy_mean', 0):.6f}")
            print(f"  Total Loss:        {stats.get('a3c_total_loss_mean', 0):.6f}")
        if stats.get('high_level_policy_loss_mean') is not None:
            print(f"High-Level Policy Loss (avg): {stats.get('high_level_policy_loss_mean', 0):.6f}")
        
        # Also show current incomplete episode (useful for large-scale packing)
        if self.total_steps > 0:
            current_util = self.env.get_utilization() if hasattr(self.env, 'get_utilization') else 0.0
            current_items = len(self.env.placed_items) if hasattr(self.env, 'placed_items') else 0
            total_items = len(self.env.items) if hasattr(self.env, 'items') else 0
            print(f"\nCurrent Incomplete Episode:")
            print(f"  Accumulated Reward:  {self.env.episode_reward:.4f}")
            print(f"  Steps Taken:         {self.env.episode_length}")
            print(f"  Items Placed:        {current_items}/{total_items}")
            print(f"  Current Util:        {current_util:.2f}%")

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

        self.logger.log_epoch_summary(stats, self.rearrange_stats, step=self.epoch_count)


def train(num_epochs=10, n_steps=2048, seed=42, device='cpu', dataset_type='random',
          debug_mask=False, debug_actions=False, vis_interval=50, vis_dir='outputs/visualizations',
          blf_only=False, fast_stability_mask=False, repack_cooldown=1,
          layered_min_height=2, layered_max_height=6,
          pp_sigma=4, pp_size_bias=3.0, pp_mean_ratio=0.25,
          checkpoint_steps=0, candidate_top_k=128, mcts_budget=20, use_mcts_prob=0.0,
          early_stop=True, early_stop_metric='utilization_mean',
          early_stop_patience=10, early_stop_min_delta=0.1, early_stop_min_epochs=10,
          tb_log_dir='logs/tensorboard', tb_experiment='train_single',
          resume_a3c=None, resume_high_level=None,
          buffer_capacity=3, max_waiting_steps=5,
          defer_penalty=-0.02, overflow_penalty=-0.5):
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
        max_episode_length=n_steps * 2,
        buffer_capacity=buffer_capacity,
        max_waiting_steps=max_waiting_steps,
        defer_penalty=defer_penalty,
        overflow_penalty=overflow_penalty,
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
    
    if resume_a3c:
        a3c.load_checkpoint(resume_a3c)
        print(f"Resumed A3C network from {resume_a3c}")
    if resume_high_level:
        print("Warning: Single-env training mode does not use HighLevelAgent. resume_high_level ignored.")
    
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
        tb_log_dir=tb_log_dir,
        tb_experiment=tb_experiment,
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
                'a3c_policy_loss_mean': stats.get('a3c_policy_loss_mean', 0.0),
                'a3c_value_loss_mean': stats.get('a3c_value_loss_mean', 0.0),
                'a3c_entropy_mean': stats.get('a3c_entropy_mean', 0.0),
                'a3c_total_loss_mean': stats.get('a3c_total_loss_mean', 0.0),
                'high_level_policy_loss_mean': stats.get('high_level_policy_loss_mean', 0.0),
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
        print("\nTraining interrupted by user. Saving current progress...")
        interrupted_ckpt = Path('logs/training/interrupted.pt')
        a3c.save_checkpoint(str(interrupted_ckpt))
        print(f"Interrupted checkpoint saved: {interrupted_ckpt}")
    
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
        # Generate training progression plots
        generate_training_plots(csv_path)
    
    return training_loop, a3c


def _a3c_worker(worker_id, shared_model, shared_optimizer,
                shared_high_level, shared_high_optimizer,
                config, global_step, global_episode, total_steps,
                last_checkpoint_step, checkpoint_lock):
    print(f"Worker {worker_id} starting...", flush=True)
    torch.set_num_threads(1)
    worker_seed = config['seed'] + worker_id * 1000

    print(f"Worker {worker_id} initializing ContainerEnv...", flush=True)
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
        buffer_capacity=config.get('buffer_capacity', 3),
        max_waiting_steps=config.get('max_waiting_steps', 5),
        defer_penalty=config.get('defer_penalty', -0.02),
        overflow_penalty=config.get('overflow_penalty', -0.5),
    )
    env.debug_mask_stats = bool(config['debug_mask'])
    print(f"Worker {worker_id} ContainerEnv initialized.", flush=True)

    print(f"Worker {worker_id} initializing A3C agent...", flush=True)
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
        tb_log_dir=config['tb_log_dir'],
        tb_experiment=f"{config['tb_experiment']}_worker_{worker_id}",
        loop_id=worker_id,
    )

    while True:
        with global_step.get_lock():
            if global_step.value >= total_steps:
                break
        episode_info, next_value, strategy_buffer, _ = loop.collect_steps(config['rollout_steps'])

        high_level_loss = loop._update_high_level_agent(strategy_buffer)
        a3c_loss = a3c.update(next_value=next_value)
        loop.logger.log_update(a3c_loss=a3c_loss, high_level_loss=high_level_loss)

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
                early_stop_patience=10, early_stop_min_delta=0.1, early_stop_min_epochs=10,
                tb_log_dir='logs/tensorboard', tb_experiment='train_async',
                resume_a3c=None, resume_high_level=None,
                buffer_capacity=3, max_waiting_steps=5,
                defer_penalty=-0.02, overflow_penalty=-0.5):
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
        max_episode_length=rollout_steps*2,
        buffer_capacity=buffer_capacity,
        max_waiting_steps=max_waiting_steps,
        defer_penalty=defer_penalty,
        overflow_penalty=overflow_penalty,
    )
    state_size = env.state_size
    action_size = env.action_size

    shared_model = ActorCriticNetwork(L=env.L, W=env.W, action_size=action_size).to(device)
    if resume_a3c:
        shared_model.load_state_dict(torch.load(resume_a3c, map_location=device))
        print(f"Resumed A3C network from {resume_a3c}", flush=True)
    shared_model.share_memory()
    shared_optimizer = SharedAdam(shared_model.parameters(), lr=learning_rate)

    shared_high_level = HighLevelAgent(input_dim=state_size).to(device)
    if resume_high_level:
        shared_high_level.load_state_dict(torch.load(resume_high_level, map_location=device))
        print(f"Resumed high-level policy from {resume_high_level}", flush=True)
    shared_high_level.share_memory()
    shared_high_optimizer = SharedAdam(shared_high_level.parameters(), lr=1e-4)

    ctx = mp.get_context('spawn')
    total_steps = num_epochs * n_steps
    global_step = ctx.Value('i', 0)
    global_episode = ctx.Value('i', 0)

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
        'tb_log_dir': tb_log_dir,
        'tb_experiment': tb_experiment,
        'buffer_capacity': buffer_capacity,
        'max_waiting_steps': max_waiting_steps,
        'defer_penalty': defer_penalty,
        'overflow_penalty': overflow_penalty,
    }

    processes = []
    last_checkpoint_step = ctx.Value('i', 0)
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

    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Terminating workers...", flush=True)
        for p in processes:
            p.terminate()
            p.join()
        
        # Save interrupted checkpoints
        interrupted_ckpt = Path('logs/training/async_interrupted.pt')
        torch.save(shared_model.state_dict(), interrupted_ckpt)
        interrupted_hl_ckpt = Path('logs/training/async_high_level_interrupted.pt')
        torch.save(shared_high_level.state_dict(), interrupted_hl_ckpt)
        print(f"Interrupted checkpoint saved: {interrupted_ckpt}", flush=True)
        print(f"Interrupted high-level checkpoint saved: {interrupted_hl_ckpt}", flush=True)
        return shared_model

    # Save final checkpoints
    final_ckpt = Path('logs/training/async_final.pt')
    torch.save(shared_model.state_dict(), final_ckpt)
    final_hl_ckpt = Path('logs/training/async_high_level_final.pt')
    torch.save(shared_high_level.state_dict(), final_hl_ckpt)
    print(f"Final checkpoint saved: {final_ckpt}", flush=True)
    print(f"Final high-level checkpoint saved: {final_hl_ckpt}", flush=True)

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
                  early_stop_patience=10, early_stop_min_delta=0.1, early_stop_min_epochs=10,
                  tb_log_dir='logs/tensorboard', tb_experiment='train_batched',
                  resume_a3c=None, resume_high_level=None,
                  buffer_capacity=3, max_waiting_steps=5,
                  defer_penalty=-0.02, overflow_penalty=-0.5):
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
            max_episode_length=rollout_steps*2,
            buffer_capacity=buffer_capacity,
            max_waiting_steps=max_waiting_steps,
            defer_penalty=defer_penalty,
            overflow_penalty=overflow_penalty,
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

    # Load checkpoints if specified
    if resume_a3c:
        a3c.load_checkpoint(resume_a3c)
        print(f"Resumed A3C network from {resume_a3c}")
    if resume_high_level:
        shared_high_level.load_state_dict(torch.load(resume_high_level, map_location=device))
        print(f"Resumed high-level policy from {resume_high_level}")

    loops = []
    per_loop_records = {}
    epoch_records = []
    summary_logger = TrainingLogger(tb_log_dir=tb_log_dir, tb_experiment=f"{tb_experiment}_summary")
    for env_idx, env in enumerate(envs):
        loop = TrainingLoop(
            env=env,
            a3c_agent=a3c,
            n_steps=rollout_steps,
            device=device,
            seed=seed,
            debug_actions=debug_actions,
            vis_interval=vis_interval,
            vis_dir=str(Path(vis_dir) / f"env_{env_idx}"),
            blf_only=blf_only,
            repack_cooldown=repack_cooldown,
            high_level_agent=shared_high_level,
            high_level_optimizer=high_optimizer,
            candidate_top_k=candidate_top_k,
            mcts_budget=mcts_budget,
            use_mcts_prob=use_mcts_prob,
            loop_id=env_idx,
            tb_log_dir=tb_log_dir,
            tb_experiment=f"{tb_experiment}_env_{env_idx}",
        )
        loops.append(loop)
        per_loop_records[env_idx] = []

    total_steps = num_epochs * n_steps
    steps_done = 0
    next_epoch_step = n_steps
    epoch_index = 0
    best_metric = None
    no_improve_epochs = 0
    next_checkpoint = checkpoint_steps if checkpoint_steps > 0 else None
    checkpoint_dir = Path('logs/training')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    try:
        while steps_done < total_steps:
            for loop in loops:
                if steps_done >= total_steps:
                    break
                episode_info, next_value, strategy_buffer, _ = loop.collect_steps(rollout_steps)
                high_level_loss = loop._update_high_level_agent(strategy_buffer)
                a3c_loss = a3c.update(next_value=next_value)
                loop.logger.log_update(
                    a3c_loss=a3c_loss,
                    high_level_loss=high_level_loss,
                    step=loop.total_steps,
                )
                steps_done += rollout_steps
                print(
                    f"Global progress | steps_done={steps_done}/{total_steps} | "
                    f"last_loop={loop.loop_id} | episodes={[l.episode_count for l in loops]}",
                    flush=True,
                )

                if steps_done >= next_epoch_step:
                    epoch_index += 1
                    print(
                        f"\nBatched epoch {epoch_index}/{num_epochs} complete | "
                        f"steps_done={steps_done}/{total_steps} | "
                        f"episodes={[l.episode_count for l in loops]}\n",
                        flush=True,
                    )
                    next_epoch_step += n_steps

                    stats_list = [l.logger.get_stats() for l in loops if l.logger.get_stats()]
                    if stats_list:
                        avg_metric = float(
                            np.mean([s.get(early_stop_metric, 0.0) for s in stats_list])
                        )
                        avg_a3c_policy = float(
                            np.mean([s.get('a3c_policy_loss_mean', 0.0) for s in stats_list])
                        )
                        avg_a3c_value = float(
                            np.mean([s.get('a3c_value_loss_mean', 0.0) for s in stats_list])
                        )
                        avg_a3c_entropy = float(
                            np.mean([s.get('a3c_entropy_mean', 0.0) for s in stats_list])
                        )
                        avg_a3c_total = float(
                            np.mean([s.get('a3c_total_loss_mean', 0.0) for s in stats_list])
                        )
                        avg_high_level = float(
                            np.mean([s.get('high_level_policy_loss_mean', 0.0) for s in stats_list])
                        )
                        print(
                            "Batched loss avg | "
                            f"a3c_policy={avg_a3c_policy:.6f} | "
                            f"a3c_value={avg_a3c_value:.6f} | "
                            f"a3c_entropy={avg_a3c_entropy:.6f} | "
                            f"a3c_total={avg_a3c_total:.6f} | "
                            f"high_level_policy={avg_high_level:.6f}",
                            flush=True,
                        )
                        total_deadlocks = 0
                        total_rearrange_attempts = 0
                        total_rearrange_success = 0
                        total_rearrange_applied = 0
                        total_rearrange_value = 0.0
                        total_unpack_depth = 0.0
                        total_mcts_active = 0
                        total_mcts_fallback = 0

                        for loop in loops:
                            stats = loop.logger.get_stats()
                            if not stats:
                                continue
                            loop.logger.log_epoch_summary(stats, loop.rearrange_stats, step=epoch_index)
                            rearr_attempts = max(loop.rearrange_stats['rearrange_attempts'], 1)
                            per_loop_records[loop.loop_id].append({
                                'epoch': epoch_index,
                                'reward_mean_last100': stats.get('reward_mean', 0.0),
                                'reward_std_last100': stats.get('reward_std', 0.0),
                                'length_mean_last100': stats.get('length_mean', 0.0),
                                'utilization_mean_last100': stats.get('utilization_mean', 0.0),
                                'success_rate_mean_last100': stats.get('success_rate_mean', 0.0),
                                'a3c_policy_loss_mean': stats.get('a3c_policy_loss_mean', 0.0),
                                'a3c_value_loss_mean': stats.get('a3c_value_loss_mean', 0.0),
                                'a3c_entropy_mean': stats.get('a3c_entropy_mean', 0.0),
                                'a3c_total_loss_mean': stats.get('a3c_total_loss_mean', 0.0),
                                'high_level_policy_loss_mean': stats.get('high_level_policy_loss_mean', 0.0),
                                'deadlocks': loop.rearrange_stats['deadlocks'],
                                'rearrange_attempts': loop.rearrange_stats['rearrange_attempts'],
                                'rearrange_success_rate': loop.rearrange_stats['rearrange_success'] / rearr_attempts,
                                'rearrange_apply_rate': loop.rearrange_stats['rearrange_applied'] / rearr_attempts,
                                'avg_rearrange_value': loop.rearrange_stats['rearrange_best_value_sum'] / rearr_attempts,
                                'avg_unpack_depth': loop.rearrange_stats['rearrange_unpack_depth_sum'] / rearr_attempts,
                                'mcts_fallback_used': loop.rearrange_stats['mcts_fallback_used'],
                                'total_steps': loop.total_steps,
                                'total_episodes': loop.episode_count,
                            })

                            total_deadlocks += int(loop.rearrange_stats['deadlocks'])
                            total_rearrange_attempts += int(loop.rearrange_stats['rearrange_attempts'])
                            total_rearrange_success += int(loop.rearrange_stats['rearrange_success'])
                            total_rearrange_applied += int(loop.rearrange_stats['rearrange_applied'])
                            total_rearrange_value += float(loop.rearrange_stats['rearrange_best_value_sum'])
                            total_unpack_depth += float(loop.rearrange_stats['rearrange_unpack_depth_sum'])
                            total_mcts_active += int(loop.rearrange_stats['mcts_active_used'])
                            total_mcts_fallback += int(loop.rearrange_stats['mcts_fallback_used'])

                        total_rearr_attempts = max(total_rearrange_attempts, 1)
                        epoch_records.append({
                            'epoch': epoch_index,
                            'reward_mean_last100': float(np.mean([s.get('reward_mean', 0.0) for s in stats_list])),
                            'reward_std_last100': float(np.mean([s.get('reward_std', 0.0) for s in stats_list])),
                            'length_mean_last100': float(np.mean([s.get('length_mean', 0.0) for s in stats_list])),
                            'utilization_mean_last100': float(np.mean([s.get('utilization_mean', 0.0) for s in stats_list])),
                            'success_rate_mean_last100': float(np.mean([s.get('success_rate_mean', 0.0) for s in stats_list])),
                            'a3c_policy_loss_mean': float(np.mean([s.get('a3c_policy_loss_mean', 0.0) for s in stats_list])),
                            'a3c_value_loss_mean': float(np.mean([s.get('a3c_value_loss_mean', 0.0) for s in stats_list])),
                            'a3c_entropy_mean': float(np.mean([s.get('a3c_entropy_mean', 0.0) for s in stats_list])),
                            'a3c_total_loss_mean': float(np.mean([s.get('a3c_total_loss_mean', 0.0) for s in stats_list])),
                            'high_level_policy_loss_mean': float(
                                np.mean([s.get('high_level_policy_loss_mean', 0.0) for s in stats_list])
                            ),
                            'deadlocks': int(total_deadlocks),
                            'rearrange_attempts': int(total_rearrange_attempts),
                            'rearrange_success_rate': float(
                                total_rearrange_success / total_rearr_attempts
                            ),
                            'rearrange_apply_rate': float(
                                total_rearrange_applied / total_rearr_attempts
                            ),
                            'avg_rearrange_value': float(
                                total_rearrange_value / total_rearr_attempts
                            ),
                            'avg_unpack_depth': float(
                                total_unpack_depth / total_rearr_attempts
                            ),
                            'mcts_fallback_used': int(total_mcts_fallback),
                            'total_steps': int(steps_done),
                            'total_episodes': int(sum(l.episode_count for l in loops)),
                        })
                        summary_stats = {
                            'reward_mean': float(np.mean([s.get('reward_mean', 0.0) for s in stats_list])),
                            'reward_std': float(np.mean([s.get('reward_std', 0.0) for s in stats_list])),
                            'length_mean': float(np.mean([s.get('length_mean', 0.0) for s in stats_list])),
                            'utilization_mean': float(np.mean([s.get('utilization_mean', 0.0) for s in stats_list])),
                            'success_rate_mean': float(np.mean([s.get('success_rate_mean', 0.0) for s in stats_list])),
                            'a3c_policy_loss_mean': float(np.mean([s.get('a3c_policy_loss_mean', 0.0) for s in stats_list])),
                            'a3c_value_loss_mean': float(np.mean([s.get('a3c_value_loss_mean', 0.0) for s in stats_list])),
                            'a3c_entropy_mean': float(np.mean([s.get('a3c_entropy_mean', 0.0) for s in stats_list])),
                            'a3c_total_loss_mean': float(np.mean([s.get('a3c_total_loss_mean', 0.0) for s in stats_list])),
                            'high_level_policy_loss_mean': float(
                                np.mean([s.get('high_level_policy_loss_mean', 0.0) for s in stats_list])
                            ),
                        }
                        summary_rearrange = {
                            'deadlocks': total_deadlocks,
                            'rearrange_attempts': total_rearrange_attempts,
                            'rearrange_success': total_rearrange_success,
                            'rearrange_applied': total_rearrange_applied,
                            'rearrange_best_value_sum': total_rearrange_value,
                            'rearrange_unpack_depth_sum': total_unpack_depth,
                            'mcts_active_used': total_mcts_active,
                            'mcts_fallback_used': total_mcts_fallback,
                        }
                        summary_logger.log_epoch_summary(summary_stats, summary_rearrange, step=epoch_index)

                        print(
                            f"Batched stats | deadlocks={total_deadlocks} | "
                            f"rearrange_attempts={total_rearrange_attempts} | "
                            f"rearrange_success_rate={total_rearrange_success / total_rearr_attempts:.3f} | "
                            f"rearrange_apply_rate={total_rearrange_applied / total_rearr_attempts:.3f} | "
                            f"mcts_active={total_mcts_active} | mcts_fallback={total_mcts_fallback}",
                            flush=True,
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
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current progress...")
        interrupted_ckpt = Path('logs/training/batched_interrupted.pt')
        a3c.save_checkpoint(str(interrupted_ckpt))
        interrupted_hl_ckpt = Path('logs/training/batched_high_level_interrupted.pt')
        torch.save(shared_high_level.state_dict(), interrupted_hl_ckpt)
        print(f"Interrupted checkpoint saved: {interrupted_ckpt}")
        print(f"Interrupted high-level checkpoint saved: {interrupted_hl_ckpt}")

    logs_root = Path('logs/training')
    logs_root.mkdir(parents=True, exist_ok=True)
    if epoch_records:
        summary_path = logs_root / 'batched_training_epoch_metrics.csv'
        with summary_path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(epoch_records[0].keys()))
            writer.writeheader()
            writer.writerows(epoch_records)
        print(f"Batched training metrics CSV saved: {summary_path}")
        # Generate summary plots
        generate_training_plots(summary_path)

    for loop_id, records in per_loop_records.items():
        if not records:
            continue
        loop_dir = logs_root / f"batched_env_{loop_id}"
        loop_dir.mkdir(parents=True, exist_ok=True)
        loop_csv = loop_dir / 'training_epoch_metrics.csv'
        with loop_csv.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
        print(f"Env {loop_id} metrics CSV saved: {loop_csv}")
        # Generate env-specific plots
        generate_training_plots(loop_csv)

    final_ckpt = logs_root / 'batched_final.pt'
    a3c.save_checkpoint(str(final_ckpt))
    final_hl_ckpt = logs_root / 'batched_high_level_final.pt'
    torch.save(shared_high_level.state_dict(), final_hl_ckpt)
    print(f"Batched final checkpoint saved: {final_ckpt}")
    print(f"Batched final high-level checkpoint saved: {final_hl_ckpt}")

    return a3c


def generate_training_plots(csv_path, output_dir=None):
    """
    Generate beautiful training plots from a metrics CSV file.
    """
    import csv
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"⚠️ CSV file not found: {csv_path}. Skipping plot generation.")
        return
        
    if output_dir is None:
        output_dir = csv_path.parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    epochs = []
    rewards = []
    utilizations = []
    success_rates = []
    policy_losses = []
    value_losses = []
    entropies = []
    total_losses = []
    deadlocks = []
    rearrange_success_rates = []
    
    try:
        with csv_path.open('r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs.append(int(row.get('epoch', len(epochs))))
                rewards.append(float(row.get('reward_mean_last100', row.get('reward_mean', 0.0))))
                utilizations.append(float(row.get('utilization_mean_last100', row.get('utilization_mean', 0.0))))
                success_rates.append(float(row.get('success_rate_mean_last100', row.get('success_rate_mean', 0.0))))
                policy_losses.append(float(row.get('a3c_policy_loss_mean', 0.0)))
                value_losses.append(float(row.get('a3c_value_loss_mean', 0.0)))
                entropies.append(float(row.get('a3c_entropy_mean', 0.0)))
                total_losses.append(float(row.get('a3c_total_loss_mean', 0.0)))
                deadlocks.append(int(row.get('deadlocks', 0)))
                rearrange_success_rates.append(float(row.get('rearrange_success_rate', 0.0)) * 100)
    except Exception as e:
        print(f"⚠️ Error reading CSV for plotting: {e}")
        return
        
    if not epochs:
        print("⚠️ No epoch records found in CSV. Skipping plot generation.")
        return

    plt.style.use('default')
    
    # Plot 1: Utilization & Success Rate
    fig, ax1 = plt.subplots(figsize=(10, 6))
    color = '#1f77b4'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Utilization (%)', color=color)
    line1 = ax1.plot(epochs, utilizations, color=color, linewidth=2, marker='o', label='Utilization')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 100)
    
    ax2 = ax1.twinx()
    color = '#2ca02c'
    ax2.set_ylabel('Success Rate (%)', color=color)
    line2 = ax2.plot(epochs, success_rates, color=color, linewidth=2, linestyle='--', marker='s', label='Success Rate')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 100)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    plt.title('Packing Performance Progression (Utilization vs Success Rate)', fontsize=14, fontweight='bold', pad=15)
    fig.tight_layout()
    plt.savefig(output_dir / 'training_utilization_success.png', dpi=150)
    plt.close()
    
    # Plot 2: Rewards
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, rewards, color='#d62728', linewidth=2, marker='o', label='Mean Reward')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Reward')
    plt.title('Training Reward Progression', fontsize=14, fontweight='bold', pad=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'training_rewards.png', dpi=150)
    plt.close()
    
    # Plot 3: A3C Losses & Entropy
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs[0, 0].plot(epochs, policy_losses, color='#9467bd', linewidth=1.5, marker='o')
    axs[0, 0].set_title('A3C Policy Loss')
    axs[0, 0].set_xlabel('Epoch')
    
    axs[0, 1].plot(epochs, value_losses, color='#8c564b', linewidth=1.5, marker='o')
    axs[0, 1].set_title('A3C Value Loss')
    axs[0, 1].set_xlabel('Epoch')
    
    axs[1, 0].plot(epochs, entropies, color='#e377c2', linewidth=1.5, marker='o')
    axs[1, 0].set_title('Policy Entropy')
    axs[1, 0].set_xlabel('Epoch')
    
    axs[1, 1].plot(epochs, total_losses, color='#7f7f7f', linewidth=1.5, marker='o')
    axs[1, 1].set_title('A3C Total Loss')
    axs[1, 1].set_xlabel('Epoch')
    
    plt.suptitle('A3C Network Convergence Metrics', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir / 'training_losses.png', dpi=150)
    plt.close()
    
    # Plot 4: MCTS Rearrangement Metrics
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].plot(epochs, deadlocks, color='#ff7f0e', linewidth=2, marker='o', label='Deadlocks')
    axs[0].set_title('Total Deadlocks per Epoch')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Count')
    
    axs[1].plot(epochs, rearrange_success_rates, color='#17becf', linewidth=2, marker='o', label='Success %')
    axs[1].set_title('MCTS Rearrangement Success Rate')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Success Rate (%)')
    axs[1].set_ylim(0, 100)
    
    plt.suptitle('MCTS Planning & Rearrangement Performance', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir / 'training_mcts_performance.png', dpi=150)
    plt.close()
    
    print(f"📊 Training plots successfully generated and saved to: {output_dir}")


if __name__ == "__main__":
    """Train A3C agent for 3D bin packing"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Train A3C agent for 3D bin packing')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--n_steps', type=int, default=2048, help='Steps per epoch')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cpu', help='Device: cpu or cuda')
    parser.add_argument('--dataset', type=str, default='perfect_pack',
                        help='Dataset: perfect_pack, perfect_pack_layered, perfect_pack_pt, or rs')
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
    parser.add_argument('--resume-a3c', type=str, default=None,
                        help='Path to A3C checkpoint to resume training')
    parser.add_argument('--resume-high-level', type=str, default=None,
                        help='Path to HighLevelAgent checkpoint to resume training')
    parser.add_argument('--buffer-capacity', type=int, default=3, help='Capacity of holding buffer (0 to disable)')
    parser.add_argument('--max-waiting-steps', type=int, default=5, help='Max waiting steps in buffer before eviction')
    parser.add_argument('--defer-penalty', type=float, default=-0.02, help='Penalty for deferring an item')
    parser.add_argument('--overflow-penalty', type=float, default=-0.5, help='Penalty for buffer overflow/rejection')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("3D BIN PACKING WITH A3C TRAINING")
    print("="*70)
    if args.dataset not in {'perfect_pack', 'perfect_pack_layered', 'perfect_pack_pt', 'rs'}:
        raise ValueError("Training hanya mendukung dataset 'perfect_pack', 'perfect_pack_layered', 'perfect_pack_pt', atau 'rs'.")

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
        f"early_stop_patience={args.early_stop_patience}, buffer_capacity={args.buffer_capacity}, "
        f"max_waiting_steps={args.max_waiting_steps}, defer_penalty={args.defer_penalty}, "
        f"overflow_penalty={args.overflow_penalty}\n"
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
            resume_a3c=args.resume_a3c,
            resume_high_level=args.resume_high_level,
            buffer_capacity=args.buffer_capacity,
            max_waiting_steps=args.max_waiting_steps,
            defer_penalty=args.defer_penalty,
            overflow_penalty=args.overflow_penalty,
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
            resume_a3c=args.resume_a3c,
            resume_high_level=args.resume_high_level,
            buffer_capacity=args.buffer_capacity,
            max_waiting_steps=args.max_waiting_steps,
            defer_penalty=args.defer_penalty,
            overflow_penalty=args.overflow_penalty,
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
            resume_a3c=args.resume_a3c,
            resume_high_level=args.resume_high_level,
            buffer_capacity=args.buffer_capacity,
            max_waiting_steps=args.max_waiting_steps,
            defer_penalty=args.defer_penalty,
            overflow_penalty=args.overflow_penalty,
        )
    
    print("\nTraining completed successfully!")
