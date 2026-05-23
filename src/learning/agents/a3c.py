"""Asynchronous Advantage Actor-Critic (A3C) Low-Level Agent.

This is a synchronous single-process variant that follows the A3C loss:
policy loss + value loss - entropy bonus.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import OrderedDict

from src.learning.models.actor_critic import ActorCriticNetwork
from src.common.memory import Memory


class A3C:
    """Advantage Actor-Critic (A3C) agent for low-level placement decisions."""

    def __init__(self, state_size, action_size, L=None, W=None,
                 learning_rate=3e-4, gamma=0.99,
                 entropy_coef=0.01, value_coef=0.5,
                 device='cpu', network=None, optimizer=None):
        """
        Initialize A3C agent.

        Args:
            state_size (int): State size (L*W + 4 for height_map + item dims + min_height)
            action_size (int): Action space size (L*W + 1)
            L (int, optional): Container length
            W (int, optional): Container width
            learning_rate (float): Learning rate
            gamma (float): Discount factor
            entropy_coef (float): Entropy bonus coefficient
            value_coef (float): Value loss coefficient
            device (str): 'cpu' atau 'cuda'
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.device = device

        if L is None or W is None:
            height_map_size = state_size - 4
            if L is None:
                L = 59
            if W is None:
                W = 23
            if L * W != height_map_size:
                print(
                    f"Warning: Inferred L={L}, W={W} don't match state_size={state_size}. "
                    f"Expected L*W={height_map_size}, got {L*W}. Using provided values anyway."
                )

        self.L = L
        self.W = W

        if network is None:
            self.network = ActorCriticNetwork(L=L, W=W, action_size=action_size).to(device)
        else:
            self.network = network.to(device)

        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        else:
            self.optimizer = optimizer

        self.memory = Memory()

    @staticmethod
    def _state_fingerprint(state):
        """Compute fast fingerprint of state for caching (using hash)."""
        try:
            if isinstance(state, np.ndarray):
                return hash(state.tobytes())
            else:
                return hash(tuple(state))
        except Exception:
            return None

    def clear_cache(self):
        """Clear forward pass cache to free memory between episodes."""
        if hasattr(self, '_forward_cache'):
            self._forward_cache.clear()
    
    @property
    def _forward_cache(self):
        """Lazy-initialized LRU cache for forward pass results."""
        if not hasattr(self, '_fc'):
            self._fc = OrderedDict()
            self._fc_max_size = 512
        return self._fc

    def mask_logits(self, logits, action_mask):
        """Mask logits berdasarkan action mask."""
        if action_mask.dtype != torch.bool:
            action_mask = action_mask.bool()

        masked_logits = logits.clone()
        masked_logits[~action_mask] = float('-inf')
        return masked_logits

    def select_action(self, state, action_mask):
        """Select action menggunakan softmax sampling dari masked logits."""
        # Check cache first
        cache_key = self._state_fingerprint(state)
        if cache_key is not None and cache_key in self._forward_cache:
            cached_result = self._forward_cache[cache_key]
            self._forward_cache.move_to_end(cache_key)
            logits, value = cached_result
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits, value = self.network(state_tensor)
            
            # Cache result (store on CPU to save GPU memory)
            if cache_key is not None:
                self._forward_cache[cache_key] = (logits.cpu(), value.cpu())
                if len(self._forward_cache) > self._forward_cache_max_size:
                    self._forward_cache.popitem(last=False)

        action_mask_tensor = torch.FloatTensor(action_mask).unsqueeze(0).to(self.device)

        masked_logits = self.mask_logits(logits, action_mask_tensor)
        probs = F.softmax(masked_logits, dim=-1)
        probs = torch.where(torch.isnan(probs), torch.zeros_like(probs), probs)

        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), value.item()

    def select_actions_batch(self, states, action_masks):
        """
        Batch action selection for multiple states (faster than sequential calls).
        
        Args:
            states: List of states or numpy array (N, state_size)
            action_masks: List of action masks or numpy array (N, action_size)
            
        Returns:
            list: [(action, log_prob, value), ...] for each state
        """
        if isinstance(states, list):
            states = np.array(states)
        if isinstance(action_masks, list):
            action_masks = np.array(action_masks)
        
        batch_size = len(states)
        state_tensor = torch.FloatTensor(states).to(self.device)
        action_mask_tensor = torch.FloatTensor(action_masks).to(self.device)
        
        with torch.no_grad():
            logits, values = self.network(state_tensor)
        
        results = []
        for i in range(batch_size):
            mask_i = action_mask_tensor[i:i+1]
            logit_i = logits[i:i+1]
            value_i = values[i:i+1]
            
            masked_logits = self.mask_logits(logit_i, mask_i)
            probs = F.softmax(masked_logits, dim=-1)
            probs = torch.where(torch.isnan(probs), torch.zeros_like(probs), probs)
            
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            results.append((action.item(), log_prob.item(), value_i.item()))
        
        return results

    def store_transition(self, state, action, reward, value, log_prob,
                        action_mask, done):
        """Store transition dalam memory."""
        self.memory.add(state, action, reward, value, log_prob, action_mask, done)

    def _compute_returns(self, next_value):
        """Compute discounted returns with bootstrap."""
        _, _, rewards, _, _, _, dones = self.memory.get_batch()

        returns = []
        R = float(next_value)
        rewards_np = rewards.numpy()
        dones_np = dones.numpy()

        for t in reversed(range(len(rewards_np))):
            R = rewards_np[t] + self.gamma * R * (1 - dones_np[t])
            returns.insert(0, R)

        return torch.FloatTensor(returns)

    def update(self, next_value):
        """Update policy and value function using A3C loss."""
        states, actions, rewards, values, _, action_masks, dones = self.memory.get_batch()
        returns = self._compute_returns(next_value)
        advantages = returns - values.squeeze()

        batch_states = states.to(self.device)
        batch_actions = actions.to(self.device)
        batch_action_masks = action_masks.to(self.device)
        batch_returns = returns.to(self.device)
        batch_advantages = advantages.to(self.device)

        logits, values_pred = self.network(batch_states)
        masked_logits = self.mask_logits(logits, batch_action_masks)
        probs = F.softmax(masked_logits, dim=-1)
        probs = torch.where(torch.isnan(probs), torch.zeros_like(probs), probs)

        dist = Categorical(probs)
        log_probs = dist.log_prob(batch_actions)
        entropy = dist.entropy().mean()

        policy_loss = -(log_probs * batch_advantages.detach()).mean()
        value_loss = F.mse_loss(values_pred.squeeze(), batch_returns)

        total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.optimizer.step()

        self.memory.clear()

    def save_checkpoint(self, filepath):
        """Save model checkpoint."""
        torch.save(self.network.state_dict(), filepath)

    def load_checkpoint(self, filepath):
        """Load model checkpoint."""
        self.network.load_state_dict(torch.load(filepath, map_location=self.device))
