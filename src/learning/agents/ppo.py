"""Proximal Policy Optimization (PPO) Agent (MODERN LOW-LEVEL ALGORITHM).

PART OF HIERARCHICAL RL:
  - HIGH-LEVEL: HighLevelAgent (models/high_level_agent.py) -> What strategy?
  - LOW-LEVEL: PPO + ActorCriticNetwork (this file + models/actor_critic.py) -> Where to place?

RELATIONSHIP:
  PPO algorithm trains ActorCriticNetwork to:
    1. Select placement position given strategy from HighLevelAgent
    2. Estimate value of current container state
    3. Update via policy gradient (PPO-Clip) + value loss

TRAINING FLOW:
  collect_steps() -> PPO.select_action() -> ActorCriticNetwork.forward()
                     |
                     v
                  Collect trajectories
                     |
                     v
                PPO.update() -> Retrain ActorCriticNetwork weights
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from src.learning.models.actor_critic import ActorCriticNetwork
from src.common.memory import Memory


class PPO:
    """Proximal Policy Optimization (PPO) - Modern Low-Level Training Algorithm.
    
    HIERARCHICAL RL ROLE (Low-Level):
      Given a strategy from HighLevelAgent, PPO learns WHERE to place items
      by training ActorCriticNetwork through policy gradient optimization.
    
    HOW IT WORKS:
      1. select_action() -> ActorCriticNetwork outputs placement position
      2. Collect step trajectories (state, action, reward, value, log_prob)
      3. update() -> Retrain network weights using PPO-Clip objective
    
    KEY FEATURES:
      - CNN-based Actor network (ActorCriticNetwork)
      - Masked logits (only valid positions allowed)
      - Soft-max sampling for exploration
      - GAE (Generalized Advantage Estimation) for better returns
      - PPO-Clip objective (prevents large policy updates)
      - Entropy bonus (encourage exploration)
      - Value loss (predict state values)
      - Mini-batch training with multiple epochs
    
    TRAINING LOOP (from train.py):
      for episode in range(num_episodes):
          for step in range(steps_per_episode):
              action, log_prob, value = ppo.select_action(state)  # Network forward
              next_state, reward, done = env.step(action)
              ppo.store(state, action, reward, value, log_prob)
          
          # Retrain network on collected trajectories
          ppo.update()  # PPO-Clip + value loss on mini-batches
    
    REFERENCE TO DEPRECATED VERSION:
      Old LowLevelAgent class is deprecated (models/low_level_agent.py)
      Use PPO + ActorCriticNetwork instead for modern implementations.
    """
    
    def __init__(self, state_size, action_size, L=None, W=None,
                 learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_ratio=0.2, entropy_coef=0.01, value_coef=0.5,
                 device='cpu'):
        """
        Initialize PPO agent.
        
        Args:
            state_size (int): State size (L*W + 3 for flattened height_map + item dims)
            action_size (int): Action space size (L*W + 1)
            L (int, optional): Container length. If None, inferred from state_size
            W (int, optional): Container width. If None, inferred from state_size
            learning_rate (float): Learning rate
            gamma (float): Discount factor
            gae_lambda (float): GAE lambda parameter
            clip_ratio (float): PPO clipping ratio (epsilon)
            entropy_coef (float): Entropy bonus coefficient
            value_coef (float): Value loss coefficient
            device (str): 'cpu' atau 'cuda'
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.device = device
        
        # Infer L, W from state_size if not provided
        # state_size = L*W + 3, action_size = L*W + 1
        # So L*W = action_size - 1 = state_size - 4
        if L is None or W is None:
            height_map_size = state_size - 3  # Remove item dims
            # Assume default container is 59x23 if inference fails
            if L is None:
                L = 59
            if W is None:
                W = 23
            # Verify consistency
            if L * W != height_map_size:
                print(f"Warning: Inferred L={L}, W={W} don't match state_size={state_size}. "
                      f"Expected L*W={height_map_size}, got {L*W}. Using provided values anyway.")
        
        self.L = L
        self.W = W
        
        # Initialize networks with CNN-based architecture
        self.network = ActorCriticNetwork(L=L, W=W, action_size=action_size).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Memory untuk trajectories
        self.memory = Memory()
    
    def mask_logits(self, logits, action_mask):
        """
        Mask logits berdasarkan action mask.
        
        Logits yang tidak valid di-set ke -inf agar softmax = 0.
        
        Args:
            logits (torch.Tensor): Actor network output (batch_size, action_size)
            action_mask (torch.Tensor): Action mask (batch_size, action_size)
            
        Returns:
            torch.Tensor: Masked logits
        """
        # Convert mask ke bool tensor jika perlu
        if action_mask.dtype != torch.bool:
            action_mask = action_mask.bool()
        
        # Mask invalid actions dengan -inf
        masked_logits = logits.clone()
        masked_logits[~action_mask] = float('-inf')
        
        return masked_logits
    
    def select_action(self, state, action_mask):
        """
        Select action menggunakan softmax sampling dari masked logits.
        
        Args:
            state (np.ndarray): Current state
            action_mask (np.ndarray): Action mask
            
        Returns:
            tuple: (action, log_prob, value)
        """
        # Convert ke tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_mask_tensor = torch.FloatTensor(action_mask).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits, value = self.network(state_tensor)
        
        # Mask logits
        masked_logits = self.mask_logits(logits, action_mask_tensor)
        
        # Softmax dengan masked logits
        probs = F.softmax(masked_logits, dim=-1)
        
        # Handle NaN dari -inf softmax without in-place operation
        probs = torch.where(torch.isnan(probs), torch.zeros_like(probs), probs)
        
        # Categorical sampling
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), value.item()
    
    def store_transition(self, state, action, reward, value, log_prob, 
                        action_mask, done):
        """
        Store transition dalam memory.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            value: Value estimation dari critic
            log_prob: Log probability dari action
            action_mask: Action mask
            done: Episode done flag
        """
        self.memory.add(state, action, reward, value, log_prob, action_mask, done)
    
    def compute_gae_advantages(self, next_value):
        """
        Compute Generalized Advantage Estimation (GAE).
        
        GAE = sum(lambda^t * delta_t) dimana delta_t = r_t + gamma*V(s_{t+1}) - V(s_t)
        
        Args:
            next_value (float): Value estimation dari next state (for bootstrap)
            
        Returns:
            tuple: (advantages, returns)
        """
        states, actions, rewards, values, log_probs, _, dones = self.memory.get_batch()
        
        advantages = []
        gae = 0.0
        
        # Compute GAE backwards
        values_with_next = list(values.squeeze().numpy()) + [next_value]
        rewards_np = rewards.numpy()
        dones_np = dones.numpy()
        
        for t in reversed(range(len(rewards_np))):
            next_value_t = values_with_next[t + 1]
            
            # TD error: delta_t = r_t + gamma*V(s_{t+1}) - V(s_t)
            delta = rewards_np[t] + self.gamma * next_value_t * (1 - dones_np[t]) - values_with_next[t]
            
            # GAE: A_t = delta_t + (lambda*gamma)^1 * delta_{t+1} + ...
            gae = delta + self.gamma * self.gae_lambda * (1 - dones_np[t]) * gae
            advantages.insert(0, gae)
        
        advantages = torch.FloatTensor(advantages)
        returns = advantages + values.squeeze()
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def update(self, next_value, num_epochs=3, batch_size=64):
        """
        Update policy dan value function menggunakan PPO loss.
        
        Loss = Policy Loss + Value Loss - Entropy Bonus
        
        Policy Loss = -min(ratio * advantage, clip(ratio, 1-eps, 1+eps) * advantage)
        Value Loss = MSE(V(s), returns)
        Entropy Bonus = -H(policy)
        
        Args:
            next_value (float): Value estimation dari next state
            num_epochs (int): Jumlah epoch untuk mini-batch update
            batch_size (int): Mini-batch size
        """
        advantages, returns = self.compute_gae_advantages(next_value)
        
        states, actions, rewards, values, old_log_probs, action_masks, dones = self.memory.get_batch()
        
        # Mini-batch training
        num_samples = len(states)
        indices = np.arange(num_samples)
        
        for epoch in range(num_epochs):
            np.random.shuffle(indices)
            
            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                batch_indices = indices[start_idx:end_idx]
                
                batch_states = states[batch_indices].to(self.device)
                batch_actions = actions[batch_indices].to(self.device)
                batch_advantages = advantages[batch_indices].to(self.device)
                batch_returns = returns[batch_indices].to(self.device)
                batch_old_log_probs = old_log_probs[batch_indices].to(self.device)
                batch_action_masks = action_masks[batch_indices].to(self.device)
                
                # Forward pass
                logits, values = self.network(batch_states)
                
                # Mask logits
                masked_logits = self.mask_logits(logits, batch_action_masks)
                
                # New probabilities dan log_probs
                probs = F.softmax(masked_logits, dim=-1)
                # Handle NaN without in-place operation
                probs = torch.where(torch.isnan(probs), torch.zeros_like(probs), probs)
                
                dist = Categorical(probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # Clipped objective (PPO)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 
                                   1 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values.squeeze(), batch_returns)
                
                # Total loss
                total_loss = (policy_loss + 
                             self.value_coef * value_loss - 
                             self.entropy_coef * entropy)
                
                # Backward pass
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
