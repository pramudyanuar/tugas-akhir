import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic network untuk PPO.
    
    - Actor: output logits untuk action selection
    - Critic: output value estimation
    """
    
    def __init__(self, state_size, action_size, hidden_size=256):
        """
        Initialize Actor-Critic network.
        
        Args:
            state_size (int): Ukuran state (height_map flattened + item_dims)
            action_size (int): Ukuran action space (L*W + 1 untuk skip)
            hidden_size (int): Hidden layer size
        """
        super(ActorCriticNetwork, self).__init__()
        
        # Shared layers
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        # Actor head: output logits
        self.actor = nn.Linear(hidden_size, action_size)
        
        # Critic head: output value
        self.critic = nn.Linear(hidden_size, 1)
        
    def forward(self, state):
        """
        Forward pass untuk actor dan critic.
        
        Args:
            state (torch.Tensor): State tensor (batch_size, state_size)
            
        Returns:
            tuple: (logits, value)
                - logits: Actor output logits (batch_size, action_size)
                - value: Critic value estimation (batch_size, 1)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        logits = self.actor(x)
        value = self.critic(x)
        
        return logits, value


class Memory:
    """
    Storage untuk trajectories dalam episode.
    
    Store: states, actions, rewards, values, log_probs, masks, dones
    """
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.action_masks = []
        self.dones = []
        
    def add(self, state, action, reward, value, log_prob, action_mask, done):
        """Tambah transition ke memory."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.action_masks.append(action_mask)
        self.dones.append(done)
    
    def clear(self):
        """Clear memory setelah update."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.action_masks.clear()
        self.dones.clear()
    
    def get_batch(self):
        """Get batch data untuk training."""
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.LongTensor(np.array(self.actions))
        rewards = torch.FloatTensor(np.array(self.rewards))
        values = torch.FloatTensor(np.array(self.values))
        log_probs = torch.FloatTensor(np.array(self.log_probs))
        action_masks = torch.FloatTensor(np.array(self.action_masks))
        dones = torch.FloatTensor(np.array(self.dones))
        
        return states, actions, rewards, values, log_probs, action_masks, dones


class PPO:
    """
    Proximal Policy Optimization (PPO) implementation.
    
    Features:
    - Actor network output logits
    - Mask logits dengan action mask
    - Softmax sampling untuk action selection
    - Store trajectories dalam memory
    - GAE (Generalized Advantage Estimation)
    - Clipped objective (PPO-Clip)
    - Entropy bonus
    - Value loss
    - Mini-batch update
    """
    
    def __init__(self, state_size, action_size, 
                 learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_ratio=0.2, entropy_coef=0.01, value_coef=0.5,
                 device='cpu'):
        """
        Initialize PPO agent.
        
        Args:
            state_size (int): State size
            action_size (int): Action space size
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
        
        # Initialize networks
        self.network = ActorCriticNetwork(state_size, action_size).to(device)
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


if __name__ == "__main__":
    """Test cases untuk PPO"""
    
    print("=" * 70)
    print("Test Case 1: Actor-Critic Network")
    print("=" * 70)
    
    state_size = 59 * 23 + 3  # height_map + item_dims
    action_size = 59 * 23 + 1  # positions + skip
    
    network = ActorCriticNetwork(state_size, action_size)
    state = torch.randn(4, state_size)  # Batch of 4
    
    logits, values = network(state)
    
    print(f"State shape: {state.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Values shape: {values.shape}")
    
    assert logits.shape == (4, action_size), "Wrong logits shape!"
    assert values.shape == (4, 1), "Wrong values shape!"
    print("✓ PASSED\n")
    
    print("=" * 70)
    print("Test Case 2: Mask Logits")
    print("=" * 70)
    
    ppo = PPO(state_size, action_size)
    
    logits = torch.randn(2, action_size)
    action_mask = torch.ones(2, action_size, dtype=torch.float32)
    action_mask[0, :100] = 0  # Mask first 100 actions for first sample
    
    masked_logits = ppo.mask_logits(logits, action_mask)
    
    print(f"Original logits shape: {logits.shape}")
    print(f"Masked logits shape: {masked_logits.shape}")
    print(f"Masked positions have -inf: {torch.isinf(masked_logits[0, :100]).all()}")
    
    assert torch.isinf(masked_logits[0, :100]).all(), "Masked logits should be -inf!"
    print("✓ PASSED\n")
    
    print("=" * 70)
    print("Test Case 3: Softmax Sampling")
    print("=" * 70)
    
    logits = torch.randn(1, 10)
    action_mask = torch.ones(1, 10, dtype=torch.bool)
    action_mask[0, :5] = False  # Mask first 5 actions
    
    masked_logits = ppo.mask_logits(logits, action_mask.float())
    probs = F.softmax(masked_logits, dim=-1)
    # Handle NaN without in-place operation
    probs = torch.where(torch.isnan(probs), torch.zeros_like(probs), probs)
    
    print(f"Masked probs shape: {probs.shape}")
    print(f"Probs sum: {probs.sum().item():.6f}")
    print(f"Masked prob values: {probs[0, :5]}")
    print(f"Valid prob values: {probs[0, 5:]}")
    
    assert not torch.isnan(probs).any(), "Should not have NaN probs!"
    assert abs(probs.sum().item() - 1.0) < 0.01, "Probs should sum to 1!"
    print("✓ PASSED\n")
    
    print("=" * 70)
    print("Test Case 4: Store Trajectories")
    print("=" * 70)
    
    memory = Memory()
    
    for i in range(10):
        state = np.random.randn(state_size)
        action = i % action_size
        reward = np.random.randn()
        value = np.random.randn()
        log_prob = np.random.randn()
        action_mask = np.ones(action_size)
        done = i == 9
        
        memory.add(state, action, reward, value, log_prob, action_mask, done)
    
    print(f"Stored transitions: {len(memory.states)}")
    print(f"Expected: 10")
    
    assert len(memory.states) == 10, "Should store 10 transitions!"
    assert len(memory.actions) == 10, "Should store 10 actions!"
    print("✓ PASSED\n")
    
    print("=" * 70)
    print("Test Case 5: GAE Advantage Computation")
    print("=" * 70)
    
    ppo = PPO(state_size, action_size)
    
    # Add some transitions
    for i in range(5):
        state = np.random.randn(state_size)
        action = i % action_size
        reward = float(i)
        value = float(i)
        log_prob = float(-i)
        action_mask = np.ones(action_size)
        done = False
        
        ppo.store_transition(state, action, reward, value, log_prob, action_mask, done)
    
    next_value = 0.0
    advantages, returns = ppo.compute_gae_advantages(next_value)
    
    print(f"Advantages shape: {advantages.shape}")
    print(f"Returns shape: {returns.shape}")
    print(f"Advantages mean: {advantages.mean().item():.6f}")
    print(f"Returns mean: {returns.mean().item():.6f}")
    
    assert advantages.shape[0] == 5, "Should have 5 advantages!"
    assert returns.shape[0] == 5, "Should have 5 returns!"
    print("✓ PASSED\n")
    
    print("=" * 70)
    print("Test Case 6: PPO Update with Mini-batch")
    print("=" * 70)
    
    ppo = PPO(state_size, action_size, learning_rate=1e-3)
    
    # Add transitions
    for i in range(20):
        state = np.random.randn(state_size)
        action = np.random.randint(0, action_size)
        reward = np.random.randn() + 1.0
        value = np.random.randn()
        log_prob = np.random.randn()
        action_mask = np.ones(action_size)
        action_mask[action] = 1  # Ensure sampled action is valid
        # Set mask untuk ensure valid action
        if action < 100:
            action_mask[:action] = 1
            action_mask[action] = 1
        done = i == 19
        
        ppo.store_transition(state, action, reward, value, log_prob, action_mask, done)
    
    initial_loss = None
    
    # Get before update
    states, actions, _, _, _, _, _ = ppo.memory.get_batch()
    with torch.no_grad():
        logits, values = ppo.network(states)
    before_values = values.clone()
    
    # Update
    ppo.update(next_value=0.0, num_epochs=2, batch_size=8)
    
    # Check after update
    with torch.no_grad():
        logits, values = ppo.network(states)
    after_values = values.clone()
    
    print(f"Transitions stored: {len(ppo.memory.states)}")
    print(f"Before update - Values mean: {before_values.mean().item():.6f}")
    print(f"After update - Values mean: {after_values.mean().item():.6f}")
    print(f"Memory cleared after update: {len(ppo.memory.states) == 0}")
    
    assert len(ppo.memory.states) == 0, "Memory should be cleared!"
    print("✓ PASSED\n")
    
    print("=" * 70)
    print("All PPO tests passed!")
    print("=" * 70)