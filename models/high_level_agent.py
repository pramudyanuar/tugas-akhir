"""High-Level Manager Agent for Hierarchical RL.

PART OF HIERARCHICAL RL:
  - HIGH-LEVEL: HighLevelAgent (this file) -> What strategy?
  - LOW-LEVEL: PPO + ActorCriticNetwork (agents/ppo.py + models/actor_critic.py) -> Where to place?

RELATIONSHIP:
  HighLevelAgent selects STRATEGY (orientation, repack, skip):
    1. Provides strategy to low-level PPO agent
    2. PPO then selects WHERE to place (position) given that strategy
    3. Together: WHAT strategy + WHERE to place = Complete decision

EXAMPLE DECISION:
  Container state: [height_map, item_width, item_height,  item_depth]
                        |
                        v
              HighLevelAgent -> Strategy (e.g., "place vertically rotated")
                        |
                        v
              PPO Agent -> Position (e.g., "place at (x=10, y=5)")
                        |
                        v
              Environment -> Execute action & get reward

HIERARCHICAL BENEFITS:
  - Search space reduction: Strategy narrows position search
  - Better exploration: High-level helps avoid local minima
  - Improved stability: Clear separation of concerns
  - Easier debugging: Each level has clear responsibility
"""

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict

from env.lbcp import LBCPClusterer


class HighLevelAgent(nn.Module):
    """High-Level Agent for Strategy Selection in Hierarchical RL.
    
    RESPONSIBILITIES (What Strategy?):
    1. LBCP clustering: Group items for balanced load distribution
    2. Strategy selection: Decide strategy (orientation, repack, skip)
    3. Guidance: Provide strategy to low-level PPO agent
    
    HOW IT WORKS:
    1. Input: Container state (height_map + item dimensions)
    2. Process: Neural network selects best strategy
    3. Output: Strategy (one of num_strategies options)
           e.g., [place-as-is, rotate-180, repack, skip, ...]
    
    TRAINING:
    - Not directly trained this phase (for future enhancement)
    - Currently uses LBCP heuristic for clustering
    - Can be jointly trained with PPO for end-to-end learning
    
    INTERACTION WITH LOW-LEVEL:
    train.py._select_hierarchical_action() uses:
      strategy = high_level_agent.select_action()
      position = ppo_agent.select_action(state, strategy)
    
    REFERENCE TO OLD APPROACH:
    Deprecated LowLevelAgent (models/low_level_agent.py) tried to do both
    strategy selection AND position selection in one network.
    Separation improves modularity and training stability.
    """
    
    def __init__(self, input_dim=59*23 + 3, hidden_dim=256, num_strategies=8):
        """
        Initialize High-Level Agent.
        
        Args:
            input_dim (int): Input dimension (height_map flattened + item dims)
            hidden_dim (int): Hidden dimension
            num_strategies (int): Number of high-level strategies
                - 6 orientations
                - 1 repack action
                - 1 no-op
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_strategies = num_strategies
        
        # Shared network layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Strategy head: output logits untuk strategy selection
        self.strategy_head = nn.Linear(hidden_dim, num_strategies)
        
        # LBCP integration
        self.lbcp_module = LBCPClusterer(num_clusters=4)
        
        # State tracking
        self.current_clusters = None
        self.cluster_assignment = None
    
    def forward(self, state, items_batch=None):
        """
        Forward pass untuk high-level decision making.
        
        Args:
            state: Normalized state tensor (batch_size or 1, input_dim)
            items_batch: Optional list of items untuk clustering
            
        Returns:
            dict: {
                'strategy_logits': tensor of strategy logits,
                'cluster_assignment': cluster assignment untuk items,
                'load_balance': load balance coefficient
            }
        """
        # Ensure state is 2D
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        # Forward pass through network
        x = self.fc1(state)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        
        # Strategy logits
        strategy_logits = self.strategy_head(x)
        
        # LBCP clustering jika ada items
        cluster_assignment = None
        load_balance = 1.0
        
        if items_batch is not None and len(items_batch) > 0:
            clusters = self.lbcp_module.cluster_by_weight(items_batch)
            cluster_assignment = clusters
            load_balance = self.lbcp_module.compute_load_balance()
            self.current_clusters = clusters
            self.cluster_assignment = cluster_assignment
        
        return {
            'strategy_logits': strategy_logits,
            'cluster_assignment': cluster_assignment,
            'load_balance': load_balance
        }
    
    def select_strategy(self, strategy_logits, sample=True):
        """
        Select strategy berdasarkan logits.
        
        Strategies:
        - 0-5: 6 possible orientations
        - 6: Repack action
        - 7: No-op
        
        Args:
            strategy_logits: Output dari strategy head (batch_size, num_strategies) or (num_strategies,)
            sample (bool): If True, sample from distribution. If False, use greedy argmax.
            
        Returns:
            tuple: (strategy_index, log_prob) - strategy is int, log_prob is float
        """
        # Ensure 2D
        if strategy_logits.dim() == 1:
            strategy_logits = strategy_logits.unsqueeze(0)
        
        # Compute probabilities
        probs = torch.softmax(strategy_logits, dim=-1)  # (batch_size, num_strategies)
        
        if sample:
            # Sample from distribution (allows gradient flow during training)
            dist = torch.distributions.Categorical(probs)
            strategy_idx = dist.sample()  # (batch_size,)
            log_prob = dist.log_prob(strategy_idx)  # (batch_size,)
            
            # Return first element since we batch with size 1
            return strategy_idx[0].item(), log_prob[0].item()
        else:
            # Greedy selection (for evaluation)
            strategy_idx = torch.argmax(probs, dim=-1)[0]
            log_prob = torch.log(probs[0, strategy_idx] + 1e-10)
            return strategy_idx.item(), log_prob.item()

    def decode_macro_decision(self, strategy):
        """
        Decode high-level strategy index menjadi macro decision.

        Returns:
            dict: {
                'orientation': int,
                'zone_priority': str,
                'allow_repacking': bool
            }
        """
        orientation = min(max(int(strategy), 0), 5)

        # Mapping sederhana orientation -> zone priority agar high-level decision
        # memengaruhi urutan candidate yang dievaluasi low-level policy.
        zone_map = {
            0: 'left_to_right',
            1: 'right_to_left',
            2: 'front_to_back',
            3: 'back_to_front',
            4: 'center',
            5: 'center',
        }

        return {
            'orientation': orientation,
            'zone_priority': zone_map.get(orientation, 'center'),
            'allow_repacking': int(strategy) == 6
        }
    
    def get_item_ordering(self, items, strategy):
        """
        Determine item ordering berdasarkan strategy.
        
        Args:
            items: List of items
            strategy: Selected strategy
            
        Returns:
            list: Ordered items
        """
        if strategy == 6:  # Repack strategy
            # Inverse order (LIFO)
            return list(reversed(items))
        elif strategy == 7:  # No-op strategy
            # Keep original order
            return items
        else:  # Orientation strategies (0-5)
            # Sort by largest dimension first
            return sorted(items, key=lambda x: max(x[0], x[1], x[2]), reverse=True)
    
    def get_load_balance_reward(self):
        """
        Get reward berdasarkan load balance.
        
        Returns:
            float: Load balance reward (0.0 to 1.0)
        """
        if self.current_clusters is None:
            return 0.0
        
        return self.lbcp_module.compute_load_balance()
    
    def compute_strategy_loss(self, strategy_logits, strategy_log_prob, reward, entropy_coef=0.01):
        """
        Compute loss untuk strategy learning menggunakan policy gradient.
        
        Args:
            strategy_logits: (1, num_strategies) - output dari forward pass
            strategy_log_prob: float - log probability dari selected strategy
            reward: float - reward signal (e.g., load balance or episodic return)
            entropy_coef: float - coefficient untuk entropy regularization
            
        Returns:
            torch.Tensor: Strategy loss untuk backprop
        """
        # Policy gradient loss: -log(pi) * advantage
        # Advantage approximated by reward
        strategy_log_prob_tensor = torch.tensor(strategy_log_prob, requires_grad=False)
        reward_tensor = torch.tensor(reward, requires_grad=False)
        
        # Policy loss: encourage good strategies, discourage bad ones
        policy_loss = -(strategy_log_prob_tensor * reward_tensor)
        
        # Entropy bonus untuk exploration
        probs = torch.softmax(strategy_logits, dim=-1)
        entropy = torch.distributions.Categorical(probs).entropy()
        entropy_loss = -entropy_coef * entropy
        
        total_loss = policy_loss + entropy_loss
        
        return total_loss
