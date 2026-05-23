"""CNN-based Actor-Critic Network for the A3C low-level policy.

PART OF HIERARCHICAL RL:
  - HIGH-LEVEL: HighLevelAgent (models/high_level_agent.py) -> Strategy selection
  - LOW-LEVEL: A3C (agents/a3c.py) + ActorCriticNetwork (this file) -> Placement

RELATIONSHIP:
  A3C uses ActorCriticNetwork to:
    1. Process current container state (height_map + item dims)
    2. Output action logits for all valid positions
    3. Output value estimate for state
    4. Learn via policy loss + value loss + entropy regularization

LEGACY:
  models/low_level_agent.py contains old implementation (not used anymore)
"""

import torch
import torch.nn as nn


class ActorCriticNetwork(nn.Module):
    """
    Low-level network: CNN-based Actor-Critic for A3C.
    
    This network is the core of the low-level placement policy.
    It replaces the old LowLevelAgent with a superior CNN architecture.
    
    Architecture:
    - Convolutional layers untuk spatial feature extraction dari height_map (2D grid)
    - Fully connected layers untuk action selection + value estimation
    - Actor head (logits): output untuk each possible container position
    - Critic head (value): scalar value estimate for current state
    
    Used by: agents/a3c.py (A3C algorithm)
    """
    
    def __init__(self, L=59, W=23, action_size=59*23+1, hidden_size=512):
        """
        Initialize Actor-Critic network.
        
        Args:
            L (int): Container length (height_map width)
            W (int): Container width (height_map height)
            action_size (int): Ukuran action space (L*W + 1 untuk skip)
            hidden_size (int): Hidden layer size FC
        """
        super(ActorCriticNetwork, self).__init__()
        
        self.L = L
        self.W = W
        self.action_size = action_size
        
        # CNN layers untuk extract spatial features dari 2D height_map
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Reduce spatial dimensions
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)   # Further reduce spatial dimensions
        )
        
        # FC layers: flattened conv output + item_dims (3) + min_height (1) → hidden → [action logits, value]
        # After 2x2 max pooling twice: spatial dims become (L//4, W//4)
        # State now includes: height_map (L*W) + item_dims (3) + min_height (1) = L*W + 4
        pooled_size = 32 * (L // 4) * (W // 4) + 4  # Updated from +3 to +4 for min_height
        self.fc_shared = nn.Sequential(
            nn.Linear(pooled_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Actor head: output logits
        self.actor = nn.Linear(hidden_size, action_size)
        
        # Critic head: output value
        self.critic = nn.Linear(hidden_size, 1)
        
    def forward(self, state, return_features=False):
        """
        Forward pass untuk actor dan critic.
        
        Args:
            state (torch.Tensor): State tensor
                - Shape: (batch_size, L*W + 3) - flattened height_map + item_dims
                - Will be reshaped to (batch_size, 1, L, W) untuk konv processing
            return_features (bool): If True, return extracted features
            
        Returns:
            tuple: (logits, value) or (logits, value, features) if return_features=True
                - logits: Actor output logits (batch_size, action_size)
                - value: Critic value estimation (batch_size, 1)
                - features: (optional) Extracted features before action/critic heads
        """
        # Handle state tensor reshaping
        batch_size = state.shape[0]
        
        # Separate height_map dan item dimensions
        # State format: [height_map.flatten() (L*W), item_l, item_w, item_h (3), min_height (1)]
        height_map_flat = state[:, :self.L * self.W]
        item_dims = state[:, self.L * self.W:]
        
        # Reshape untuk conv: (batch_size, L*W) → (batch_size, 1, L, W)
        height_map_2d = height_map_flat.view(batch_size, 1, self.L, self.W)
        
        # Process height_map melalui CNN
        x = self.conv(height_map_2d)
        
        # Flatten: (batch_size, 32, L, W) → (batch_size, 32*L*W)
        x = x.view(batch_size, -1)
        
        # Concatenate dengan item dimensions
        x = torch.cat([x, item_dims], dim=1)
        
        # Shared FC layers
        features = self.fc_shared(x)
        
        # Actor dan Critic heads
        logits = self.actor(features)
        value = self.critic(features)
        
        if return_features:
            return logits, value, features
        return logits, value
