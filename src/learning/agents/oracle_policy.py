"""Oracle/Baseline policies for 3D container loading comparison."""

import numpy as np
from src.core.lbcp import LBCPClusterer


class OraclePolicy:
    """Greedy baseline policy for 3D container loading.

    The policy always chooses from the same valid action mask as the learned
    A3C low-level agent, so comparisons stay fair: all baselines obey the same
    geometric and stability filters from the environment.
    """
    
    def __init__(self, env, priority='load_balance'):
        """
        Initialize oracle policy.
        
        Args:
            env: ContainerEnv instance
            priority (str): 'load_balance', 'height', 'nearest_center',
                            'dblf', atau 'bottom_left_front'
        """
        self.env = env
        self.priority = priority
        self.lbcp = LBCPClusterer(num_clusters=4)
    
    def select_action(self, state, action_mask, max_items=None):
        """
        Select action menggunakan greedy heuristic.
        
        Args:
            state: Current state
            action_mask: Valid actions mask
            max_items: Optional max items in batch (for LBCP clustering)
            
        Returns:
            int: Selected action index (position in flattened grid)
        """
        valid_actions = np.where(action_mask > 0)[0]
        
        if len(valid_actions) == 0:
            # No valid actions, place at end (skip action)
            return self.env.L * self.env.W
        
        if self.priority == 'load_balance':
            return self._select_load_balance(valid_actions)
        elif self.priority == 'height':
            return self._select_minimal_height(valid_actions)
        elif self.priority == 'nearest_center':
            return self._select_nearest_center(valid_actions)
        elif self.priority in ('dblf', 'bottom_left_front', 'blf'):
            return self._select_dblf(valid_actions)
        else:
            # Default: random valid action
            return valid_actions[np.random.randint(len(valid_actions))]

    def _coords(self, action):
        """Convert flattened action index to (x, y) grid coordinate."""
        return action % self.env.L, action // self.env.L

    def _height_at(self, x, y):
        """Read height map value using the environment's (x, y) convention."""
        try:
            return self.env.height_map.map[x, y]
        except AttributeError:
            return self.env.height_map[x, y]
    
    def _select_load_balance(self, valid_actions):
        """
        Select position that best maintains load balance.
        
        Strategy: Choose position yang menghasilkan distribution load terseimbang.
        """
        if len(valid_actions) == 1:
            return valid_actions[0]
        
        best_action = valid_actions[0]
        best_balance = -np.inf
        
        for action in valid_actions:
            # Get 2D position from action index
            if action < self.env.L * self.env.W:
                x, y = self._coords(action)
                
                # Prefer positions near the center
                center_x, center_y = self.env.L / 2, self.env.W / 2
                distance_to_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                
                # Also prefer lower positions (better stability)
                min_height = self._height_at(x, y)
                height_score = 1.0 / (1 + min_height)  # Prefer lower heights
                
                # Combined score: balance + height preference
                score = height_score - 0.1 * distance_to_center
                
                if score > best_balance:
                    best_balance = score
                    best_action = action
        
        return best_action
    
    def _select_minimal_height(self, valid_actions):
        """
        Select position dengan minimal height (first-fit-decreasing inspired).
        """
        if len(valid_actions) == 1:
            return valid_actions[0]
        
        min_height = np.inf
        best_action = valid_actions[0]
        
        for action in valid_actions:
            if action < self.env.L * self.env.W:
                x, y = self._coords(action)
                height = self._height_at(x, y)
                
                if height < min_height:
                    min_height = height
                    best_action = action
        
        return best_action
    
    def _select_nearest_center(self, valid_actions):
        """
        Select position nearest to container center.
        """
        if len(valid_actions) == 1:
            return valid_actions[0]
        
        center_x, center_y = self.env.L / 2, self.env.W / 2
        best_action = valid_actions[0]
        best_distance = np.inf
        
        for action in valid_actions:
            if action < self.env.L * self.env.W:
                x, y = self._coords(action)
                distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                
                if distance < best_distance:
                    best_distance = distance
                    best_action = action
        
        return best_action

    def _select_dblf(self, valid_actions):
        """Select the deepest-bottom-left-front-like valid action.

        DBLF is commonly used as a constructive 3D packing baseline. In this
        grid-action environment, z is implied by the current height map, so this
        variant chooses the valid position with the lowest support height first,
        then the smallest y/front coordinate, then the smallest x/left
        coordinate.
        """
        if len(valid_actions) == 1:
            return valid_actions[0]

        best_action = valid_actions[0]
        best_key = None

        for action in valid_actions:
            if action >= self.env.L * self.env.W:
                continue
            x, y = self._coords(action)
            key = (self._height_at(x, y), y, x)
            if best_key is None or key < best_key:
                best_key = key
                best_action = action

        return best_action


class RandomPolicy:
    """
    Random baseline policy - selects uniformly random valid actions.
    """
    
    def __init__(self, env):
        """Initialize random policy."""
        self.env = env
    
    def select_action(self, state, action_mask, max_items=None):
        """
        Select random valid action.
        
        Args:
            state: Current state
            action_mask: Valid actions mask
            max_items: Unused
            
        Returns:
            int: Random action index
        """
        valid_actions = np.where(action_mask > 0)[0]
        
        if len(valid_actions) == 0:
            return self.env.L * self.env.W
        
        return valid_actions[np.random.randint(len(valid_actions))]
