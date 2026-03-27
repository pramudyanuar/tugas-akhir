"""MCTS Node definition for tree search."""

import numpy as np


class MCTSNode:
    """
    Node dalam MCTS tree.
    
    Attributes:
        state: Environment state
        parent: Parent node
        children: Dict of child nodes {action: node}
        action: Action yang menghasilkan node ini dari parent
        visits: Visit count untuk UCB calculation
        value: Accumulated value dari node
        untried_actions: List of actions yang belum dicoba
    """
    
    def __init__(self, state, parent=None, action=None, untried_actions=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.value = 0.0
        self.untried_actions = untried_actions if untried_actions is not None else []
    
    def ucb_value(self, c=1.4):
        """
        Calculate UCB1 value untuk exploration-exploitation tradeoff.
        
        UCB1 = (value/visits) + c * sqrt(ln(parent_visits) / visits)
        
        Args:
            c (float): Exploration constant (default 1.4)
            
        Returns:
            float: UCB1 value
        """
        if self.visits == 0:
            return float('inf')  # Prioritize unvisited nodes
        
        exploitation = self.value / self.visits
        exploration = c * np.sqrt(np.log(self.parent.visits) / self.visits) if self.parent else 0
        
        return exploitation + exploration
    
    def best_child(self, c=1.4):
        """
        Select child dengan highest UCB value.
        
        Args:
            c (float): Exploration constant
            
        Returns:
            MCTSNode: Child node dengan highest UCB value
        """
        return max(self.children.values(), key=lambda child: child.ucb_value(c))
    
    def backup(self, value):
        """
        Backprop value melalui tree.
        
        Args:
            value (float): Value yang akan di-backprop
        """
        self.visits += 1
        self.value += value
        
        if self.parent:
            self.parent.backup(value)
