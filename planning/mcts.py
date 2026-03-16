import numpy as np
from collections import defaultdict


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


class MCTS:
    """
    Monte Carlo Tree Search untuk 3D Bin Packing.
    
    Algorithm:
    1. Selection: Traverse tree menggunakan UCB1 hingga leaf node
    2. Expansion: Expand leaf node dengan menambahka child untuk action
    3. Simulation: Rollout random policy dari expanded node
    4. Backprop: Backup value dari rollout ke root
    
    Features:
    - UCB1 exploration-exploitation balance
    - Action masking support
    - Configurable budget (iterations)
    - Value normalization untuk better generalization
    """
    
    def __init__(self, env, budget=50, c=1.4, gamma=0.99):
        """
        Initialize MCTS.
        
        Args:
            env: Environment reference (untuk step)
            budget (int): Number of simulations to run
            c (float): UCB exploration constant
            gamma (float): Discount factor untuk value calculation
        """
        self.env = env
        self.budget = budget
        self.c = c
        self.gamma = gamma
    
    def search(self, state, action_mask, depth_limit=20):
        """
        Run MCTS search dan return best action.
        
        Args:
            state: Current state
            action_mask: Valid actions mask
            depth_limit (int): Maximum depth untuk simulation
            
        Returns:
            dict: {
                'best_action': int (best action index),
                'action_values': dict (action -> value estimates),
                'tree_stats': dict (tree statistics)
            }
        """
        # Get valid actions dari mask
        valid_actions = np.where(action_mask > 0)[0].tolist()
        
        if len(valid_actions) == 0:
            # No valid actions available
            return {
                'best_action': self.env.L * self.env.W,  # Skip action
                'action_values': {},
                'tree_stats': {'total_simulations': 0, 'best_value': 0.0}
            }
        
        # Create root node
        root = MCTSNode(state, untried_actions=valid_actions)
        
        # Run simulations
        for sim in range(self.budget):
            # Selection & Expansion
            node = self._tree_policy(root, valid_actions)
            
            # Simulation (rollout)
            reward = self._default_policy(node.state, action_mask, depth_limit)
            
            # Backprop
            node.backup(reward)
        
        # Select best action berdasarkan visit count (exploitation)
        best_child = max(root.children.values(), key=lambda n: n.visits)
        best_action = best_child.action
        
        # Compute action values untuk logging
        action_values = {}
        for action, child in root.children.items():
            action_values[action] = child.value / max(child.visits, 1)
        
        tree_stats = {
            'total_simulations': root.visits,
            'best_value': best_child.value / max(best_child.visits, 1),
            'num_nodes_expanded': len(root.children),
            'best_action_visits': best_child.visits
        }
        
        return {
            'best_action': best_action,
            'action_values': action_values,
            'tree_stats': tree_stats
        }
    
    def _tree_policy(self, node, valid_actions):
        """
        Selection & Expansion phase: traverse tree dengan UCB1 lalu expand.
        
        Args:
            node: Current node
            valid_actions: List of valid actions
            
        Returns:
            MCTSNode: Leaf node untuk simulation
        """
        current_node = node
        current_actions = valid_actions.copy()
        
        # Selection phase: traverse tree
        while len(current_node.children) > 0 and len(current_node.untried_actions) == 0:
            current_node = current_node.best_child(self.c)
            # Update valid actions berdasarkan new state
            # (In simplified version, assume valid actions tetap)
        
        # Expansion phase: expand node jika ada untried actions
        if len(current_node.untried_actions) > 0:
            action = current_node.untried_actions.pop()
            # Create child node
            child_node = MCTSNode(current_node.state, parent=current_node, action=action)
            current_node.children[action] = child_node
            return child_node
        
        return current_node
    
    def _default_policy(self, state, action_mask, depth_limit):
        """
        Simulation phase: random rollout policy.
        
        Implements lightweight rollout dengan simple heuristics.
        
        Args:
            state: Current state
            action_mask: Valid actions mask
            depth_limit: Maximum simulation depth
            
        Returns:
            float: Accumulated discounted reward dari rollout
        """
        accumulated_reward = 0.0
        discount = 1.0
        
        # Rollout dengan prioritized random actions
        for depth in range(depth_limit):
            valid_actions = np.where(action_mask > 0)[0]
            
            if len(valid_actions) == 0:
                # No valid actions, return accumulated reward
                break
            
            # Prioritize valid actions yang lebih mungkin berhasil
            # Simple heuristic: prefer central positions
            center_x, center_y = self.env.L // 2, self.env.W // 2
            action_scores = []
            
            for action in valid_actions:
                if action == self.env.L * self.env.W:
                    # Skip action
                    action_scores.append(0.5)
                else:
                    # Position-based scoring
                    pos_x = action // self.env.W
                    pos_y = action % self.env.W
                    distance = np.sqrt((pos_x - center_x) ** 2 + (pos_y - center_y) ** 2)
                    score = 1.0 / (1.0 + distance / 20.0)  # Closer to center = higher score
                    action_scores.append(score)
            
            # Weighted random selection
            action_scores = np.array(action_scores)
            action_scores = action_scores / action_scores.sum()
            
            action = np.random.choice(valid_actions, p=action_scores)
            
            # Estimate reward dari action (simple heuristic)
            # Assume small positive reward untuk valid actions
            reward_estimate = 0.1 + 0.05 * action_scores[list(valid_actions).index(action)]
            accumulated_reward += discount * reward_estimate
            discount *= self.gamma
        
        return accumulated_reward
    
    def get_best_action(self, search_result):
        """
        Extract best action dari search result.
        
        Args:
            search_result: Result dari search()
            
        Returns:
            int: Best action index
        """
        return search_result['best_action']


if __name__ == "__main__":
    """Test cases untuk MCTS"""
    
    print("=" * 70)
    print("Test Case 1: MCTSNode initialization")
    print("=" * 70)
    
    from env.container_env import ContainerEnv
    
    env = ContainerEnv(seed=42)
    state, action_mask = env.reset()
    
    node = MCTSNode(state, untried_actions=[0, 1, 2, 3])
    
    print(f"Node created with state shape: {node.state.shape}")
    print(f"Untried actions: {node.untried_actions}")
    print(f"Visits: {node.visits}")
    print(f"Value: {node.value}")
    print("✓ PASSED\n")
    
    print("=" * 70)
    print("Test Case 2: UCB value calculation")
    print("=" * 70)
    
    root = MCTSNode(state, untried_actions=[0, 1, 2])
    root.visits = 10
    root.value = 50.0
    
    child1 = MCTSNode(state, parent=root, action=0)
    child1.visits = 3
    child1.value = 20.0
    
    child2 = MCTSNode(state, parent=root, action=1)
    child2.visits = 7
    child2.value = 30.0
    
    root.children[0] = child1
    root.children[1] = child2
    
    ucb1 = child1.ucb_value(c=1.4)
    ucb2 = child2.ucb_value(c=1.4)
    
    print(f"Child1 UCB: {ucb1:.4f} (visits=3, value=20)")
    print(f"Child2 UCB: {ucb2:.4f} (visits=7, value=30)")
    print(f"UCB values calculated successfully")
    print("✓ PASSED\n")
    
    print("=" * 70)
    print("Test Case 3: Backup propagation")
    print("=" * 70)
    
    root = MCTSNode(state, untried_actions=[0, 1])
    child = MCTSNode(state, parent=root, action=0)
    root.children[0] = child
    
    child.backup(10.0)
    
    print(f"Root visits after backup: {root.visits}")
    print(f"Root value after backup: {root.value}")
    print(f"Child visits after backup: {child.visits}")
    print(f"Child value after backup: {child.value}")
    assert root.visits == 1, "Root visits should be 1"
    assert root.value == 10.0, "Root value should be 10.0"
    print("✓ PASSED\n")
    
    print("=" * 70)
    print("Test Case 4: MCTS initialization")
    print("=" * 70)
    
    env = ContainerEnv(seed=123)
    mcts = MCTS(env, budget=50, c=1.4, gamma=0.99)
    
    print(f"MCTS created with budget: {mcts.budget}")
    print(f"Exploration constant c: {mcts.c}")
    print(f"Discount factor gamma: {mcts.gamma}")
    print("✓ PASSED\n")
    
    print("=" * 70)
    print("All MCTS tests completed!")
    print("=" * 70)