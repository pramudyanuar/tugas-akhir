import numpy as np
from src.common.mcts_node import MCTSNode
from src.utils.item_utils import get_item_dims, get_item_stacking, make_item


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

    def search_rearrangement(self, failed_item=None, max_unpack=3, apply_to_env=False):
        """
        Rearrangement MCTS for deadlock handling.

        Phases:
        1) Selection (UCB)
        2) Expansion (unpack top-k items)
        3) Simulation (repack unpacked items + failed item)
        4) Backpropagation

        Args:
            failed_item (tuple|None): Item yang gagal dipasang (l, w, h)
            max_unpack (int): Batas maksimum jumlah item top-most untuk di-unpack
            apply_to_env (bool): Apply best simulated rearrangement ke environment utama

        Returns:
            dict: Rearrangement search result
        """
        if failed_item is None:
            if self.env.current_index >= len(self.env.items):
                return {
                    'success': False,
                    'best_sequence': [],
                    'best_value': 0.0,
                    'tree_stats': {'total_simulations': 0, 'num_nodes_expanded': 0},
                    'applied': False,
                }
            failed_item = self.env.items[self.env.current_index]

        root_snapshot = self._capture_env_snapshot()
        root_actions = self._generate_unpack_actions(root_snapshot, max_unpack=max_unpack)

        if len(root_actions) == 0:
            return {
                'success': False,
                'best_sequence': [],
                'best_value': 0.0,
                'tree_stats': {'total_simulations': 0, 'num_nodes_expanded': 0},
                'applied': False,
            }

        root = MCTSNode(root_snapshot, untried_actions=root_actions)

        for _ in range(self.budget):
            node = self._tree_policy_rearrangement(root, max_unpack=max_unpack)
            reward, sim_snapshot, success = self._simulate_repack_rollout(node.state, failed_item)
            node.rollout_snapshot = sim_snapshot
            node.rollout_success = success
            node.backup(reward)

        if len(root.children) == 0:
            return {
                'success': False,
                'best_sequence': [],
                'best_value': 0.0,
                'tree_stats': {'total_simulations': root.visits, 'num_nodes_expanded': 0},
                'applied': False,
            }

        best_child = max(
            root.children.values(),
            key=lambda n: (n.value / max(n.visits, 1), n.visits)
        )

        best_value = best_child.value / max(best_child.visits, 1)
        applied = False
        if apply_to_env and getattr(best_child, 'rollout_success', False):
            self._apply_env_snapshot(best_child.rollout_snapshot)
            applied = True

        return {
            'success': bool(getattr(best_child, 'rollout_success', False)),
            'best_sequence': list(best_child.action) if best_child.action is not None else [],
            'best_value': best_value,
            'tree_stats': {
                'total_simulations': root.visits,
                'num_nodes_expanded': len(root.children),
                'best_action_visits': best_child.visits,
            },
            'applied': applied,
        }
    
    def search(self, state, action_mask, depth_limit=20):
        """
        Run MCTS search dan return best action.
        
        With early termination: stops when best action has sufficient visits.
        
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
        
        # Early termination parameters
        early_term_ratio = 0.7  # Stop if best action has 70% of visits
        early_term_threshold = min(self.budget // 3, 10)  # Min simulations before early termination
        
        # Run simulations with early termination
        for sim in range(self.budget):
            # Selection & Expansion
            node = self._tree_policy(root, valid_actions)
            
            # Simulation (rollout)
            reward = self._default_policy(node.state, action_mask, depth_limit)
            
            # Backprop
            node.backup(reward)
            
            # Early termination check: if best action dominant, stop
            if sim > early_term_threshold and len(root.children) > 0:
                best_child = max(root.children.values(), key=lambda n: n.visits)
                if best_child.visits > (root.visits * early_term_ratio):
                    # Best action is dominant, stop searching
                    break
        
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
                    pos_x = action % self.env.L
                    pos_y = action // self.env.L
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

    def _tree_policy_rearrangement(self, node, max_unpack=3):
        """Selection + expansion policy for rearrangement tree."""
        current_node = node

        while len(current_node.children) > 0 and len(current_node.untried_actions) == 0:
            current_node = current_node.best_child(self.c)

        if len(current_node.untried_actions) > 0:
            action = current_node.untried_actions.pop()
            child_snapshot = self._simulate_unpack(current_node.state, action)
            child_actions = self._generate_unpack_actions(child_snapshot, max_unpack=max_unpack)
            child_node = MCTSNode(
                child_snapshot,
                parent=current_node,
                action=action,
                untried_actions=child_actions
            )
            current_node.children[action] = child_node
            return child_node

        return current_node

    def _capture_env_snapshot(self):
        """Capture mutable environment parts for simulation and restoration."""
        return {
            'height_map': self.env.height_map.map.copy(),
            'placed_items': list(self.env.placed_items),
            'placed_positions': list(self.env.placed_positions),
            'top_item_map': self.env.top_item_map.copy() if hasattr(self.env, 'top_item_map') else None,
            'current_index': int(self.env.current_index),
            'episode_reward': float(self.env.episode_reward),
            'episode_length': int(self.env.episode_length),
            'unpacked_items': [],
            'unpack_sequence': [],
        }

    def _apply_env_snapshot(self, snapshot):
        """Restore environment mutable parts from snapshot."""
        self.env.height_map.map = snapshot['height_map'].copy()
        self.env.placed_items = list(snapshot['placed_items'])
        self.env.placed_positions = list(snapshot['placed_positions'])
        if snapshot.get('top_item_map') is not None and hasattr(self.env, 'top_item_map'):
            self.env.top_item_map = snapshot['top_item_map'].copy()
        self.env.current_index = int(snapshot['current_index'])
        self.env.episode_reward = float(snapshot.get('episode_reward', self.env.episode_reward))
        self.env.episode_length = int(snapshot.get('episode_length', self.env.episode_length))

    def _generate_unpack_actions(self, snapshot, max_unpack=3):
        """Generate unpack actions as tuple of top-most item indices."""
        n_items = len(snapshot['placed_items'])
        if n_items == 0:
            return []

        top_indices = self._top_most_item_indices(snapshot)
        if len(top_indices) == 0:
            return []

        max_k = max(1, min(max_unpack, len(top_indices)))
        actions = []
        for k in range(1, max_k + 1):
            actions.append(tuple(top_indices[:k]))
        return actions

    def _top_most_item_indices(self, snapshot):
        """Sort item indices by top surface height descending."""
        ranked = []
        for idx, (pos, item) in enumerate(zip(snapshot['placed_positions'], snapshot['placed_items'])):
            x, y, z = pos
            l, w, h = get_item_dims(item)
            ranked.append((z + h, idx))
        ranked.sort(reverse=True)
        return [idx for _, idx in ranked]

    def _simulate_unpack(self, snapshot, unpack_indices):
        """Apply unpack action on snapshot and rebuild height map from remaining items."""
        unpack_set = set(unpack_indices)

        remaining_items = []
        remaining_positions = []
        removed_items = []

        for idx, (item, pos) in enumerate(zip(snapshot['placed_items'], snapshot['placed_positions'])):
            if idx in unpack_set:
                removed_items.append(item)
            else:
                remaining_items.append(item)
                remaining_positions.append(pos)

        new_height_map = np.zeros_like(snapshot['height_map'])
        top_item_map = np.full(new_height_map.shape, -1, dtype=np.int32)
        for idx, ((x, y, z), item) in enumerate(zip(remaining_positions, remaining_items)):
            item_l, item_w, item_h = get_item_dims(item)
            new_height_map[x:x + item_l, y:y + item_w] = z + item_h
            top_item_map[x:x + item_l, y:y + item_w] = idx

        return {
            'height_map': new_height_map,
            'placed_items': remaining_items,
            'placed_positions': remaining_positions,
            'top_item_map': top_item_map,
            'current_index': snapshot['current_index'],
            'episode_reward': snapshot.get('episode_reward', 0.0),
            'episode_length': snapshot.get('episode_length', 0),
            'unpacked_items': list(snapshot.get('unpacked_items', [])) + removed_items,
            'unpack_sequence': list(snapshot.get('unpack_sequence', [])) + [tuple(unpack_indices)],
        }

    def _simulate_repack_rollout(self, snapshot, failed_item):
        """
        Simulate repacking unpacked items + failed item, and evaluate reward.
        """
        original = self._capture_env_snapshot()

        try:
            self._apply_env_snapshot(snapshot)

            candidate_items = list(snapshot.get('unpacked_items', [])) + [failed_item]
            if len(candidate_items) == 0:
                return 0.0, self._capture_env_snapshot(), False

            # Pack larger items first for better fit chance.
            candidate_items.sort(
                key=lambda it: get_item_dims(it)[0] * get_item_dims(it)[1] * get_item_dims(it)[2],
                reverse=True,
            )

            placed_now = 0
            for item in candidate_items:
                item_l, item_w, item_h = get_item_dims(item)
                action = self._find_first_valid_action(
                    item_l,
                    item_w,
                    item_h,
                    item_stacking=get_item_stacking(item),
                )
                if action is None:
                    continue

                x = action % self.env.L
                y = action // self.env.L
                base_height = self.env.height_map.max_height_in_region(x, y, item_l, item_w)
                new_height = base_height + item_h
                self.env.height_map.update_region(x, y, item_l, item_w, new_height)
                stacking = get_item_stacking(item)
                self.env.placed_items.append(make_item(item_l, item_w, item_h, stacking))
                self.env.placed_positions.append((x, y, base_height))
                if hasattr(self.env, 'top_item_map'):
                    placed_index = len(self.env.placed_items) - 1
                    self.env.top_item_map[x:x + item_l, y:y + item_w] = placed_index
                placed_now += 1

            success_rate = placed_now / max(len(candidate_items), 1)
            utilization = self.env.get_utilization() / 100.0
            reward = 0.6 * utilization + 0.4 * success_rate
            if placed_now < len(candidate_items):
                reward -= 0.1

            final_snapshot = self._capture_env_snapshot()
            final_snapshot['unpack_sequence'] = list(snapshot.get('unpack_sequence', []))
            return reward, final_snapshot, placed_now == len(candidate_items)
        finally:
            self._apply_env_snapshot(original)

    def _find_first_valid_action(self, item_l, item_w, item_h, item_stacking=None):
        """Find first legal placement action for an item on current env snapshot."""
        for y in range(self.env.W):
            for x in range(self.env.L):
                if self.env._is_valid_position(x, y, item_l, item_w, item_h, item_stacking):
                    return y * self.env.L + x
        return None
