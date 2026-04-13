"""
Algorithm 2: Tree Expansion & Simulation
Hierarchical search untuk object placement dengan consideration untuk stability dan utilization.

Main Features:
- Tree expansion untuk candidate actions
- Recursive search untuk valid placement sequences
- Reward-based sorting untuk prioritization
- Object accessibility checking
- Bin fullness checking
"""

import numpy as np
import copy


class TreeExpander:
    """
    Algorithm 2: Tree Expansion Implementation

    The algorithm builds decision tree untuk placement sequences:

    Pseudocode (simplified):
        TREEEXPANSION(v, X, χ, d, n, requireFullPack):
        1. O ← (), n = 0? stop = true : stop = false
        2. for each o ∈ N ∪ C do
        3.     for each ϕ ∈ {0, π/2} do
        4.         a(o,ϕ) ← πlow(s(o,ϕ)) compute r(s(o,ϕ), a(o,ϕ))
        5.         if o accessible and a(o,ϕ) ≠ NoPositionAction:
        6.             stop ← false
        7.         APPEND(O, (o, ϕ))
        8.     for each (o, ϕ) ∈ O:
        9.         χ' ← APPEND(χ, (o, ϕ, a(o,ϕ), d))
        10.        if a(o,ϕ) ≠ NoPositionAction:
        11.            v' ← CHILDNODE(v, o, ϕ, a(o,ϕ))
        12.            full ← BINFULL(v')
        13.        if a(o,ϕ) = NoPositionAction OR full:
        14.            χ̃ ← SEQUENCE(χ')
        15.            X ← X ∪ {χ̃}
        16.            if requireFullPack and full:
        17.                return X, true
        18.        else:
        19.            X, solved ← TREEEXPANSION(v', X, χ', d+1, n', requireFullPack)
    """

    def __init__(self, env, max_depth=20, reward_weights=None):
        """
        Initialize TreeExpander.

        Args:
            env: Environment reference
            max_depth (int): Maximum search depth
            reward_weights (dict): Weights untuk reward computation
                {
                    'utilization': float,
                    'stability': float,
                    'edge_contact': float,
                    'height': float
                }
        """
        self.env = env
        self.max_depth = max_depth
        
        # Default reward weights
        if reward_weights is None:
            reward_weights = {
                'utilization': 1.0,
                'stability': 0.5,
                'edge_contact': 0.3,
                'height': -0.1
            }
        self.reward_weights = reward_weights
        
        # For tracking
        self.sequence_count = 0

    def tree_expansion(self, root_state, placements_tried=None, sequence=None,
                       depth=0, node_count=0, require_full_pack=False):
        """
        Algorithm 2: Tree Expansion

        Ekspansi decision tree dengan consideration untuk placement validity.

        Args:
            root_state (dict): Root state (environment snapshot)
            placements_tried (set): Set of already tried placements
            sequence (list): Current placement sequence
            depth (int): Current depth
            node_count (int): Number of nodes processed
            require_full_pack (bool): Require full bin packing

        Returns:
            tuple: (sequences, solved)
                - sequences: List of valid sequence results
                - solved: Flag if full pack found (when require_full_pack=True)
        """
        sequences = []
        solved = False
        
        if placements_tried is None:
            placements_tried = set()
        if sequence is None:
            sequence = []

        # Line 1: Initialize
        candidates = []  # O in algorithm
        stop = (node_count == 0)  # If first call, might have nothing to place

        # Get current items and placements
        current_items = root_state.get('items', [])
        current_index = root_state.get('current_index', 0)
        
        if current_index >= len(current_items):
            # No more items
            return sequences, solved

        current_item = current_items[current_index]
        l, w, h = current_item

        # Line 2-7: Generate candidate actions (object + orientation)
        # ϕ = rotation angle {0, π/2} = {0, 1} (no rotation, 90 degree rotation)
        for phi in range(2):  # 2 orientations
            # Apply rotation if phi = 1
            if phi == 1:
                item_l, item_w = w, l  # Rotate 90 degrees
            else:
                item_l, item_w = l, w

            # Find best placement untuk this orientation
            action = self._low_level_policy(root_state, item_l, item_w, h)
            
            # Compute reward
            reward = self._compute_reward(root_state, action, item_l, item_w, h)
            
            # Check accessibility
            is_accessible = action is not None and action != (self.env.L * self.env.W,)
            
            if is_accessible:
                stop = False
            
            candidates.append({
                'object_idx': current_index,
                'phi': phi,
                'action': action,
                'reward': reward,
                'is_accessible': is_accessible
            })

        # If nothing accessible, return early
        if stop and len(sequence) == 0:
            return sequences, solved

        # Line 8: Sort candidates by reward
        candidates.sort(key=lambda c: c['reward'], reverse=True)

        # Line 8-18: Process each candidate
        for candidate in candidates:
            phi = candidate['phi']
            action = candidate['action']
            
            # Apply rotation untuk item
            if phi == 1:
                item_l, item_w = w, l
            else:
                item_l, item_w = l, w

            # Update sequence
            new_sequence = sequence + [(current_index, phi, action, depth)]
            
            # Check if can place
            can_place = (action is not None and 
                        action != (self.env.L * self.env.W,) and
                        candidate['is_accessible'])
            
            if not can_place:
                # Line 13-15: Terminal sequence
                sorted_sequence = self._sort_sequence(new_sequence)
                sequences.append(sorted_sequence)
            else:
                # Line 11: Create child node
                child_state = self._create_child_state(root_state, current_index, phi, action)
                
                # Line 12: Check if bin full
                bin_full = self._is_bin_full(child_state)
                
                if bin_full:
                    # Bin full, terminal sequence
                    sorted_sequence = self._sort_sequence(new_sequence)
                    sequences.append(sorted_sequence)
                    
                    if require_full_pack:
                        # Found full pack solution
                        return sequences, True
                else:
                    # Recursively expand
                    child_sequences, child_solved = self.tree_expansion(
                        child_state,
                        placements_tried=placements_tried,
                        sequence=new_sequence,
                        depth=depth + 1,
                        node_count=node_count + 1,
                        require_full_pack=require_full_pack
                    )
                    
                    sequences.extend(child_sequences)
                    
                    if child_solved:
                        return sequences, True

        return sequences, solved

    def _low_level_policy(self, state, item_l, item_w, item_h):
        """
        Low-level policy: find best placement untuk item.

        Args:
            state (dict): Current state
            item_l, item_w, item_h: Item dimensions

        Returns:
            tuple: (x, y, z) position atau None if no valid placement
        """
        height_map = state.get('height_map')
        if height_map is None:
            return None

        # Find position dengan max edge contact
        best_pos = None
        best_score = -np.inf

        for z in range(self.env.H - item_h + 1):
            for y in range(self.env.W - item_w + 1):
                for x in range(self.env.L - item_l + 1):
                    region_max = np.max(height_map[x:x+item_l, y:y+item_w])
                    
                    if region_max + item_h <= self.env.H and region_max == z:
                        score = self._edge_contact_score(height_map, x, y, item_l, item_w)
                        
                        if score > best_score:
                            best_score = score
                            best_pos = (x, y, z)

        return best_pos

    def _compute_reward(self, state, action, item_l, item_w, item_h):
        """
        Compute reward untuk placement.

        Args:
            state (dict): Current state
            action (tuple): Position action
            item_l, item_w, item_h: Item dimensions

        Returns:
            float: Reward value
        """
        if action is None:
            return -10.0  # Penalty untuk invalid action

        x, y, z = action
        
        # Utilization impact
        item_volume = item_l * item_w * item_h
        util_reward = self.reward_weights['utilization'] * (item_volume / self.env.L / self.env.W / self.env.H)
        
        # Edge contact reward
        height_map = state.get('height_map')
        edge_reward = self.reward_weights['edge_contact'] * self._edge_contact_score(
            height_map, x, y, item_l, item_w
        )
        
        # Height penalty (prefer lower positions)
        height_penalty = self.reward_weights['height'] * z
        
        return util_reward + edge_reward + height_penalty

    def _edge_contact_score(self, height_map, x, y, l, w):
        """
        Compute edge contact score untuk position.

        Args:
            height_map (np.ndarray): Current height map
            x, y: Position
            l, w: Item dimensions

        Returns:
            float: Edge contact score
        """
        score = 0.0
        
        # Bottom contact
        if y == 0:
            score += w
        elif y > 0 and np.any(height_map[x:x+l, y-1] > 0):
            score += np.sum(height_map[x:x+l, y-1] > 0)
        
        # Left contact
        if x == 0:
            score += l
        elif x > 0 and np.any(height_map[x-1, y:y+w] > 0):
            score += np.sum(height_map[x-1, y:y+w] > 0)
        
        return score

    def _create_child_state(self, parent_state, item_idx, phi, action):
        """
        Create child state setelah placement.

        Args:
            parent_state (dict): Parent state
            item_idx (int): Item index
            phi (int): Orientation
            action (tuple): Placement action

        Returns:
            dict: Child state
        """
        child_state = copy.deepcopy(parent_state)
        
        # Update current index
        child_state['current_index'] = item_idx + 1
        
        # Update height map dengan item yang baru di-place
        if action is not None:
            x, y, z = action
            items = parent_state.get('items', [])
            if item_idx < len(items):
                l, w, h = items[item_idx]
                
                # Rotate jika phi = 1
                if phi == 1:
                    l, w = w, l
                
                # Update height map
                height_map = child_state.get('height_map')
                if height_map is not None:
                    height_map[x:x+l, y:y+w] = z + h
        
        return child_state

    def _is_bin_full(self, state):
        """
        Check apakah bin sudah full.

        Args:
            state (dict): Current state

        Returns:
            bool: True jika bin full
        """
        height_map = state.get('height_map')
        if height_map is None:
            return False

        # Bin full jika max height >= H
        return np.max(height_map) >= self.env.H

    def _sort_sequence(self, sequence):
        """
        Sort placement sequence (optional - untuk canonical form).

        Args:
            sequence (list): Sequence of placements

        Returns:
            list: Sorted sequence
        """
        return sequence
