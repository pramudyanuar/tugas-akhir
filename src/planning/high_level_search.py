"""
Algorithm 1: High-Level Search
Main search algorithm untuk placement decisions dengan hierarchical control.

Main Features:
- Tree expansion untuk candidate generation
- Simulation dan generation untuk action evaluation
- Best action selection dengan reward-based criteria
- Repacktrial untuk deadlock resolution
"""

import numpy as np
import copy
from .tree_expansion import TreeExpander
from .repack_trial import RepackTrial
from ..core.height_map import HeightMap
from ..core.feasibility_map import FeasibilityMap


class HighLevelSearcher:
    """
    Algorithm 1: High-Level Search Implementation

    Pseudocode:
        HIGH-LEVEL SEARCH(shigh, useRepack, requireFullPack):
        1. vroot ← shigh
        2. X, solved ← TREEEXPANSION(vroot, ∅, (), 0, 0, false)
        3. if X = ∅ then
        4.     ahigh ← (terminate)
        5.     util ← CurrentUtilizationRatio
        6. else
        7.     A ← SIMULATIONANDGENERATION(X)
        8.     χ*, ahigh ← BESTACTIONSELECTION(X, A)
        9.     util ← util(χ*)
        10.    if (X = ∅ or NoPositionAction in χ*) and useRepack then
        11.        a*, RepackSuccess ← REPACKTRIAL(vroot, util, requireFullPack)
        12.        if RepackSuccess then
        13.            ahigh ← a*
        14.    return ahigh
    """

    def __init__(self, env, max_depth=20, mcts_budget=50, use_repack=True, 
                 require_full_pack=False, reward_weights=None):
        """
        Initialize HighLevelSearcher.

        Args:
            env: Environment reference
            max_depth (int): Maximum search depth
            mcts_budget (int): MCTS simulation budget
            use_repack (bool): Whether to use repacking
            require_full_pack (bool): Require full packing
            reward_weights (dict): Weights untuk reward computation
        """
        self.env = env
        self.max_depth = max_depth
        self.mcts_budget = mcts_budget
        self.use_repack = use_repack
        self.require_full_pack = require_full_pack
        
        # Initialize sub-components
        self.tree_expander = TreeExpander(env, max_depth, reward_weights)
        self.repack_trial = RepackTrial(
            container_dims=(env.L, env.W, env.H),
            time_limit=5.0,
            env=env,
        )

    def search(self, env_state):
        """
        Algorithm 1: High-Level Search

        Main entry point untuk hierarchical placement search.

        Args:
            env_state (dict): Current environment state containing:
                - 'items': List of all items
                - 'current_index': Current item index
                - 'height_map': Current height map
                - 'placed_items': Already placed items
                - 'placed_positions': Positions of placed items

        Returns:
            dict: {
                'success': bool,
                'action': tuple or list,
                'utilization': float,
                'search_depth': int,
                'sequences_found': int
            }
        """
        # Line 1: Initialize root from environment state
        vroot_state = copy.deepcopy(env_state)

        # Line 2: Tree Expansion
        sequences, solved = self.tree_expander.tree_expansion(
            vroot_state,
            placements_tried=set(),
            sequence=None,
            depth=0,
            node_count=0,
            require_full_pack=self.require_full_pack
        )

        # Current utilization
        current_util = self._compute_utilization(env_state)

        # Line 3-5: Check if expansion found anything
        if len(sequences) == 0:
            # No valid placement found initially
            if self.use_repack and len(env_state.get('placed_items', [])) > 0:
                # Try repacking
                repack_result = self.repack_trial.attempt_repack(
                    env_state,
                    require_full_pack=self.require_full_pack
                )
                
                if repack_result['success']:
                    return {
                        'success': True,
                        'action': repack_result['actions'],
                        'utilization': repack_result['best_util'],
                        'search_depth': 0,
                        'sequences_found': 0,
                        'action_type': 'repack'
                    }
            
            # Terminate
            return {
                'success': False,
                'action': ('terminate',),
                'utilization': current_util,
                'search_depth': 0,
                'sequences_found': 0,
                'action_type': 'terminate'
            }

        # Line 7: Simulation and Generation
        simulations = self._simulate_sequences(sequences, env_state)

        # Line 8: Best Action Selection
        best_sequence, best_action = self._select_best_action(
            sequences, simulations, env_state
        )

        best_util = simulations.get(id(best_sequence), {}).get('utilization', current_util)

        # Line 10-13: Repacking trial jika diperlukan
        if self.use_repack:
            has_no_position = ('no_position' in str(best_action) or 
                              best_action is None)
            
            if has_no_position or len(sequences) == 0:
                repack_result = self.repack_trial.attempt_repack(
                    env_state,
                    require_full_pack=self.require_full_pack
                )
                
                if repack_result['success'] and repack_result['best_util'] > best_util:
                    return {
                        'success': True,
                        'action': repack_result['actions'],
                        'utilization': repack_result['best_util'],
                        'search_depth': self.max_depth,
                        'sequences_found': len(sequences),
                        'action_type': 'repack'
                    }

        return {
            'success': True,
            'action': best_action,
            'utilization': best_util,
            'search_depth': self.max_depth,
            'sequences_found': len(sequences),
            'action_type': 'placement'
        }

    def _simulate_sequences(self, sequences, env_state):
        """
        Algorithm 1: Simulation and Generation

        Simulate each sequence dan evaluate reward.

        Args:
            sequences (list): Candidate sequences dari tree expansion
            env_state (dict): Current environment state

        Returns:
            dict: Simulation results for each sequence
        """
        simulations = {}

        for sequence in sequences:
            # Simulate placement sequence
            sim_result = self._simulate_single_sequence(sequence, env_state)
            simulations[id(sequence)] = sim_result

        return simulations

    def _simulate_single_sequence(self, sequence, env_state):
        """
        Simulate single replacement sequence.

        Args:
            sequence (list): Sequence of placements
            env_state (dict): Current environment state

        Returns:
            dict: Simulation result
        """
        # Clone environment untuk simulation
        sim_state = copy.deepcopy(env_state)
        
        # Apply sequence actions
        for placement in sequence:
            # placement format: (item_idx, phi, action, depth)
            if len(placement) >= 3:
                item_idx, phi, action = placement[0], placement[1], placement[2]
                
                # Apply placement ke simulation state
                if action is not None:
                    # Update height map
                    sim_state = self._apply_placement(sim_state, item_idx, phi, action)

        # Compute metrics
        util = self._compute_utilization(sim_state)
        reward = self._compute_sequence_reward(sequence, util)

        return {
            'utilization': util,
            'reward': reward,
            'sequence_length': len(sequence),
            'final_state': sim_state
        }

    def _select_best_action(self, sequences, simulations, env_state):
        """
        Algorithm 1: Best Action Selection

        Select best action berdasarkan simulation rewards.

        Args:
            sequences (list): Candidate sequences
            simulations (dict): Simulation results
            env_state (dict): Current environment state

        Returns:
            tuple: (best_sequence, best_action)
        """
        best_sequence = None
        best_reward = -np.inf
        best_action = None

        for sequence in sequences:
            sim_key = id(sequence)
            if sim_key not in simulations:
                continue

            sim = simulations[sim_key]
            reward = sim.get('reward', 0.0)

            if reward > best_reward:
                best_reward = reward
                best_sequence = sequence
                
                # Extract first action dari sequence
                if len(sequence) > 0:
                    placement = sequence[0]
                    if len(placement) >= 3:
                        item_idx, phi, action = placement[0], placement[1], placement[2]
                        best_action = (item_idx, phi, action)

        if best_action is None:
            best_action = ('no_position',)

        return best_sequence, best_action

    def _apply_placement(self, state, item_idx, phi, action):
        """
        Apply single placement ke state.

        Args:
            state (dict): Current state
            item_idx (int): Item index
            phi (int): Orientation
            action (tuple): Position action

        Returns:
            dict: Updated state
        """
        new_state = copy.deepcopy(state)
        
        if action is None:
            return new_state

        x, y, z = action
        items = new_state.get('items', [])
        
        if item_idx >= len(items):
            return new_state

        l, w, h = items[item_idx]
        
        # Apply rotation
        if phi == 1:
            l, w = w, l

        # Update height map
        height_map = new_state.get('height_map')
        if height_map is not None:
            if hasattr(height_map, 'update_region'):
                height_map.update_region(x, y, l, w, z + h)
            else:
                height_map[x:x+l, y:y+w] = z + h

        # Track placement for utilization
        placed_items = new_state.get('placed_items')
        placed_positions = new_state.get('placed_positions')

        if placed_items is None:
            placed_items = []
            new_state['placed_items'] = placed_items
        if placed_positions is None:
            placed_positions = []
            new_state['placed_positions'] = placed_positions

        placed_items.append((l, w, h))
        placed_positions.append((x, y, z))

        return new_state

    def _compute_sequence_reward(self, sequence, utilization):
        """
        Compute reward untuk complete sequence.

        Args:
            sequence (list): Placement sequence
            utilization (float): Final utilization

        Returns:
            float: Reward
        """
        # Primary reward: utilization
        reward = utilization * 100.0
        
        # Secondary reward: sequence efficiency (fewer placements is better)
        reward -= len(sequence) * 0.5
        
        return reward

    def _compute_utilization(self, state):
        """
        Compute current utilization dari state.

        Args:
            state (dict): Environment state

        Returns:
            float: Utilization ratio [0, 1]
        """
        placed_items = state.get('placed_items', [])
        if len(placed_items) == 0:
            return 0.0

        total_volume = sum(item[0] * item[1] * item[2] for item in placed_items)
        container_volume = self.env.L * self.env.W * self.env.H
        
        return total_volume / container_volume
