"""
Algorithm 3: Repack Trial
Repacking mechanism untuk reorganisasi items dengan greedy tree expansion strategy.

Main Features:
- Subset generation (last-packed-first strategy)
- Time-limited search
- Tree expansion untuk repack attempts
- Utilization maximization
"""

import numpy as np
import time
import copy
import itertools
from types import SimpleNamespace

from .tree_expansion import TreeExpander
from src.core.height_map import HeightMap
from src.core.lbcp import validate_structural_stability, update_feasibility_map
from src.utils.item_utils import get_item_dims, get_item_stacking, make_item


class RepackTrial:
    """
    Algorithm 3: Repack Trial Implementation

    Pseudocode:
        Input: vroot, util, requireFullPack
        Output: a*, RepackSuccess

        1. a* ← (), a ← (), RepackSuccess ← false
        2. while time limit not reached do
        3.     for i ← 1 to |I| do
        4.         for each subset U of I with i elements (last-packed-first) do
        5.             Clone B, N, I, C to B̂, N̂, Î, Ĉ
        6.             for each item o in U do
        7.                 Execute unpack(o) and update v̂root
        8.                 a ← APPEND(a, unpack(o))
        9.             end for
        10.            X, solved ← TREEEXPANSION(v̂root, ∅, (), 0, 0, requireFullPack)
        11.            if X ≠ ∅ then
        12.                A ← SIMULATIONANDGENERATION(X)
        13.                χ*, aχ* ← BESTACTIONSELECTION(X, A)
        14.                aχ* ← ACTIONUPDATE(a, aχ*)
        15.                if requireFullPack and NoPositionAction ∉ χ* then
        16.                    RepackSuccess ← true
        17.                    return aχ*, RepackSuccess
        18.                else if not requireFullPack and util < util(χ*) then
        19.                    RepackSuccess ← true
        20.                    a* ← aχ*, util ← util(χ*)
        21.                end if
        22.            end if
        23.            Reset a ← ()
        24.        end for
        25.    end for
        26. end while
        27. return a*, RepackSuccess
    """

    def __init__(self, container_dims=(59, 23, 23), time_limit=5.0,
                 env=None, max_depth=20, reward_weights=None):
        """
        Initialize RepackTrial.

        Args:
            container_dims (tuple): (L, W, H) container dimensions
            time_limit (float): Time limit untuk repack attempts (seconds)
        """
        self.L, self.W, self.H = container_dims
        self.container_volume = self.L * self.W * self.H
        self.time_limit = time_limit

        # Use provided env when available; fallback to a minimal proxy.
        if env is None:
            env = SimpleNamespace(
                L=self.L,
                W=self.W,
                H=self.H,
                action_mask_calculator=None,
                use_structural_validation=False,
                cog_tolerance=0.15,
            )
        self.env = env
        self.tree_expander = TreeExpander(self.env, max_depth=max_depth, reward_weights=reward_weights)

    def attempt_repack(self, env_state, require_full_pack=False):
        """
        Algorithm 3: RepackTrial Attempt

        Main entry point untuk repack attempt dengan time-limited search.

        Args:
            env_state (dict): Current environment state containing:
                - 'placed_items': List of (l, w, h)
                - 'placed_positions': List of (x, y, z)
                - Other state needed untuk unpack/repack
            require_full_pack (bool): Require full packing atau utilization maximize

        Returns:
            dict: {
                'success': bool,
                'actions': list of (action_type, item, position),
                'best_util': float,
                'items_unpacked': int,
                'time_elapsed': float
            }
        """
        start_time = time.time()
        best_actions = []
        best_positions = None
        best_util = self._compute_utilization(env_state)
        success = False

        placed_items = env_state.get('placed_items', [])
        placed_positions = env_state.get('placed_positions', [])
        num_items = len(placed_items)

        if num_items == 0:
            return {
                'success': False,
                'actions': [],
                'best_util': best_util,
                'items_unpacked': 0,
                'time_elapsed': 0.0
            }

        # Line 2: while time limit not reached
        while (time.time() - start_time) < self.time_limit:
            # Line 3: for i <- 1 to |I|
            for i in range(1, num_items + 1):
                if (time.time() - start_time) >= self.time_limit:
                    break

                # Line 4: for each subset U with i elements (last-packed-first)
                for subset_indices in self._iter_subsets_last_packed_first(num_items, i):
                    if (time.time() - start_time) >= self.time_limit:
                        break

                    # Line 5: Clone environment state
                    cloned_state, positions_by_index = self._build_repack_state(
                        env_state,
                        subset_indices,
                    )

                    # Line 6-9: Execute unpack for each item in subset
                    unpack_actions = [('unpack', idx) for idx in subset_indices]

                    # Line 10: Tree expansion
                    sequences, solved = self.tree_expander.tree_expansion(
                        cloned_state,
                        placements_tried=set(),
                        sequence=None,
                        depth=0,
                        node_count=0,
                        require_full_pack=require_full_pack,
                    )

                    if not sequences:
                        continue

                    # Line 12: Simulation and generation
                    simulations = self._simulate_sequences(sequences, cloned_state)

                    # Line 13: Best action selection
                    best_sequence, best_actions = self._select_best_action(sequences, simulations)
                    sim_result = simulations.get(id(best_sequence), {})
                    new_util = sim_result.get('utilization', best_util)
                    final_state = sim_result.get('final_state')
                    has_no_position = any(p[2] is None for p in best_sequence)

                    if final_state is None:
                        continue

                    final_positions = final_state.get('positions_by_index', positions_by_index)

                    # Line 15-21: Full pack or improved utilization
                    if require_full_pack and not has_no_position:
                        success = True
                        best_actions = unpack_actions + best_actions
                        best_util = new_util
                        elapsed = time.time() - start_time
                        return {
                            'success': True,
                            'actions': best_actions,
                            'best_util': best_util,
                            'items_unpacked': len(subset_indices),
                            'time_elapsed': elapsed,
                            'positions': final_positions,
                        }

                    if not require_full_pack and new_util > best_util:
                        success = True
                        best_actions = unpack_actions + best_actions
                        best_util = new_util
                        best_positions = final_positions

        elapsed = time.time() - start_time
        return {
            'success': success,
            'actions': best_actions,
            'best_util': best_util,
            'items_unpacked': len(best_actions),
            'time_elapsed': elapsed,
            'positions': best_positions if success else None,
        }

    def _iter_subsets_last_packed_first(self, total_items, subset_size):
        """
        Generate subsets with last-packed-first priority.

        Yields index lists ordered by most recently packed items first.
        """
        if subset_size <= 0 or subset_size > total_items:
            return

        indices = list(range(total_items - 1, -1, -1))
        for combo in itertools.combinations(indices, subset_size):
            yield list(sorted(combo, reverse=True))

    def _build_repack_state(self, env_state, unpacked_indices):
        """Clone state and rebuild maps after unpacking selected items."""
        cloned_state = copy.deepcopy(env_state)

        placed_items = cloned_state.get('placed_items', [])
        placed_positions = cloned_state.get('placed_positions', [])
        total_items = len(placed_items)

        remaining_items = []
        remaining_positions = []
        positions_by_index = [None] * total_items

        for idx, (item, pos) in enumerate(zip(placed_items, placed_positions)):
            if idx in unpacked_indices:
                continue
            remaining_items.append(item)
            remaining_positions.append(pos)
            positions_by_index[idx] = pos

        # Rebuild height map and feasibility map from remaining items
        height_map = HeightMap(self.L, self.W, self.H)
        feasibility_map = np.ones((self.L, self.W), dtype=bool)

        top_item_map = np.full((self.L, self.W), -1, dtype=np.int32)

        for idx, (item, (x, y, base_height)) in enumerate(zip(remaining_items, remaining_positions)):
            item_l, item_w, item_h = get_item_dims(item)
            support_polygon = None
            if getattr(self.env, 'use_structural_validation', False):
                obj_payload = {'x': x, 'y': y, 'w': item_l, 'd': item_w}
                valid, support_polygon, _ = validate_structural_stability(
                    obj_payload,
                    None,
                    height_map.map,
                    feasibility_map,
                    getattr(self.env, 'cog_tolerance', 0.15),
                )
                if not valid:
                    # Skip feasibility update if validation fails
                    support_polygon = None

            height_map.update_region(x, y, item_l, item_w, base_height + item_h)
            if support_polygon is not None:
                feasibility_map = update_feasibility_map(feasibility_map, support_polygon)
            top_item_map[x:x + item_l, y:y + item_w] = idx

        # Items to place: unpacked items
        repack_items = [placed_items[idx] for idx in unpacked_indices]
        repack_indices = list(unpacked_indices)

        cloned_state['height_map'] = height_map
        cloned_state['feasibility_map'] = feasibility_map
        cloned_state['placed_items'] = remaining_items.copy()
        cloned_state['placed_positions'] = remaining_positions.copy()
        cloned_state['top_item_map'] = top_item_map
        cloned_state['items'] = repack_items
        cloned_state['item_indices'] = repack_indices
        cloned_state['positions_by_index'] = positions_by_index
        cloned_state['current_index'] = 0

        return cloned_state, positions_by_index

    def _simulate_sequences(self, sequences, env_state):
        simulations = {}
        for sequence in sequences:
            sim_result = self._simulate_single_sequence(sequence, env_state)
            simulations[id(sequence)] = sim_result
        return simulations

    def _simulate_single_sequence(self, sequence, env_state):
        sim_state = copy.deepcopy(env_state)

        for placement in sequence:
            if len(placement) >= 3:
                item_idx, phi, action = placement[0], placement[1], placement[2]
                if action is not None:
                    sim_state = self._apply_placement(sim_state, item_idx, phi, action)

        util = self._compute_utilization(sim_state)
        reward = util * 100.0 - len(sequence) * 0.5

        return {
            'utilization': util,
            'reward': reward,
            'sequence_length': len(sequence),
            'final_state': sim_state,
        }

    def _select_best_action(self, sequences, simulations):
        best_sequence = None
        best_reward = -np.inf
        best_action = None
        best_actions = []

        for sequence in sequences:
            sim_key = id(sequence)
            if sim_key not in simulations:
                continue
            sim = simulations[sim_key]
            reward = sim.get('reward', 0.0)
            if reward > best_reward:
                best_reward = reward
                best_sequence = sequence
                if len(sequence) > 0:
                    placement = sequence[0]
                    if len(placement) >= 3:
                        item_idx, phi, action = placement[0], placement[1], placement[2]
                        best_action = (item_idx, phi, action)

        if best_sequence is None:
            return None, []

        if best_action is None:
            best_action = ('no_position',)

        sim = simulations.get(id(best_sequence), {})
        final_state = sim.get('final_state') if sim else None
        item_indices = final_state.get('item_indices') if final_state else None

        for placement in best_sequence:
            if len(placement) < 3:
                continue
            item_idx, phi, action = placement[0], placement[1], placement[2]
            if action is None:
                continue
            original_idx = item_indices[item_idx] if item_indices and item_idx < len(item_indices) else item_idx
            best_actions.append(('place', original_idx, phi, action))

        return best_sequence, best_actions

    def _apply_placement(self, state, item_idx, phi, action):
        new_state = copy.deepcopy(state)

        if action is None:
            return new_state

        x, y, z = action
        items = new_state.get('items', [])
        if item_idx >= len(items):
            return new_state

        item = items[item_idx]
        l, w, h = get_item_dims(item)
        stacking = get_item_stacking(item)
        if phi == 1:
            l, w = w, l

        # Update height map
        height_map = new_state.get('height_map')
        if height_map is not None:
            if hasattr(height_map, 'update_region'):
                height_map.update_region(x, y, l, w, z + h)
            else:
                height_map[x:x+l, y:y+w] = z + h

        # Update feasibility map if enabled
        if (
            getattr(self.env, 'use_structural_validation', False)
            and new_state.get('feasibility_map') is not None
            and height_map is not None
        ):
            try:
                obj_payload = {'x': x, 'y': y, 'w': l, 'd': w}
                valid, support_polygon, _ = validate_structural_stability(
                    obj_payload,
                    None,
                    height_map.map if hasattr(height_map, 'map') else height_map,
                    new_state.get('feasibility_map'),
                    getattr(self.env, 'cog_tolerance', 0.15),
                )
                if valid and support_polygon is not None and len(support_polygon) >= 3:
                    new_state['feasibility_map'] = update_feasibility_map(
                        new_state.get('feasibility_map'),
                        support_polygon,
                    )
            except Exception:
                pass

        # Track placed items/positions
        placed_items = new_state.get('placed_items')
        placed_positions = new_state.get('placed_positions')
        if placed_items is None:
            placed_items = []
            new_state['placed_items'] = placed_items
        if placed_positions is None:
            placed_positions = []
            new_state['placed_positions'] = placed_positions

        placed_items.append(make_item(l, w, h, stacking))
        placed_positions.append((x, y, z))

        top_item_map = new_state.get('top_item_map')
        if top_item_map is None:
            top_item_map = np.full((self.L, self.W), -1, dtype=np.int32)
            new_state['top_item_map'] = top_item_map
        placed_index = len(placed_items) - 1
        top_item_map[x:x + l, y:y + w] = placed_index

        # Update positions_by_index mapping
        item_indices = new_state.get('item_indices', [])
        positions_by_index = new_state.get('positions_by_index')
        if positions_by_index is not None and item_idx < len(item_indices):
            original_idx = item_indices[item_idx]
            positions_by_index[original_idx] = (x, y, z)

        return new_state

    def attempt_repack_bottom_left_fill(self, items, positions):
        """Legacy helper: simple bottom-left-fill repacking."""
        success, new_positions, max_height = self._bottom_left_fill(items)
        return success, new_positions, max_height

    def attempt_repack_load_balanced(self, items, positions):
        """Legacy helper: load-balanced repacking (uses BLF baseline)."""
        success, new_positions, max_height = self._bottom_left_fill(items)
        utilization = self._compute_utilization_from_items(items)
        metric = utilization if success else 0.0
        return success, new_positions, metric

    def attempt_repack_minimize_height(self, items, positions):
        """Legacy helper: minimize max height (uses BLF baseline)."""
        success, new_positions, max_height = self._bottom_left_fill(items)
        metric = max_height if success else float('inf')
        return success, new_positions, metric

    def auto_repack(self, items, positions, strategy='auto'):
        """Legacy helper: choose repack strategy and return dict result."""
        if strategy == 'load_balanced':
            success, new_positions, metric = self.attempt_repack_load_balanced(items, positions)
            return {
                'success': success,
                'positions': new_positions,
                'metric': metric,
                'strategy': 'load_balanced',
            }
        if strategy == 'min_height':
            success, new_positions, metric = self.attempt_repack_minimize_height(items, positions)
            return {
                'success': success,
                'positions': new_positions,
                'metric': metric,
                'strategy': 'min_height',
            }
        if strategy == 'blf':
            success, new_positions, metric = self.attempt_repack_bottom_left_fill(items, positions)
            return {
                'success': success,
                'positions': new_positions,
                'metric': metric,
                'strategy': 'blf',
            }

        # Auto: pick best by lowest height, fallback to utilization
        lb_success, lb_positions, lb_metric = self.attempt_repack_load_balanced(items, positions)
        mh_success, mh_positions, mh_metric = self.attempt_repack_minimize_height(items, positions)

        if mh_success:
            return {
                'success': True,
                'positions': mh_positions,
                'metric': mh_metric,
                'strategy': 'min_height',
            }
        if lb_success:
            return {
                'success': True,
                'positions': lb_positions,
                'metric': lb_metric,
                'strategy': 'load_balanced',
            }

        return {
            'success': False,
            'positions': [],
            'metric': 0.0,
            'strategy': 'auto',
        }

    def _bottom_left_fill(self, items):
        """Place items in bottom-left order using first-fit scanning."""
        if not items:
            return True, [], 0.0

        height_map = HeightMap(self.L, self.W, self.H)
        new_positions = []

        for item in items:
            item_l, item_w, item_h = get_item_dims(item)
            placed = False
            for x in range(self.L - item_l + 1):
                if placed:
                    break
                for y in range(self.W - item_w + 1):
                    region_max = np.max(height_map.get_region(x, y, item_l, item_w))
                    if region_max + item_h <= self.H:
                        height_map.update_region(x, y, item_l, item_w, region_max + item_h)
                        new_positions.append((x, y, int(region_max)))
                        placed = True
                        break

            if not placed:
                return False, new_positions, float('inf')

        max_height = float(np.max(height_map.map))
        return True, new_positions, max_height

    def _compute_utilization_from_items(self, items):
        if not items:
            return 0.0
        total_volume = sum(
            get_item_dims(item)[0] * get_item_dims(item)[1] * get_item_dims(item)[2]
            for item in items
        )
        return total_volume / self.container_volume

    def _compute_utilization(self, env_state):
        """
        Compute current utilization dari environment state.

        Args:
            env_state (dict): Environment state

        Returns:
            float: Utilization ratio [0, 1]
        """
        placed_items = env_state.get('placed_items', [])
        if len(placed_items) == 0:
            return 0.0

        total_volume = sum(
            get_item_dims(item)[0] * get_item_dims(item)[1] * get_item_dims(item)[2]
            for item in placed_items
        )
        return total_volume / self.container_volume
