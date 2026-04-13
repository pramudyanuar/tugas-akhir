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

    def __init__(self, container_dims=(59, 23, 23), time_limit=5.0):
        """
        Initialize RepackTrial.

        Args:
            container_dims (tuple): (L, W, H) container dimensions
            time_limit (float): Time limit untuk repack attempts (seconds)
        """
        self.L, self.W, self.H = container_dims
        self.container_volume = self.L * self.W * self.H
        self.time_limit = time_limit

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
        best_util = self._compute_utilization(env_state)
        success = False

        placed_items = env_state.get('placed_items', [])
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
            for i in range(1, min(num_items + 1, 5)):  # Limit to 4 items max for speed
                # Time check
                if (time.time() - start_time) >= self.time_limit:
                    break

                # Line 4: for each subset U with i elements (last-packed-first)
                subsets = self._generate_subsets_last_packed_first(
                    placed_items, i, num_items
                )

                for subset_indices in subsets:
                    # Time check
                    if (time.time() - start_time) >= self.time_limit:
                        break

                    # Line 5: Clone environment state
                    cloned_state = copy.deepcopy(env_state)

                    # Line 6-9: Execute unpack for each item in subset
                    unpack_actions = []
                    for item_idx in subset_indices:
                        unpack_actions.append(('unpack', item_idx))

                    # Line 10: Tree expansion (simplified greedy placement)
                    repack_result = self._tree_expansion_greedy(
                        cloned_state, subset_indices
                    )

                    if repack_result['success']:
                        # Line 18-21: Check if better utilization or full pack achieved
                        new_util = repack_result.get('utilization', best_util)

                        if require_full_pack and repack_result.get('full_pack', False):
                            # Line 16-17: Full pack achieved
                            success = True
                            best_actions = unpack_actions + repack_result.get('actions', [])
                            best_util = new_util
                            elapsed = time.time() - start_time
                            return {
                                'success': True,
                                'actions': best_actions,
                                'best_util': best_util,
                                'items_unpacked': len(subset_indices),
                                'time_elapsed': elapsed
                            }
                        elif not require_full_pack and new_util > best_util:
                            # Line 20: Better utilization found
                            success = True
                            best_actions = unpack_actions + repack_result.get('actions', [])
                            best_util = new_util

        elapsed = time.time() - start_time
        return {
            'success': success,
            'actions': best_actions,
            'best_util': best_util,
            'items_unpacked': len(best_actions),
            'time_elapsed': elapsed
        }

    def _generate_subsets_last_packed_first(self, items, subset_size, total_items):
        """
        Generate subsets dengan last-packed-first strategy.

        Items yang di-unpack adalah yang paling akhir di-pack (highest indices).

        Args:
            items (list): All placed items
            subset_size (int): Size of subset
            total_items (int): Total number of items

        Returns:
            list: List of subset index combinations (last-packed-first)
        """
        subsets = []

        # Generate combinations starting dari last items (highest indices)
        # For simplicity, use greedy: last k items
        if subset_size <= total_items:
            # Last subset_size items
            last_indices = list(range(total_items - subset_size, total_items))
            subsets.append(last_indices)

        return subsets

    def _tree_expansion_greedy(self, cloned_state, unpacked_indices):
        """
        Simplified tree expansion menggunakan greedy placement.

        Args:
            cloned_state (dict): Cloned environment state
            unpacked_indices (list): Indices dari items yang di-unpack

        Returns:
            dict: {
                'success': bool,
                'actions': list of placement actions,
                'utilization': float,
                'full_pack': bool
            }
        """
        # Create new height map dan place remaining items
        from src.core.height_map import HeightMap
        from src.core.lbcp import is_stable

        height_map = HeightMap(self.L, self.W, self.H)
        
        placed_items = cloned_state.get('placed_items', [])
        placed_positions = cloned_state.get('placed_positions', [])

        # Re-place all items except unpacked ones
        new_positions = []
        successful_placements = 0
        
        for idx, (item, old_pos) in enumerate(zip(placed_items, placed_positions)):
            if idx in unpacked_indices:
                # Skip unpacked items
                continue

            l, w, h = item
            
            # Find best position untuk item ini
            placed = False
            for z in range(self.H - h + 1):
                if placed:
                    break
                for y in range(self.W - w + 1):
                    if placed:
                        break
                    for x in range(self.L - l + 1):
                        # Check stability
                        region_max = np.max(height_map.get_region(x, y, l, w))
                        if region_max + h <= self.H:
                            # Place item
                            height_map.update_region(x, y, l, w, region_max + h)
                            new_positions.append((x, y, region_max))
                            successful_placements += 1
                            placed = True
                            break

        # Try to place unpacked items
        unpacked_items = [placed_items[i] for i in unpacked_indices]
        unpacked_placed = 0

        for item in unpacked_items:
            l, w, h = item
            placed = False

            for z in range(self.H - h + 1):
                if placed:
                    break
                for y in range(self.W - w + 1):
                    if placed:
                        break
                    for x in range(self.L - l + 1):
                        region_max = np.max(height_map.get_region(x, y, l, w))
                        if region_max + h <= self.H:
                            height_map.update_region(x, y, l, w, region_max + h)
                            new_positions.append((x, y, region_max))
                            unpacked_placed += 1
                            placed = True
                            break

        # Compute new utilization
        total_volume = sum(item[0] * item[1] * item[2] for item in placed_items)
        new_util = total_volume / self.container_volume

        success = (successful_placements + unpacked_placed) == len(placed_items)
        full_pack = unpacked_placed == len(unpacked_items)

        return {
            'success': success,
            'actions': [],  # Could be populated dengan placement actions
            'utilization': new_util,
            'full_pack': full_pack,
            'positions': new_positions
        }

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

        total_volume = sum(item[0] * item[1] * item[2] for item in placed_items)
        return total_volume / self.container_volume
