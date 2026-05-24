import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import sys
import os

from src.utils.item_utils import get_item_stacking

# Use relative imports for clean module structure
from .height_map import HeightMap
from .lbcp import is_stable, validate_structural_stability

# Parallel processing (optional, fallback to sequential if unavailable)
try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False


class ActionMask:
    """
    Action masking untuk 3D bin packing problem.
    
    Masks:
    1. Mask out-of-bound: posisi yang keluar dari container
    2. Mask overflow: posisi yang akan melebihi max height
    3. Mask unstable (LBCP): posisi yang akan unstable
    4. Mask no-position: jangan skip jika masih ada valid position
    """
    
    def __init__(self, container_length=59, container_width=23, 
                 container_height=23, skip_action_idx=-1):
        """
        Initialize action masking.
        
        Args:
            container_length (int): Panjang container (L)
            container_width (int): Lebar container (W)
            container_height (int): Tinggi container (H)
            skip_action_idx (int): Index untuk skip/no-op action (-1 jika di akhir)
        """
        self.L = container_length
        self.W = container_width
        self.H = container_height
        self.skip_action_idx = skip_action_idx
        self._bound_cache = {}
    
    @staticmethod
    def _validate_single_position(x, y, item_length, item_width, item_height, 
                                  height_map, max_height,
                                  use_structural_validation, feasibility_map, 
                                  cog_tolerance):
        """
        Validate a single position for stability (used in parallel processing).
        
        Args:
            x, y: Position coordinates
            item_length, item_width, item_height: Item dimensions
            height_map: Height map array
            max_height: Container max height
            use_structural_validation: Whether to use LBCP validation
            feasibility_map: Feasibility map for LBCP
            cog_tolerance: CoG tolerance for LBCP
            
        Returns:
            tuple: (x, y, is_valid) - position and validation result
        """
        try:
            if use_structural_validation and feasibility_map is not None:
                obj_payload = {'x': int(x), 'y': int(y), 'w': item_length, 'd': item_width}
                valid, _, _ = validate_structural_stability(
                    obj_payload,
                    None,
                    height_map,
                    feasibility_map,
                    cog_tolerance,
                )
                return (x, y, valid)
            else:
                is_item_stable = is_stable(
                    height_map,
                    int(x), int(y), item_length, item_width, item_height,
                    max_height,
                    strict_mode=False,
                )
                return (x, y, is_item_stable)
        except Exception:
            return (x, y, False)
        
    def mask_out_of_bound(self, item_length, item_width, height_map):
        """
        Mask out-of-bound: posisi yang akan membuat item keluar dari container.
        
        Args:
            item_length (int): Panjang item
            item_width (int): Lebar item
            height_map (HeightMap): Current height map
            
        Returns:
            numpy array: Boolean mask (True = valid, False = invalid/masked)
        """
        cache_key = (int(item_length), int(item_width))
        cached = self._bound_cache.get(cache_key)
        if cached is not None:
            return cached

        mask = np.zeros((self.L, self.W), dtype=bool)
        max_x = self.L - item_length + 1
        max_y = self.W - item_width + 1
        if max_x > 0 and max_y > 0:
            mask[:max_x, :max_y] = True

        self._bound_cache[cache_key] = mask
        return mask
    
    def mask_overflow(self, item_length, item_width, item_height, height_map):
        """
        Mask overflow: posisi yang akan membuat height melebihi container height.
        
        Args:
            item_length (int): Panjang item
            item_width (int): Lebar item
            item_height (int): Tinggi item
            height_map (HeightMap): Current height map
            
        Returns:
            numpy array: Boolean mask (True = valid, False = overflow)
        """
        mask = np.zeros((self.L, self.W), dtype=bool)

        max_x = self.L - item_length + 1
        max_y = self.W - item_width + 1
        if max_x <= 0 or max_y <= 0:
            return mask

        hm = height_map.map if hasattr(height_map, 'map') else height_map
        windows = sliding_window_view(hm, (item_length, item_width))
        base_heights = windows.max(axis=(-2, -1))
        valid = (base_heights + item_height) <= self.H
        mask[:max_x, :max_y] = valid

        return mask
    
    def mask_unstable_lbcp(self, item_length, item_width, item_height, height_map,
                           feasibility_map=None, use_structural_validation=False,
                           cog_tolerance=0.15, fast_stability_mask=False,
                           candidate_mask=None):
        """
        Mask unstable (LBCP): posisi yang akan unstable berdasarkan LBCP validation.
        
        Menggunakan LBCP untuk check apakah item akan stabil di posisi tersebut.
        Parallel processing digunakan ketika ada banyak kandidat.
        
        Args:
            item_length (int): Panjang item
            item_width (int): Lebar item
            item_height (int): Tinggi item
            height_map (HeightMap): Current height map
            
        Returns:
            numpy array: Boolean mask (True = stable, False = unstable)
        """
        mask = np.zeros((self.L, self.W), dtype=bool)

        max_x = self.L - item_length + 1
        max_y = self.W - item_width + 1
        if max_x <= 0 or max_y <= 0:
            return mask

        if candidate_mask is None:
            candidate_mask = np.zeros((self.L, self.W), dtype=bool)
            candidate_mask[:max_x, :max_y] = True

        hm = height_map.map if hasattr(height_map, 'map') else height_map

        # Fast path: use feasibility_map with sliding window (very fast, approximate)
        if use_structural_validation and feasibility_map is not None and fast_stability_mask:
            windows = sliding_window_view(feasibility_map, (item_length, item_width))
            valid_fm = windows.any(axis=(-2, -1))
            mask[:max_x, :max_y] = valid_fm
            mask &= candidate_mask
            return mask

        # Slow path: run full LBCP validation per position
        valid_positions = np.argwhere(candidate_mask[:max_x, :max_y])
        
        # Use parallel processing if joblib is available and we have many candidates
        if HAS_JOBLIB and len(valid_positions) > 32:
            # Parallel batch validation
            results = Parallel(n_jobs=-1, backend='threading')(
                delayed(self._validate_single_position)(
                    int(x), int(y), item_length, item_width, item_height,
                    hm, self.H, use_structural_validation, feasibility_map, cog_tolerance
                )
                for x, y in valid_positions
            )
            for x, y, is_valid in results:
                if is_valid:
                    mask[x, y] = True
        else:
            # Sequential fallback for few candidates or without joblib
            for x, y in valid_positions:
                _, _, is_valid = self._validate_single_position(
                    x, y, item_length, item_width, item_height,
                    hm, self.H, use_structural_validation, feasibility_map, cog_tolerance
                )
                if is_valid:
                    mask[x, y] = True

        return mask
    
    def combine_masks(self, item_length, item_width, item_height,
                      height_map, has_valid_position=True,
                      top_item_map=None, placed_items=None, item_stacking=None,
                      feasibility_map=None, use_structural_validation=False,
                      cog_tolerance=0.15, fast_stability_mask=False):
        """
        Combine semua masks dengan logic:
        1. out-of-bound AND
        2. overflow AND
        3. unstable AND
        4. (no-op/skip hanya valid jika tidak ada valid position yang tersisa)
        
        Args:
            item_length (int): Panjang item
            item_width (int): Lebar item
            item_height (int): Tinggi item
            height_map (HeightMap): Current height map
            has_valid_position (bool): Apakah sudah ada position yang valid ditemukan
            
        Returns:
            dict: Dictionary berisi:
                - 'combined_mask': boolean array mask (L, W)
                - 'valid_positions': list of (x, y) valid positions
                - 'can_skip': bool, apakah boleh skip action
                - 'num_valid': int, jumlah valid position
        """
        # Combine semua masks
        mask_bound = self.mask_out_of_bound(item_length, item_width, height_map)
        mask_overflow = self.mask_overflow(item_length, item_width, item_height, height_map)
        candidate_mask = mask_bound & mask_overflow
        mask_unstable = self.mask_unstable_lbcp(
            item_length,
            item_width,
            item_height,
            height_map,
            feasibility_map=feasibility_map,
            use_structural_validation=use_structural_validation,
            cog_tolerance=cog_tolerance,
            fast_stability_mask=fast_stability_mask,
            candidate_mask=candidate_mask,
        )
        mask_stacking = self.mask_stacking_policy(
            item_length,
            item_width,
            height_map,
            top_item_map=top_item_map,
            placed_items=placed_items,
            item_stacking=item_stacking,
        )
        
        # Combined mask: intersection dari semua mask (AND logic)
        combined_mask = mask_bound & mask_overflow & mask_unstable & mask_stacking
        
        # Get valid positions
        valid_positions = np.where(combined_mask)
        valid_positions = list(zip(valid_positions[0], valid_positions[1]))
        num_valid = len(valid_positions)
        
        # Logic: skip/no-op hanya valid jika tidak ada valid position
        can_skip = (num_valid == 0)
        
        return {
            'combined_mask': combined_mask,
            'valid_positions': valid_positions,
            'can_skip': can_skip,
            'num_valid': num_valid,
            'mask_bound': mask_bound,
            'mask_overflow': mask_overflow,
            'mask_unstable': mask_unstable,
            'mask_stacking': mask_stacking
        }
    
    def get_action_vector(self, item_length, item_width, item_height,
                          height_map, include_skip=True,
                          top_item_map=None, placed_items=None, item_stacking=None,
                          feasibility_map=None, use_structural_validation=False,
                          cog_tolerance=0.15, fast_stability_mask=False):
        """
        Get action vector untuk disimpalin ke neural network.
        
        Format: mask di-flatten dari (L, W) -> (L*W,) dan di-append bit skip
        
        Args:
            item_length: Panjang item
            item_width: Lebar item
            item_height: Tinggi item
            height_map: Current height map
            include_skip: Include skip action dalam mask
            
        Returns:
            numpy array: Action mask vector (L*W + 1,) jika include_skip
        """
        masking_result = self.combine_masks(
            item_length,
            item_width,
            item_height,
            height_map,
            top_item_map=top_item_map,
            placed_items=placed_items,
            item_stacking=item_stacking,
            feasibility_map=feasibility_map,
            use_structural_validation=use_structural_validation,
            cog_tolerance=cog_tolerance,
            fast_stability_mask=fast_stability_mask,
        )
        
        combined_mask = masking_result['combined_mask']
        can_skip = masking_result['can_skip']
        
        # Flatten mask in column-major order so index = x + y*L
        action_mask = combined_mask.flatten(order='F')
        
        # Append skip action
        if include_skip:
            skip_mask = np.array([can_skip], dtype=bool)
            action_mask = np.concatenate([action_mask, skip_mask])
        
        return action_mask.astype(np.float32)

    def mask_stacking_policy(self, item_length, item_width, height_map,
                             top_item_map=None, placed_items=None, item_stacking=None):
        """Mask positions that violate stacking policy based on top items."""
        mask = np.ones((self.L, self.W), dtype=bool)

        if top_item_map is None or placed_items is None:
            return mask

        if item_stacking is None:
            item_stacking = 'stackable'

        hm = height_map.map if hasattr(height_map, 'map') else height_map

        for x in range(self.L):
            for y in range(self.W):
                if x + item_length > self.L or y + item_width > self.W:
                    mask[x, y] = False
                    continue

                base_height = np.max(hm[x:x + item_length, y:y + item_width])
                if base_height <= 0:
                    continue

                region = hm[x:x + item_length, y:y + item_width]
                support_mask = region == base_height
                if not np.any(support_mask):
                    continue

                support_indices = set(
                    top_item_map[x:x + item_length, y:y + item_width][support_mask].tolist()
                )
                for idx in support_indices:
                    if idx < 0 or idx >= len(placed_items):
                        continue
                    support_item = placed_items[idx]
                    support_stack = get_item_stacking(support_item)
                    if support_stack == 'no_stack':
                        mask[x, y] = False
                        break
                    if support_stack == 'fragile' and item_stacking != 'fragile':
                        mask[x, y] = False
                        break

        return mask
