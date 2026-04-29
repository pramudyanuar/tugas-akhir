import numpy as np
import sys
import os

# Use relative imports for clean module structure
from .height_map import HeightMap
from .lbcp import is_stable, validate_structural_stability


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
        mask = np.ones((self.L, self.W), dtype=bool)
        
        # Mask jika item akan keluar dari boundary
        for x in range(self.L):
            for y in range(self.W):
                if x + item_length > self.L or y + item_width > self.W:
                    mask[x, y] = False
        
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
        mask = np.ones((self.L, self.W), dtype=bool)
        
        # Mask posisi dimana base_height (max di footprint) + item_height > max_height
        for x in range(self.L):
            for y in range(self.W):
                if x + item_length > self.L or y + item_width > self.W:
                    mask[x, y] = False
                    continue
                base_height = height_map.max_height_in_region(x, y, item_length, item_width)
                if base_height + item_height > self.H:
                    mask[x, y] = False
        
        return mask
    
    def mask_unstable_lbcp(self, item_length, item_width, item_height, height_map,
                           feasibility_map=None, use_structural_validation=False,
                           cog_tolerance=0.15):
        """
        Mask unstable (LBCP): posisi yang akan unstable berdasarkan LBCP validation.
        
        Menggunakan LBCP untuk check apakah item akan stabil di posisi tersebut.
        
        Args:
            item_length (int): Panjang item
            item_width (int): Lebar item
            item_height (int): Tinggi item
            height_map (HeightMap): Current height map
            
        Returns:
            numpy array: Boolean mask (True = stable, False = unstable)
        """
        mask = np.ones((self.L, self.W), dtype=bool)
        
        # Check stabilitas di setiap posisi
        for x in range(self.L):
            for y in range(self.W):
                # Cek boundary dulu
                if x + item_length > self.L or y + item_width > self.W:
                    mask[x, y] = False
                    continue
                
                # Cek LBCP stability
                try:
                    if use_structural_validation and feasibility_map is not None:
                        obj_payload = {'x': x, 'y': y, 'w': item_length, 'd': item_width}
                        valid, _, _ = validate_structural_stability(
                            obj_payload,
                            None,
                            height_map.map,
                            feasibility_map,
                            cog_tolerance,
                        )
                        if not valid:
                            mask[x, y] = False
                            continue

                    is_item_stable = is_stable(
                        height_map.map,
                        x, y, item_length, item_width, item_height,
                        self.H
                    )
                    if not is_item_stable:
                        mask[x, y] = False
                except Exception:
                    # Jika ada error di LBCP check, mask posisi ini
                    mask[x, y] = False
        
        return mask
    
    def combine_masks(self, item_length, item_width, item_height,
                      height_map, has_valid_position=True,
                      feasibility_map=None, use_structural_validation=False,
                      cog_tolerance=0.15):
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
        mask_unstable = self.mask_unstable_lbcp(
            item_length,
            item_width,
            item_height,
            height_map,
            feasibility_map=feasibility_map,
            use_structural_validation=use_structural_validation,
            cog_tolerance=cog_tolerance,
        )
        
        # Combined mask: intersection dari semua mask (AND logic)
        combined_mask = mask_bound & mask_overflow & mask_unstable
        
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
            'mask_unstable': mask_unstable
        }
    
    def get_action_vector(self, item_length, item_width, item_height,
                          height_map, include_skip=True,
                          feasibility_map=None, use_structural_validation=False,
                          cog_tolerance=0.15):
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
            feasibility_map=feasibility_map,
            use_structural_validation=use_structural_validation,
            cog_tolerance=cog_tolerance,
        )
        
        combined_mask = masking_result['combined_mask']
        can_skip = masking_result['can_skip']
        
        # Flatten mask
        action_mask = combined_mask.flatten()
        
        # Append skip action
        if include_skip:
            skip_mask = np.array([can_skip], dtype=bool)
            action_mask = np.concatenate([action_mask, skip_mask])
        
        return action_mask.astype(np.float32)
