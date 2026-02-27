import numpy as np
import sys
import os

# Add parent directory to path untuk import
sys.path.insert(0, os.path.dirname(__file__))

from height_map import HeightMap
from lbcp import is_stable


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
    
    def mask_overflow(self, item_height, height_map):
        """
        Mask overflow: posisi yang akan membuat height melebihi container height.
        
        Args:
            item_height (int): Tinggi item
            height_map (HeightMap): Current height map
            
        Returns:
            numpy array: Boolean mask (True = valid, False = overflow)
        """
        mask = np.ones((self.L, self.W), dtype=bool)
        
        # Mask posisi dimana base_height + item_height > max_height
        for x in range(self.L):
            for y in range(self.W):
                base_height = height_map.map[x, y]
                if base_height + item_height > self.H:
                    mask[x, y] = False
        
        return mask
    
    def mask_unstable_lbcp(self, item_length, item_width, item_height, height_map):
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
                     height_map, has_valid_position=True):
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
        mask_overflow = self.mask_overflow(item_height, height_map)
        mask_unstable = self.mask_unstable_lbcp(item_length, item_width, 
                                               item_height, height_map)
        
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
                         height_map, include_skip=True):
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
        masking_result = self.combine_masks(item_length, item_width, item_height,
                                          height_map)
        
        combined_mask = masking_result['combined_mask']
        can_skip = masking_result['can_skip']
        
        # Flatten mask
        action_mask = combined_mask.flatten()
        
        # Append skip action
        if include_skip:
            skip_mask = np.array([can_skip], dtype=bool)
            action_mask = np.concatenate([action_mask, skip_mask])
        
        return action_mask.astype(np.float32)


if __name__ == "__main__":
    """Test cases untuk Action Masking"""
    
    print("=" * 70)
    print("Test Case 1: Mask out-of-bound")
    print("=" * 70)
    
    height_map = HeightMap(59, 23, 23)
    height_map.reset()
    
    action_mask = ActionMask(59, 23, 23)
    
    # Test item yang besar (15x12)
    item_l, item_w = 15, 12
    mask_bound = action_mask.mask_out_of_bound(item_l, item_w, height_map)
    
    # Valid positions: harus x+15 <= 59 dan y+12 <= 23
    # Valid x: 0-43, Valid y: 0-10
    valid_x = np.sum(np.sum(mask_bound, axis=1) > 0)
    valid_y = np.sum(np.sum(mask_bound, axis=0) > 0)
    
    print(f"Item size: {item_l}x{item_w}")
    print(f"Valid x range: 0-{59-item_l} (max {59-item_l+1} positions)")
    print(f"Valid y range: 0-{23-item_w} (max {23-item_w+1} positions)")
    print(f"Total valid positions: {np.sum(mask_bound)}")
    print(f"Total positions: {59*23}")
    assert np.sum(mask_bound) == (59-item_l+1) * (23-item_w+1), "Out-of-bound mask wrong!"
    print("✓ PASSED\n")
    
    print("=" * 70)
    print("Test Case 2: Mask overflow")
    print("=" * 70)
    
    height_map.reset()
    # Place some items first
    height_map.update_region(0, 0, 10, 10, 20)  # Heights = 20
    height_map.update_region(20, 0, 10, 10, 15)  # Heights = 15
    
    item_h = 5
    mask_overflow = action_mask.mask_overflow(item_h, height_map)
    
    # Position (0, 0) has base height 20, 20+5 > 23, should be masked
    # Position (20, 0) has base height 15, 15+5 <= 23, should be valid
    
    print(f"Item height: {item_h}")
    print(f"Max container height: 23")
    print(f"(0,0) base_height=20, 20+5=25 > 23: masked={not mask_overflow[0, 0]}")
    print(f"(20,0) base_height=15, 15+5=20 <= 23: valid={mask_overflow[20, 0]}")
    assert not mask_overflow[0, 0], "Overflow at (0,0) should be masked!"
    assert mask_overflow[20, 0], "No overflow at (20,0)!"
    print("✓ PASSED\n")
    
    print("=" * 70)
    print("Test Case 3: Mask unstable (LBCP)")
    print("=" * 70)
    
    height_map.reset()
    # Create unstable scenario: flat floor at z=0
    # Item 5x5x5 on flat floor should be stable
    item_l, item_w, item_h = 5, 5, 5
    
    mask_unstable = action_mask.mask_unstable_lbcp(item_l, item_w, item_h, height_map)
    
    # Flat floor (all 0) should be stable
    stable_count = np.sum(mask_unstable)
    total_count = np.sum(action_mask.mask_out_of_bound(item_l, item_w, height_map))
    
    print(f"Item: {item_l}x{item_w}x{item_h} on flat floor")
    print(f"Stable positions: {stable_count}/{total_count}")
    print(f"All should be stable on flat floor")
    assert stable_count == total_count, "Flat floor should be all stable!"
    print("✓ PASSED\n")
    
    print("=" * 70)
    print("Test Case 4: Combine masks")
    print("=" * 70)
    
    height_map.reset()
    item_l, item_w, item_h = 10, 10, 5
    
    result = action_mask.combine_masks(item_l, item_w, item_h, height_map)
    
    print(f"Item: {item_l}x{item_w}x{item_h}")
    print(f"Valid positions: {result['num_valid']}")
    print(f"Can skip: {result['can_skip']} (should be False - ada valid position)")
    
    assert result['num_valid'] > 0, "Should have valid positions!"
    assert not result['can_skip'], "Should NOT skip - ada valid position!"
    print("✓ PASSED\n")
    
    print("=" * 70)
    print("Test Case 5: Mask no-position (harus skip jika tidak ada valid)")
    print("=" * 70)
    
    height_map.reset()
    # Fill sebagian container saja untuk membuat tidak ada posisi yang valid
    # Buat tall columns yang block placement
    for x in range(0, 59, 6):
        for y in range(0, 23, 6):
            # Pastikan tidak overflow boundary
            end_x = min(x + 5, 59)
            end_y = min(y + 5, 23)
            if end_x > x and end_y > y:
                height_map.update_region(x, y, end_x - x, end_y - y, 23)
    
    # Item yang tidak akan fit di sisa ruang
    item_l, item_w, item_h = 15, 15, 5
    
    result = action_mask.combine_masks(item_l, item_w, item_h, height_map)
    
    print(f"Item: {item_l}x{item_w}x{item_h}")
    print(f"Valid positions: {result['num_valid']}")
    print(f"Can skip: {result['can_skip']} (should be True - tidak ada valid position)")
    
    # Jika tidak ada valid position, skip harus allowed
    if result['num_valid'] == 0:
        assert result['can_skip'], "Should allow skip - tidak ada posisi valid!"
        print("✓ PASSED (no valid position, can skip)\n")
    else:
        print(f"Note: Masih ada {result['num_valid']} valid positions\n")
    
    print("=" * 70)
    print("Test Case 6: Get action vector")
    print("=" * 70)
    
    height_map.reset()
    item_l, item_w, item_h = 5, 5, 5
    
    action_vector = action_mask.get_action_vector(item_l, item_w, item_h, 
                                                  height_map, include_skip=True)
    
    expected_size = 59 * 23 + 1  # positions + skip
    
    print(f"Action vector size: {len(action_vector)}")
    print(f"Expected size: {expected_size}")
    print(f"Valid action count: {np.sum(action_vector > 0)}")
    
    assert len(action_vector) == expected_size, f"Wrong size! {len(action_vector)} vs {expected_size}"
    assert action_vector.dtype == np.float32, "Should be float32!"
    print("✓ PASSED\n")
    
    print("=" * 70)
    print("All Action Masking tests passed!")
    print("=" * 70)
