"""Test cases untuk Action Masking module."""

import pytest
import numpy as np
from src.core.action_mask import ActionMask
from src.core.height_map import HeightMap


class TestActionMask:
    """Test suite for ActionMask class."""

    def test_mask_out_of_bound(self):
        """Test mask out-of-bound: posisi yang akan membuat item keluar dari container."""
        height_map = HeightMap(59, 23, 23)
        height_map.reset()
        
        action_mask = ActionMask(59, 23, 23)
        
        # Test item yang besar (15x12)
        item_l, item_w = 15, 12
        mask_bound = action_mask.mask_out_of_bound(item_l, item_w, height_map)
        
        # Valid positions: harus x+15 <= 59 dan y+12 <= 23
        # Valid x: 0-43, Valid y: 0-10
        expected_valid = (59 - item_l + 1) * (23 - item_w + 1)
        assert np.sum(mask_bound) == expected_valid, "Out-of-bound mask wrong!"
        assert mask_bound.dtype == bool, "Mask should be boolean"

    def test_mask_overflow(self):
        """Test mask overflow: posisi yang akan membuat height melebihi container height."""
        height_map = HeightMap(59, 23, 23)
        height_map.reset()
        
        action_mask = ActionMask(59, 23, 23)
        
        # Place some items first
        height_map.update_region(0, 0, 10, 10, 20)  # Heights = 20
        height_map.update_region(20, 0, 10, 10, 15)  # Heights = 15
        
        item_h = 5
        mask_overflow = action_mask.mask_overflow(item_h, height_map)
        
        # Position (0, 0) has base height 20, 20+5 > 23, should be masked
        # Position (20, 0) has base height 15, 15+5 <= 23, should be valid
        assert not mask_overflow[0, 0], "Overflow at (0,0) should be masked!"
        assert mask_overflow[20, 0], "No overflow at (20,0)!"

    def test_mask_unstable_lbcp(self):
        """Test mask unstable (LBCP): posisi yang akan unstable."""
        height_map = HeightMap(59, 23, 23)
        height_map.reset()
        
        action_mask = ActionMask(59, 23, 23)
        
        # Create unstable scenario: flat floor at z=0
        # Item 5x5x5 on flat floor should be stable
        item_l, item_w, item_h = 5, 5, 5
        
        mask_unstable = action_mask.mask_unstable_lbcp(item_l, item_w, item_h, height_map)
        
        # Flat floor (all 0) should be stable
        stable_count = np.sum(mask_unstable)
        total_count = np.sum(action_mask.mask_out_of_bound(item_l, item_w, height_map))
        
        assert stable_count == total_count, "Flat floor should be all stable!"

    def test_combine_masks(self):
        """Test combine masks: intersection of all masks."""
        height_map = HeightMap(59, 23, 23)
        height_map.reset()
        
        action_mask = ActionMask(59, 23, 23)
        item_l, item_w, item_h = 10, 10, 5
        
        result = action_mask.combine_masks(item_l, item_w, item_h, height_map)
        
        assert result['num_valid'] > 0, "Should have valid positions!"
        assert not result['can_skip'], "Should NOT skip - ada valid position!"
        assert 'valid_positions' in result
        assert 'combined_mask' in result

    def test_combine_masks_no_position(self):
        """Test combine masks: harus skip jika tidak ada valid position."""
        height_map = HeightMap(59, 23, 23)
        height_map.reset()
        
        # Fill sebagian container untuk membuat tidak ada posisi yang valid
        for x in range(0, 59, 6):
            for y in range(0, 23, 6):
                end_x = min(x + 5, 59)
                end_y = min(y + 5, 23)
                if end_x > x and end_y > y:
                    height_map.update_region(x, y, end_x - x, end_y - y, 23)
        
        action_mask = ActionMask(59, 23, 23)
        item_l, item_w, item_h = 15, 15, 5
        
        result = action_mask.combine_masks(item_l, item_w, item_h, height_map)
        
        # Jika tidak ada valid position, skip harus allowed
        if result['num_valid'] == 0:
            assert result['can_skip'], "Should allow skip - tidak ada posisi valid!"

    def test_get_action_vector(self):
        """Test get action vector: flatten mask untuk neural network."""
        height_map = HeightMap(59, 23, 23)
        height_map.reset()
        
        action_mask = ActionMask(59, 23, 23)
        item_l, item_w, item_h = 5, 5, 5
        
        action_vector = action_mask.get_action_vector(item_l, item_w, item_h, 
                                                      height_map, include_skip=True)
        
        expected_size = 59 * 23 + 1  # positions + skip
        
        assert len(action_vector) == expected_size, f"Wrong size! {len(action_vector)} vs {expected_size}"
        assert action_vector.dtype == np.float32, "Should be float32!"
        assert np.all((action_vector == 0) | (action_vector == 1)), "Mask should be binary (0 or 1)"
