"""Test cases untuk Repacking module."""

import pytest
from src.planning.repack import Repacker


class TestRepacking:
    """Test suite for Repacking functionality."""

    @pytest.fixture
    def repacker(self):
        """Create repacker instance for testing."""
        return Repacker(container_dims=(59, 23, 23))

    @pytest.fixture
    def test_items(self):
        """Test items for repacking."""
        return [
            (10, 10, 10),  # volume 1000
            (8, 8, 8),     # volume 512
            (6, 6, 6),     # volume 216
            (5, 5, 5),     # volume 125
        ]

    @pytest.fixture
    def test_positions(self):
        """Test positions for items."""
        return [
            (0, 0, 0),
            (10, 0, 0),
            (20, 0, 0),
            (30, 0, 0),
        ]

    def test_repacker_initialization(self, repacker):
        """Test repacker initialization."""
        assert repacker is not None, "Repacker should be initialized"
        # Check that dimensions are stored
        assert hasattr(repacker, 'L'), "Should have L dimension"
        assert hasattr(repacker, 'W'), "Should have W dimension"
        assert hasattr(repacker, 'H'), "Should have H dimension"
        assert repacker.L == 59, "Length should be 59"
        assert repacker.W == 23, "Width should be 23"
        assert repacker.H == 23, "Height should be 23"
        assert repacker.container_volume == 59 * 23 * 23, "Container volume should be calculated correctly"

    def test_bottom_left_fill_repacking(self, repacker, test_items, test_positions):
        """Test Bottom-Left-Fill repacking strategy."""
        success, positions, metric = repacker.attempt_repack_bottom_left_fill(test_items, test_positions)
        
        assert isinstance(success, bool), "Success should be boolean"
        assert isinstance(metric, (int, float)), "Metric should be numeric"
        
        if success:
            assert len(positions) == len(test_items), "Should have positions for all items"

    def test_load_balanced_repacking(self, repacker, test_items, test_positions):
        """Test Load-Balanced repacking strategy."""
        success, positions, metric = repacker.attempt_repack_load_balanced(test_items, test_positions)
        
        assert isinstance(success, bool), "Success should be boolean"
        assert isinstance(metric, (int, float)), "Metric should be numeric"
        
        if success:
            assert len(positions) == len(test_items), "Should have positions for all items"

    def test_minimize_height_repacking(self, repacker, test_items, test_positions):
        """Test Minimize Height repacking strategy."""
        success, positions, metric = repacker.attempt_repack_minimize_height(test_items, test_positions)
        
        assert isinstance(success, bool), "Success should be boolean"
        assert isinstance(metric, (int, float)), "Metric should be numeric"
        
        if success:
            assert len(positions) == len(test_items), "Should have positions for all items"

    def test_auto_repack(self, repacker, test_items, test_positions):
        """Test auto repack strategy selection."""
        result = repacker.auto_repack(test_items, test_positions, strategy='auto')
        
        assert result is not None, "Auto repack should return result"
        assert isinstance(result, dict), "Result should be a dictionary"

    def test_repack_single_item(self, repacker):
        """Test repacking with single item."""
        items = [(5, 5, 5)]
        positions = [(0, 0, 0)]
        
        success, new_positions, metric = repacker.attempt_repack_bottom_left_fill(items, positions)
        
        assert isinstance(success, bool), "Should handle single item"

    def test_repack_empty_items(self, repacker):
        """Test repacking with empty items list."""
        items = []
        positions = []
        
        # Should handle empty list gracefully
        try:
            success, new_positions, metric = repacker.attempt_repack_bottom_left_fill(items, positions)
            # If it doesn't raise an exception, check the result
            assert isinstance(success, bool), "Should return boolean for empty list"
        except (IndexError, ValueError):
            # It's acceptable to raise an error for empty list
            pass
