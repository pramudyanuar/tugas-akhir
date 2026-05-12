"""Test cases untuk RepackTrial module (Algorithm 3)."""

import pytest
import numpy as np
from src.planning.repack_trial import RepackTrial
from src.core.height_map import HeightMap


class TestRepackTrial:
    """Test suite for RepackTrial (Algorithm 3) functionality."""

    @pytest.fixture
    def repack_trial(self):
        """Create RepackTrial instance for testing."""
        return RepackTrial(container_dims=(59, 23, 23), time_limit=5.0)

    @pytest.fixture
    def test_env_state(self):
        """Test environment state for repacking."""
        height_map = HeightMap(59, 23, 23)
        return {
            'items': [
                {'l': 10, 'w': 10, 'h': 10, 'stacking': 'stackable'},
                {'l': 8, 'w': 8, 'h': 8, 'stacking': 'stackable'},
                {'l': 6, 'w': 6, 'h': 6, 'stacking': 'stackable'},
                {'l': 5, 'w': 5, 'h': 5, 'stacking': 'stackable'},
            ],
            'current_index': 4,
            'height_map': height_map,
            'placed_items': [
                {'l': 10, 'w': 10, 'h': 10, 'stacking': 'stackable'},
                {'l': 8, 'w': 8, 'h': 8, 'stacking': 'stackable'},
                {'l': 6, 'w': 6, 'h': 6, 'stacking': 'stackable'},
            ],
            'placed_positions': [(0, 0, 0), (10, 0, 0), (20, 0, 0)]
        }

    def test_repack_trial_initialization(self, repack_trial):
        """Test RepackTrial initialization."""
        assert repack_trial is not None, "RepackTrial should be initialized"
        assert repack_trial.L == 59, "Length should be 59"
        assert repack_trial.W == 23, "Width should be 23"
        assert repack_trial.H == 23, "Height should be 23"
        assert repack_trial.time_limit == 5.0, "Time limit should be 5.0"

    def test_attempt_repack_returns_dict(self, repack_trial, test_env_state):
        """Test that attempt_repack returns a dictionary with expected keys."""
        result = repack_trial.attempt_repack(test_env_state, require_full_pack=False)
        
        assert isinstance(result, dict), "Result should be a dictionary"
        assert 'success' in result, "Result should have 'success' key"
        assert 'actions' in result, "Result should have 'actions' key"
        assert 'best_util' in result, "Result should have 'best_util' key"

    def test_attempt_repack_with_placed_items(self, repack_trial, test_env_state):
        """Test attempt_repack with already placed items."""
        result = repack_trial.attempt_repack(test_env_state, require_full_pack=False)
        
        # Result should indicate success or failure
        assert result['success'] in [True, False]
        # Best utilization should be a number
        assert isinstance(result['best_util'], (int, float))
        # Actions should be a list
        assert isinstance(result['actions'], (list, tuple))

    def test_repack_trial_with_empty_state(self, repack_trial):
        """Test RepackTrial with empty placed items."""
        env_state = {
            'items': [{'l': 10, 'w': 10, 'h': 10, 'stacking': 'stackable'}],
            'current_index': 0,
            'height_map': HeightMap(59, 23, 23),
            'placed_items': [],
            'placed_positions': []
        }
        
        result = repack_trial.attempt_repack(env_state, require_full_pack=False)
        
        # With no placed items, repack should not find anything to rearrange
        assert isinstance(result, dict), "Should return a dictionary"
        
        success = result.get('success')
        metric = result.get('best_util')

        assert isinstance(success, bool), "Success should be boolean"
        assert isinstance(metric, (int, float)), "Metric should be numeric"

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
        items = [{'l': 5, 'w': 5, 'h': 5, 'stacking': 'stackable'}]
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
