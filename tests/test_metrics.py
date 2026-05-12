"""Test cases untuk Metrics module."""

import pytest
import numpy as np
from src.utils.metrics import Metrics


class TestMetrics:
    """Test suite for Metrics class."""

    def test_calculate_utilization(self):
        """Test hitung utilization percentage."""
        placed_items = [
            {'l': 10, 'w': 10, 'h': 5, 'stacking': 'stackable'},
            {'l': 8, 'w': 8, 'h': 6, 'stacking': 'stackable'},
            {'l': 6, 'w': 6, 'h': 4, 'stacking': 'stackable'},
        ]
        container = (30, 30, 30)  # 27000 volume
        
        utilization = Metrics.calculate_utilization(placed_items, container)
        
        assert 0 <= utilization <= 100, "Utilization out of range!"
        assert isinstance(utilization, (int, float)), "Utilization should be numeric"

    def test_calculate_success_rate_perfect(self):
        """Test success rate: perfect placement."""
        success_rate = Metrics.calculate_success_rate(10, 10)
        assert success_rate == 100.0, "Should be 100%!"

    def test_calculate_success_rate_partial(self):
        """Test success rate: partial placement."""
        success_rate = Metrics.calculate_success_rate(5, 10)
        assert success_rate == 50.0, "Should be 50%!"

    def test_calculate_success_rate_none(self):
        """Test success rate: no placement."""
        success_rate = Metrics.calculate_success_rate(0, 10)
        assert success_rate == 0.0, "Should be 0%!"

    def test_calculate_average_height_distribution(self):
        """Test hitung average height distribution."""
        placed_items = [
            {'l': 5, 'w': 5, 'h': 2, 'stacking': 'stackable'},
            {'l': 5, 'w': 5, 'h': 3, 'stacking': 'stackable'},
            {'l': 5, 'w': 5, 'h': 4, 'stacking': 'stackable'},
            {'l': 5, 'w': 5, 'h': 5, 'stacking': 'stackable'},
        ]
        
        distribution = Metrics.calculate_average_height_distribution(placed_items)
        
        assert distribution['average'] == 3.5, "Average should be 3.5"
        assert distribution['max'] == 5, "Max should be 5"
        assert distribution['min'] == 2, "Min should be 2"
        assert distribution['count'] == 4, "Count should be 4"
        assert 'std_dev' in distribution, "Should have std_dev"

    def test_calculate_height_distribution_empty(self):
        """Test height distribution with empty list."""
        distribution = Metrics.calculate_average_height_distribution([])
        
        # Should handle empty list gracefully
        assert distribution['count'] == 0, "Count should be 0 for empty list"
