"""
Integration tests untuk Algorithm 1-4.
Memvalidasi bahwa semua algoritma terimplementasi dan bekerja dengan baik dalam pipeline.
"""

import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.perfect_pack_generator import PerfectPackGenerator, generate_perfect_pack
from src.core.feasibility_map import FeasibilityMap, update_feasibility_map
from src.core.height_map import HeightMap
from src.core.stability_validator import StabilityValidator
from src.planning.repack_trial import RepackTrial
from src.planning.tree_expansion import TreeExpander
from src.planning.high_level_search import HighLevelSearcher
from src.core.container_env import ContainerEnv


class TestAlgorithm4:
    """Test Algorithm 4: 100% Set Instance Generation"""

    def test_perfect_pack_generator_init(self):
        """Test generator initialization"""
        gen = PerfectPackGenerator(bin_width=23, bin_height=23, sigma=2, seed=42)
        assert gen.W == 23
        assert gen.H == 23
        assert gen.sigma == 2
        assert gen.rng is not None

    def test_perfect_pack_generate(self):
        """Test perfect pack generation"""
        gen = PerfectPackGenerator(bin_width=23, bin_height=23, sigma=2, seed=42)
        items = gen.generate_perfect_pack(num_attempts=3)
        
        assert isinstance(items, list)
        assert len(items) > 0
        
        # Check utilization
        total_volume = sum(item[0] * item[1] for item in items)
        container_volume = 23 * 23
        util = total_volume / container_volume
        
        # Must achieve ~100% utilization (at least 95%)
        assert util >= 0.95, f"Utilization too low: {util}"

    def test_perfect_pack_function(self):
        """Test convenience function"""
        items = generate_perfect_pack(bin_width=23, bin_height=23, sigma=2, seed=42)
        assert isinstance(items, list)
        assert len(items) > 0

    def test_generate_episode_interface(self):
        """Test generate_episode interface"""
        gen = PerfectPackGenerator(bin_width=23, bin_height=23, seed=42)
        items = gen.generate_episode(num_items=50)  # num_items ignored
        
        assert isinstance(items, list)
        assert len(items) > 0


class TestAlgorithm2:
    """Test Algorithm 2: Update Feasibility Map"""

    def test_feasibility_map_init(self):
        """Test FeasibilityMap initialization"""
        fm = FeasibilityMap(length=59, width=23)
        assert fm.L == 59
        assert fm.W == 23
        assert fm.map.shape == (59, 23)
        # Initially all feasible
        assert np.all(fm.map == True)

    def test_feasibility_map_reset(self):
        """Test reset functionality"""
        fm = FeasibilityMap(length=59, width=23)
        fm.map[0, 0] = False
        fm.reset()
        assert np.all(fm.map == True)

    def test_feasibility_map_update(self):
        """Test update from placement"""
        hm = HeightMap(59, 23)
        fm = FeasibilityMap(59, 23)
        
        # Place an item
        hm.update_region(5, 5, 3, 3, 5)
        
        # Update feasibility map
        success = fm.update_from_placement(hm, 5, 5, 3, 3, 5)
        # May succeed or fail depending on support cells, but should not crash
        assert isinstance(success, bool)

    def test_feasibility_check(self):
        """Test feasibility checking"""
        fm = FeasibilityMap(59, 23)
        assert fm.is_feasible(0, 0) == True
        assert fm.is_feasible(100, 100) == False  # Out of bounds


class TestAlgorithm3:
    """Test Algorithm 3: Repack Trial"""

    def test_repack_trial_init(self):
        """Test RepackTrial initialization"""
        rt = RepackTrial(container_dims=(59, 23, 23), time_limit=5.0)
        assert rt.L == 59
        assert rt.W == 23
        assert rt.H == 23
        assert rt.time_limit == 5.0

    def test_repack_trial_attempt(self):
        """Test repack attempt"""
        rt = RepackTrial(container_dims=(59, 23, 23), time_limit=1.0)
        
        env_state = {
            'placed_items': [(5, 5, 5), (3, 3, 3)],
            'placed_positions': [(0, 0, 0), (5, 0, 0)]
        }
        
        result = rt.attempt_repack(env_state, require_full_pack=False)
        
        assert isinstance(result, dict)
        assert 'success' in result
        assert 'actions' in result
        assert 'best_util' in result
        assert 'time_elapsed' in result


class TestAlgorithm1And2:
    """Test Algorithm 1 & 2: Tree Expansion"""

    def test_tree_expander_init(self):
        """Test TreeExpander initialization"""
        env = ContainerEnv(container_length=20, container_width=10, container_height=10)
        expander = TreeExpander(env, max_depth=10)
        
        assert expander.env is env
        assert expander.max_depth == 10

    def test_tree_expansion(self):
        """Test tree expansion"""
        env = ContainerEnv(container_length=20, container_width=10, container_height=10)
        expander = TreeExpander(env, max_depth=5)
        
        # Create simple state
        state = {
            'items': [(5, 5, 5), (3, 3, 3)],
            'current_index': 0,
            'height_map': HeightMap(20, 10).map,
        }
        
        sequences, solved = expander.tree_expansion(state)
        
        # Should return lists
        assert isinstance(sequences, list)
        assert isinstance(solved, bool)


class TestAlgorithm1HighLevel:
    """Test Algorithm 1: High-Level Search"""

    def test_high_level_searcher_init(self):
        """Test HighLevelSearcher initialization"""
        env = ContainerEnv(container_length=20, container_width=10, container_height=10)
        searcher = HighLevelSearcher(env, max_depth=10, use_repack=True)
        
        assert searcher.env is env
        assert searcher.max_depth == 10
        assert searcher.use_repack == True

    def test_high_level_search(self):
        """Test high-level search"""
        env = ContainerEnv(container_length=20, container_width=10, container_height=10)
        searcher = HighLevelSearcher(env, max_depth=5, mcts_budget=10)
        
        # Create simple state
        state = {
            'items': [(5, 5, 5), (3, 3, 3)],
            'current_index': 0,
            'height_map': HeightMap(20, 10).map,
            'placed_items': [],
            'placed_positions': []
        }
        
        result = searcher.search(state)
        
        assert isinstance(result, dict)
        assert 'success' in result
        assert 'action' in result
        assert 'utilization' in result


class TestIntegration:
    """Integration tests untuk semua algoritma dalam pipeline"""

    def test_container_env_with_perfect_pack(self):
        """Test ContainerEnv dengan perfect_pack dataset"""
        env = ContainerEnv(
            container_length=20,
            container_width=10,
            container_height=10,
            dataset_type='perfect_pack',
            seed=42
        )
        
        state, action_mask = env.reset()
        
        assert state.shape[0] == env.state_size
        assert action_mask.shape[0] == env.action_size

    def test_feasibility_map_in_workflow(self):
        """Test feasibility map dalam workflow"""
        hm = HeightMap(59, 23)
        fm = FeasibilityMap(59, 23)
        
        # Simulate multiple item placements
        for i in range(3):
            x, y = i * 5, i * 3
            hm.update_region(x, y, 3, 3, 5)
            update_feasibility_map(fm, hm, x, y, 3, 3, 5)
        
        # Check some feasibility
        feasible_count = np.sum(fm.map)
        assert feasible_count > 0

    def test_stability_validation_consistency(self):
        """Test stability validation untuk 100% pack items"""
        gen = PerfectPackGenerator(bin_width=23, bin_height=23, seed=42)
        items = gen.generate_perfect_pack()
        
        # All items dari perfect pack should be valid dimensions
        for item in items:
            assert len(item) == 2
            assert item[0] > 0 and item[1] > 0
            assert item[0] <= 23 and item[1] <= 23

    def test_full_pipeline_flow(self):
        """Test complete pipeline flow"""
        # 1. Generate perfect pack items
        gen = PerfectPackGenerator(bin_width=10, bin_height=10, seed=42)
        items = gen.generate_perfect_pack()
        
        # 2. Create environment
        env = ContainerEnv(
            container_length=10,
            container_width=10,
            container_height=10,
            dataset_type='perfect_pack',
            seed=42
        )
        state, mask = env.reset()
        
        # 3. Test feasibility map update
        fm = FeasibilityMap(10, 10)
        initial_ratio = fm.get_feasibility_ratio()
        assert initial_ratio == 1.0  # Initially all feasible
        
        # 4. Test repack trial
        env_state = {
            'placed_items': [(3, 3, 3)],
            'placed_positions': [(0, 0, 0)]
        }
        rt = RepackTrial((10, 10, 10))
        repack_result = rt.attempt_repack(env_state)
        assert 'success' in repack_result


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])
