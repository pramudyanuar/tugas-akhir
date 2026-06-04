"""Test cases untuk ContainerEnv module."""

import pytest
import numpy as np
from src.core.container_env import ContainerEnv


class TestContainerEnv:
    """Test suite for ContainerEnv class."""

    def test_environment_initialization(self):
        """Test environment initialization."""
        env = ContainerEnv(seed=42)
        state, action_mask = env.reset()
        
        assert state.shape == (env.state_size,), f"Wrong state shape! {state.shape}"
        assert action_mask.shape == (env.action_size,), f"Wrong action mask shape!"
        assert np.sum(action_mask > 0) > 0, "Should have some valid actions"
        assert isinstance(state, np.ndarray), "State should be numpy array"
        assert isinstance(action_mask, np.ndarray), "Action mask should be numpy array"

    def test_environment_reset(self):
        """Test environment reset."""
        env = ContainerEnv(seed=42)
        
        state1, mask1 = env.reset()
        state2, mask2 = env.reset()
        
        # Different resets might have different states due to random item generation
        # But state and mask shapes should be consistent
        assert state1.shape == state2.shape, "State shapes should be consistent"
        assert mask1.shape == mask2.shape, "Mask shapes should be consistent"

    def test_step_with_valid_placement(self):
        """Test step with valid placement."""
        env = ContainerEnv(seed=123)
        state, action_mask = env.reset()
        
        # Find first valid action
        valid_positions = np.where(action_mask > 0)[0]
        if len(valid_positions) > 0:
            action = valid_positions[0]
            (next_state, next_mask), reward, done, info = env.step(action)
            
            assert next_state.shape == (env.state_size,), "Wrong next state shape!"
            assert reward > 0, "Reward should be positive for valid placement!"
            assert 'success' in info, "Info should contain success flag"
            assert isinstance(done, (bool, np.bool_)), "Done should be boolean"

    def test_skip_action(self):
        """Test skip action."""
        env = ContainerEnv(seed=456)
        state, action_mask = env.reset()
        
        skip_action = env.L * env.W
        (next_state, next_mask), reward, done, info = env.step(skip_action)
        
        assert info['action_type'] in ['skip', 'defer'], f"Should be skip or defer action, got: {info['action_type']}"


    def test_utilization_calculation(self):
        """Test utilization calculation during episode."""
        env = ContainerEnv(seed=789)
        state, action_mask = env.reset()
        
        # Place a few items
        for _ in range(3):
            valid_positions = np.where(action_mask > 0)[0]
            if len(valid_positions) > 0:
                action = valid_positions[0]
                (state, action_mask), reward, done, info = env.step(action)
                
                if done:
                    break

    def test_environment_dimensions(self):
        """Test environment has correct dimensions."""
        env = ContainerEnv(container_length=60, container_width=24, container_height=26)
        
        assert env.L == 60, "Length should be 60"
        assert env.W == 24, "Width should be 24"
        assert env.H == 26, "Height should be 26"

    def test_max_items_constraint(self):
        """Test max items constraint."""
        max_items = 5
        env = ContainerEnv(max_items=max_items, seed=999)
        state, action_mask = env.reset()
        
        # Keep stepping until we hit max items
        for _ in range(max_items + 5):
            valid_positions = np.where(action_mask > 0)[0]
            if len(valid_positions) > 0:
                action = valid_positions[0]
                (state, action_mask), reward, done, info = env.step(action)
            else:
                action = env.L * env.W
                (state, action_mask), reward, done, info = env.step(action)
            
            if done:
                assert env.current_index <= max_items, "Should not exceed max items"
                break
