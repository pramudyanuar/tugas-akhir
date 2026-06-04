"""Test cases untuk holding buffer di ContainerEnv."""

import pytest
import numpy as np
from src.core.container_env import ContainerEnv

class TestHoldingBuffer:
    """Test suite for deferred-item holding buffer in ContainerEnv."""

    def test_buffer_initialization(self):
        """Test that the buffer initializes correctly."""
        env = ContainerEnv(
            seed=42,
            buffer_capacity=3,
            max_waiting_steps=5,
            defer_penalty=-0.02,
            overflow_penalty=-0.5
        )
        assert env.buffer_capacity == 3
        assert env.max_waiting_steps == 5
        assert env.defer_penalty == -0.02
        assert env.overflow_penalty == -0.5
        assert len(env.deferred_buffer) == 0
        assert env.num_deferred_items == 0
        assert env.num_deferred_success == 0
        assert env.num_rejected_items == 0

    def test_defer_on_skip(self):
        """Test that skipping an item places it in the buffer."""
        env = ContainerEnv(
            seed=42,
            buffer_capacity=3,
            max_waiting_steps=5
        )
        state, action_mask = env.reset()
        
        # Get current item before skip
        current_item = env.items[env.current_index]
        
        # Skip action
        skip_action = env.L * env.W
        (next_state, next_mask), reward, done, info = env.step(skip_action)
        
        assert len(env.deferred_buffer) == 1
        assert env.deferred_buffer[0]['l'] == current_item['l']
        assert env.deferred_buffer[0]['w'] == current_item['w']
        assert env.deferred_buffer[0]['h'] == current_item['h']
        assert env.num_deferred_items == 1
        assert reward == env.defer_penalty

    def test_buffer_overflow(self):
        """Test buffer overflow when capacity is exceeded and items cannot be placed."""
        env = ContainerEnv(
            seed=42,
            buffer_capacity=1,
            max_waiting_steps=5
        )
        env.reset()
        
        # Set custom items: first is a giant item that can never fit in the bin.
        # Second is a normal item.
        env.items = [
            {'l': 100, 'w': 100, 'h': 100, 'stacking': 'stackable', 'weight': 1000, 'load_bearing': 10},
            {'l': 1, 'w': 1, 'h': 1, 'stacking': 'stackable', 'weight': 1, 'load_bearing': 1000},
        ]
        env.current_index = 0
        
        # Skip the giant item (goes to buffer)
        skip_action = env.L * env.W
        env.step(skip_action)
        assert len(env.deferred_buffer) == 1
        
        # Skip the second item. Since buffer is full (capacity 1), it tries to place
        # the giant item, fails, and rejects the incoming item.
        (next_state, next_mask), reward, done, info = env.step(skip_action)
        
        assert len(env.deferred_buffer) == 1
        assert env.num_rejected_items == 1
        assert reward == env.overflow_penalty

    def test_waiting_time_and_sorting(self):
        """Test that waiting time increments and buffer sorting prioritizes older items."""
        env = ContainerEnv(
            seed=42,
            buffer_capacity=3,
            max_waiting_steps=2
        )
        env.reset()
        
        # Set custom items
        env.items = [
            {'l': 1, 'w': 1, 'h': 1, 'stacking': 'stackable', 'weight': 1, 'load_bearing': 1000},
            {'l': 1, 'w': 1, 'h': 1, 'stacking': 'stackable', 'weight': 1, 'load_bearing': 1000},
            {'l': 1, 'w': 1, 'h': 1, 'stacking': 'stackable', 'weight': 1, 'load_bearing': 1000},
        ]
        env.current_index = 0
        
        # Skip one item (goes to buffer, waiting_time becomes 1 at the end of the step)
        skip_action = env.L * env.W
        env.step(skip_action)
        assert len(env.deferred_buffer) == 1
        assert env.deferred_buffer[0]['waiting_time'] == 1
        
        # Place another item (increment waiting time of buffer item to 2)
        env.step(0)  # place at (0,0)
        assert env.deferred_buffer[0]['waiting_time'] == 2
        
        # Since max_waiting_steps is 2, this item now has waiting_time >= max_waiting_steps
        # When sorting, it should be prioritized.
        # Let's check the sort order
        env.deferred_buffer.append({'l': 1, 'w': 1, 'h': 1, 'stacking': 'stackable', 'waiting_time': 0})
        env.deferred_buffer.sort(key=lambda x: (
            0 if x.get('waiting_time', 0) >= env.max_waiting_steps else 1,
            0 if x.get('stacking') == 'fragile' else 1,
            -x.get('waiting_time', 0)
        ))
        # The first item should be the one with waiting_time == 2
        assert env.deferred_buffer[0]['waiting_time'] == 2

    def test_auto_retry_success(self):
        """Test that deferred items are placed when space becomes available."""
        # Create a tiny container to easily test placement
        env = ContainerEnv(
            container_length=3,
            container_width=3,
            container_height=3,
            buffer_capacity=2,
            max_waiting_steps=5,
            use_structural_validation=False,
            seed=42
        )
        env.reset()
        
        # Manually set items
        env.items = [
            {'l': 2, 'w': 2, 'h': 2, 'stacking': 'stackable', 'weight': 8, 'load_bearing': 1000},
            {'l': 1, 'w': 1, 'h': 1, 'stacking': 'stackable', 'weight': 1, 'load_bearing': 1000},
        ]
        env.current_index = 0
        
        # Skip the 2x2x2 item (deferred)
        skip_action = env.L * env.W
        env.step(skip_action)
        assert len(env.deferred_buffer) == 1
        
        # Place the 1x1x1 item. After it is placed, 2x2x2 item should be retried and placed.
        # Action at (0, 0)
        (next_state, next_mask), reward, done, info = env.step(0)
        
        # Check if the deferred 2x2x2 item was successfully placed during retry!
        assert env.num_deferred_success == 1
        assert len(env.deferred_buffer) == 0
        assert len(env.placed_items) == 2
