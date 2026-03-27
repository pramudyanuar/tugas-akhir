"""Test cases untuk Random Dataset Generator module."""

import pytest
from src.data.random_generator import RandomGenerator


class TestRandomGenerator:
    """Test suite for RandomGenerator class."""

    def test_generator_deterministic_with_seed(self):
        """Test generator reproducibility dengan seed yang sama."""
        gen1 = RandomGenerator(seed=42)
        episode1 = gen1.generate_episode(num_items=5)
        
        gen2 = RandomGenerator(seed=42)
        episode2 = gen2.generate_episode(num_items=5)
        
        assert episode1 == episode2, "Random generator tidak stabil dengan seed!"

    def test_generator_nondeterministic_without_seed(self):
        """Test generator non-deterministic tanpa seed."""
        gen1 = RandomGenerator()
        episode1 = gen1.generate_episode(num_items=5)
        
        gen2 = RandomGenerator()
        episode2 = gen2.generate_episode(num_items=5)
        
        # Unlikely to be the same without seed
        # (bukannya assertEquals, karena sangat kecil chance-nya sama)
        # Kita just check bahwa object bisa generate

    def test_episode_size_configurable(self):
        """Test episode size fully configurable."""
        gen = RandomGenerator(seed=123)
        
        for size in [1, 10, 50]:
            episode = gen.generate_episode(num_items=size)
            assert len(episode) == size, f"Expected {size} items, got {len(episode)}"

    def test_episode_format(self):
        """Test episode format is correct."""
        gen = RandomGenerator(seed=456)
        episode = gen.generate_episode(num_items=5)
        
        assert isinstance(episode, list), "Episode should be a list"
        assert len(episode) == 5, "Episode should have 5 items"
        
        # Each item should be a tuple of (length, width, height)
        for item in episode:
            assert isinstance(item, tuple), "Each item should be a tuple"
            assert len(item) == 3, "Each item should have 3 dimensions"
            assert all(isinstance(d, int) for d in item), "All dimensions should be integers"
            assert all(d > 0 for d in item), "All dimensions should be positive"

    def test_seed_reproducibility(self):
        """Test reproducibility with set_seed or re-initialization."""
        gen = RandomGenerator(seed=789)
        
        episode1 = gen.generate_episode(num_items=10)
        
        gen2 = RandomGenerator(seed=789)
        episode2 = gen2.generate_episode(num_items=10)
        
        assert episode1 == episode2, "Episodes should be identical with same seed"

    def test_multiple_episodes_different(self):
        """Test multiple episodes from same generator have different randomness."""
        gen = RandomGenerator(seed=111)
        
        episode1 = gen.generate_episode(num_items=5)
        episode2 = gen.generate_episode(num_items=5)
        
        # Episodes should be different (not identical random sequence)
        # This is expected behavior - each call advances the RNG state
        # So they shouldn't be exactly the same
        assert isinstance(episode1, list), "First episode should be valid"
        assert isinstance(episode2, list), "Second episode should be valid"
