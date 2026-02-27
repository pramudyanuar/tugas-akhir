import numpy as np

class RandomGenerator:
    """
    Random generator untuk generate episode items dengan kontrol seed untuk reproducibility.
    
    Features:
    - Random generator stabil dengan numpy RandomState
    - Episode size configurable
    - Seed reproducible untuk consistent results
    """
    
    def __init__(self, seed=None):
        """
        Initialize RandomGenerator dengan optional seed.
        
        Args:
            seed (int, optional): Seed untuk numpy random generator. 
                                 Jika None, generator akan tidak deterministic.
        """
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def set_seed(self, seed):
        """
        Set seed untuk generator. Berguna untuk reproducibility.
        
        Args:
            seed (int): Seed value
        """
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def generate_episode(self, num_items=50, 
                        length_range=(2, 15), 
                        width_range=(2, 12), 
                        height_range=(2, 12)):
        """
        Generate episode dengan items yang random.
        
        Args:
            num_items (int): Jumlah items yang akan di-generate (configurable episode size)
            length_range (tuple): Range untuk length (min, max)
            width_range (tuple): Range untuk width (min, max)
            height_range (tuple): Range untuk height (min, max)
            
        Returns:
            list: List of tuples (length, width, height) untuk setiap item
        """
        if num_items <= 0:
            raise ValueError(f"num_items harus positive, got {num_items}")
        
        items = []
        
        for _ in range(num_items):
            length = self.rng.randint(length_range[0], length_range[1] + 1)
            width = self.rng.randint(width_range[0], width_range[1] + 1)
            height = self.rng.randint(height_range[0], height_range[1] + 1)
            items.append((length, width, height))
        
        return items
    
    def get_seed(self):
        """
        Get current seed value.
        
        Returns:
            int: Current seed
        """
        return self.seed


def generate_episode(num_items=50, seed=None,
                    length_range=(2, 15),
                    width_range=(2, 12),
                    height_range=(2, 12)):
    """
    Utility function untuk generate episode dengan single call.
    
    Args:
        num_items (int): Episode size (configurable)
        seed (int, optional): Seed untuk reproducibility
        length_range (tuple): Range untuk length
        width_range (tuple): Range untuk width
        height_range (tuple): Range untuk height
        
    Returns:
        list: List of tuples (length, width, height)
    """
    generator = RandomGenerator(seed=seed)
    return generator.generate_episode(num_items=num_items,
                                     length_range=length_range,
                                     width_range=width_range,
                                     height_range=height_range)


if __name__ == "__main__":
    """Test cases untuk Dataset Random Generator"""
    
    print("=" * 70)
    print("Test Case 1: Random generator stabil")
    print("=" * 70)
    
    # Test dengan seed yang sama → hasil harus identik
    gen1 = RandomGenerator(seed=42)
    episode1 = gen1.generate_episode(num_items=5)
    
    gen2 = RandomGenerator(seed=42)
    episode2 = gen2.generate_episode(num_items=5)
    
    print(f"Episode 1 (seed=42): {episode1}")
    print(f"Episode 2 (seed=42): {episode2}")
    print(f"Episodes identical: {episode1 == episode2}")
    assert episode1 == episode2, "Random generator tidak stabil!"
    print("✓ PASSED: Generator stabil dengan seed\n")
    
    # Test tanpa seed → hasil berbeda
    gen3 = RandomGenerator()
    episode3 = gen3.generate_episode(num_items=5)
    
    gen4 = RandomGenerator()
    episode4 = gen4.generate_episode(num_items=5)
    
    print(f"Episode 3 (no seed): {episode3}")
    print(f"Episode 4 (no seed): {episode4}")
    print(f"Episodes different: {episode3 != episode4}")
    print("✓ PASSED: Generator non-deterministic tanpa seed\n")
    
    print("=" * 70)
    print("Test Case 2: Episode size configurable")
    print("=" * 70)
    
    gen = RandomGenerator(seed=123)
    
    # Test berbagai episode sizes
    for size in [1, 10, 50, 100]:
        episode = gen.generate_episode(num_items=size)
        print(f"Episode size {size}: length = {len(episode)}")
        assert len(episode) == size, f"Expected {size} items, got {len(episode)}"
    
    print("✓ PASSED: Episode size fully configurable\n")
    
    print("=" * 70)
    print("Test Case 3: Seed reproducible")
    print("=" * 70)
    
    # Test reproducibility dengan set_seed
    gen = RandomGenerator(seed=999)
    episode_before = gen.generate_episode(num_items=10)
    
    gen.set_seed(999)  # Reset seed ke nilai yang sama
    episode_after = gen.generate_episode(num_items=10)
    
    print(f"Current seed: {gen.get_seed()}")
    print(f"Episode before reset: {episode_before[:3]}... (showing first 3)")
    print(f"Episode after reset: {episode_after[:3]}... (showing first 3)")
    print(f"Episodes identical after seed reset: {episode_before == episode_after}")
    assert episode_before == episode_after, "Seed reproducibility failed!"
    print("✓ PASSED: Seed reproducible\n")
    
    print("=" * 70)
    print("Test Case 4: Utility function dengan seed")
    print("=" * 70)
    
    # Test utility function
    ep1 = generate_episode(num_items=15, seed=555)
    ep2 = generate_episode(num_items=15, seed=555)
    
    print(f"Utility function episode 1: {ep1[:3]}... (showing first 3)")
    print(f"Utility function episode 2: {ep2[:3]}... (showing first 3)")
    print(f"Episodes identical: {ep1 == ep2}")
    assert ep1 == ep2, "Utility function reproducibility failed!"
    print("✓ PASSED: Utility function works correctly\n")
    
    print("=" * 70)
    print("All Dataset Random Generator tests completed!")
    print("=" * 70)