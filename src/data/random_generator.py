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

    print("=" * 70)