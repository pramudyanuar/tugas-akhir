import numpy as np

class HeightMap:
    def __init__(self, length=59, width=23, height=23):
        self.L = length
        self.W = width
        self.H = height
        self.map = np.zeros((self.L, self.W), dtype=np.int32)

    def reset(self):
        self.map.fill(0)

    def _is_valid_bounds(self, x, y, l, w):
        """
        Collision boundary check: memastikan koordinat dan dimensi berada dalam batas
        
        Args:
            x, y: koordinat awal
            l, w: panjang dan lebar region
            
        Returns:
            bool: True jika boundary valid, False jika invalid
        """
        return (x >= 0 and y >= 0 and 
                x + l <= self.L and y + w <= self.W)

    def get_region(self, x, y, l, w):
        """Get region dengan collision boundary check"""
        if not self._is_valid_bounds(x, y, l, w):
            raise ValueError(
                f"Invalid bounds: x={x}, y={y}, l={l}, w={w}. "
                f"Map dimensions: L={self.L}, W={self.W}"
            )
        return self.map[x:x+l, y:y+w]

    def max_height_in_region(self, x, y, l, w):
        """Get max height dengan collision boundary check"""
        if not self._is_valid_bounds(x, y, l, w):
            raise ValueError(
                f"Invalid bounds: x={x}, y={y}, l={l}, w={w}. "
                f"Map dimensions: L={self.L}, W={self.W}"
            )
        return np.max(self.get_region(x, y, l, w))

    def update_region(self, x, y, l, w, new_height):
        """
        Update region dengan:
        - Collision boundary check (memastikan region dalam batas map)
        - Overflow check (memastikan height tidak melebihi H)
        
        Args:
            x, y: koordinat awal
            l, w: panjang dan lebar region
            new_height: tinggi baru untuk region
            
        Raises:
            ValueError: jika boundary invalid atau overflow terjadi
        """
        # Collision boundary check
        if not self._is_valid_bounds(x, y, l, w):
            raise ValueError(
                f"Invalid bounds: x={x}, y={y}, l={l}, w={w}. "
                f"Map dimensions: L={self.L}, W={self.W}"
            )
        
        # Overflow check
        if new_height > self.H:
            raise ValueError(
                f"Overflow: new_height={new_height} exceeds max height H={self.H}"
            )
        
        if new_height < 0:
            raise ValueError(
                f"Invalid height: new_height={new_height} must be non-negative"
            )
        
        self.map[x:x+l, y:y+w] = new_height

    def normalize(self):
        return self.map / self.H