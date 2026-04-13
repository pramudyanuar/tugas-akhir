"""
Algorithm 2: Structural Stability Update
Updates feasibility map berdasarkan Load-Balanced Contact Points (LBCPs).

Main Features:
- Maintains feasibility map (FMt) untuk tracking stable placement regions
- Updates based on support polygon dari newly placed items
- Integrates with StabilityValidator untuk consistency
"""

import numpy as np
from .stability_validator import StabilityValidator


class FeasibilityMap:
    """
    Feasibility Map untuk tracking stable placement regions.

    The feasibility map FMt adalah 2D boolean array yang menunjukkan:
    - FMt[x, y] = True: feasible untuk place items (part of stable support polygon)
    - FMt[x, y] = False: tidak feasible (tidak stable)

    Update strategy:
    1. Ketika item baru di-place, compute support polygon (convex hull dari support cells)
    2. Mark semua points dalam support polygon sebagai feasible
    3. Accumulate feasibility information across placements
    """

    def __init__(self, length=59, width=23):
        """
        Initialize FeasibilityMap.

        Args:
            length (int): Container length (L dimension)
            width (int): Container width (W dimension)
        """
        self.L = length
        self.W = width
        
        # Initialize: semua points initially feasible (sebelum ada items)
        self.map = np.ones((self.L, self.W), dtype=bool)
        
        # Track history untuk debugging
        self.update_history = []

    def reset(self):
        """Reset feasibility map ke initial state."""
        self.map = np.ones((self.L, self.W), dtype=bool)
        self.update_history.clear()

    def update_from_placement(self, height_map, x, y, l, w, h):
        """
        Algorithm 2: UpdateFeasibilityMap

        Input:
            - FMt: Current feasibility map
            - Pt+1: Support polygon dari newly placed item

        Output:
            - FMt+1: Updated feasibility map

        Procedure:
            1. Get support polygon P△_new dari newly placed item
            2. For each (x, y) dalam P△_new:
                FMt+1[x, y] ← true (mark as feasible)

        Args:
            height_map (HeightMap): Current height map
            x, y: Koordinat awal item
            l, w: Panjang dan lebar item
            h: Tinggi item
        """
        try:
            # Extract height map as numpy array
            hm = height_map.map if hasattr(height_map, 'map') else height_map
            
            # Compute base height (tempat item bertumpu)
            region = hm[x:x+l, y:y+w]
            base_height = np.max(region)
            
            # Step 1: Compute support cells dari placement region
            support_cells = StabilityValidator.compute_support_cells(
                hm, x, y, l, w, base_height
            )
            
            if len(support_cells) < 3:
                # Tidak cukup support cells, tidak update
                return False
            
            # Step 2: Compute support polygon (convex hull)
            try:
                hull_points, _ = StabilityValidator.compute_convex_hull(support_cells)
            except Exception:
                # Convex hull computation failed (collinear), tidak update
                return False
            
            # Step 3: Mark points dalam support polygon sebagai feasible
            # (Algoritm 2 line 7-9)
            for px, py in hull_points:
                px, py = int(px), int(py)
                if 0 <= px < self.L and 0 <= py < self.W:
                    self.map[px, py] = True
            
            # Record update untuk history
            self.update_history.append({
                'item_pos': (x, y, l, w, h),
                'base_height': base_height,
                'support_cells': len(support_cells),
                'hull_size': len(hull_points)
            })
            
            return True
            
        except Exception as e:
            # Error saat update, tidak modify map
            return False

    def is_feasible(self, x, y):
        """
        Check apakah position (x, y) adalah feasible.

        Args:
            x, y: Koordinat untuk check

        Returns:
            bool: True jika feasible
        """
        if 0 <= x < self.L and 0 <= y < self.W:
            return bool(self.map[x, y])
        return False

    def get_feasible_region(self, x, y, l, w):
        """
        Get feasibility status untuk entire region.

        Args:
            x, y: Koordinat awal region
            l, w: Dimensi region

        Returns:
            np.ndarray: Boolean array, True = feasible
        """
        if x < 0 or y < 0 or x + l > self.L or y + w > self.W:
            # Out of bounds
            return np.zeros((l, w), dtype=bool)
        
        return self.map[x:x+l, y:y+w].copy()

    def get_feasible_positions(self, x, y, l, w):
        """
        Get list of feasible positions dalam region.

        Args:
            x, y: Koordinat awal region
            l, w: Dimensi region

        Returns:
            list: List of (px, py) feasible positions
        """
        positions = []
        for px in range(x, min(x + l, self.L)):
            for py in range(y, min(y + w, self.W)):
                if self.map[px, py]:
                    positions.append((px, py))
        return positions

    def get_feasibility_ratio(self):
        """
        Get feasibility ratio (percentage of feasible cells).

        Returns:
            float: Ratio dalam range [0, 1]
        """
        total = self.L * self.W
        feasible = np.sum(self.map)
        return float(feasible) / total if total > 0 else 0.0

    def visualize(self):
        """
        Get visualization array untuk feasibility map.

        Returns:
            np.ndarray: Visualization (1 = feasible, 0 = not feasible)
        """
        return self.map.astype(np.int32)


def update_feasibility_map(fm, height_map, x, y, l, w, h):
    """
    Convenience function untuk update feasibility map.

    Args:
        fm (FeasibilityMap): Feasibility map object
        height_map: Height map object atau array
        x, y: Item position
        l, w: Item dimensions
        h: Item height

    Returns:
        bool: True if update successful
    """
    return fm.update_from_placement(height_map, x, y, l, w, h)
