import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from env.lbcp import is_stable
from env.height_map import HeightMap


class Repacker:
    """
    Repacking mechanism untuk reorganisasi items dalam kontainer.
    
    Tujuan repacking:
    1. Maximize volume utilization (pack lebih rapat)
    2. Optimize load distribution (balance beban)
    3. Minimize maximum height (free floor space)
    
    Strategies:
    - Bottom-Left-Fill (BLF): Pack dari sudut kiri-bawah
    - Best-Fit Decreasing (BFD): Sorted by volume, fit ke slot terbaik
    - Load-Balanced: Prioritize balanced weight distribution
    """
    
    def __init__(self, container_dims=(59, 23, 23)):
        """
        Initialize repacker.
        
        Args:
            container_dims: Tuple of (length, width, height)
        """
        self.L, self.W, self.H = container_dims
        self.container_volume = self.L * self.W * self.H
    
    def attempt_repack_bottom_left_fill(self, placed_items, placed_positions):
        """
        Repack menggunakan Bottom-Left-Fill heuristic.
        
        Algoritma:
        1. Collect semua placed items
        2. Clear container
        3. Sort items by volume (descending)
        4. Place items dari sudut kiri-bawah
        5. Return new positions jika berhasil
        
        Args:
            placed_items: List of (length, width, height)
            placed_positions: List of (x, y, z) - current positions
            
        Returns:
            tuple: (success, new_positions, improvement_ratio)
                - success: bool, repacking berhasil?
                - new_positions: list of new (x, y, z)
                - improvement_ratio: new_util / old_util
        """
        if len(placed_items) == 0:
            return False, [], 1.0
        
        # Calculate original utilization
        original_volume = sum(item[0] * item[1] * item[2] for item in placed_items)
        original_util = original_volume / self.container_volume
        
        # Create new height map
        height_map = HeightMap(self.L, self.W, self.H)
        
        # Sort items by volume (descending)
        sorted_indices = sorted(range(len(placed_items)), 
                               key=lambda i: placed_items[i][0] * placed_items[i][1] * placed_items[i][2],
                               reverse=True)
        sorted_items = [placed_items[i] for i in sorted_indices]
        
        # Try to place items
        new_positions = []
        placed_count = 0
        
        for item_l, item_w, item_h in sorted_items:
            # Try to find position using Bottom-Left-Fill
            placed = False
            
            # Scan dari z=0 (bottom)
            for z in range(self.H - item_h + 1):
                if placed:
                    break
                
                # Scan dari y=0 (left)
                for y in range(self.W - item_w + 1):
                    if placed:
                        break
                    
                    # Scan dari x=0 (left)
                    for x in range(self.L - item_l + 1):
                        # Check if position is free and stable
                        if self._can_place_at(height_map, x, y, item_l, item_w, item_h, z):
                            # Place item
                            height_map.update_region_absolute(x, y, item_l, item_w, z + item_h)
                            new_positions.append((x, y, z))
                            placed_count += 1
                            placed = True
                            break
        
        # Check if repacking was successful
        if placed_count < len(placed_items):
            # Not all items could be placed, abort repacking
            return False, [], 1.0
        
        # Calculate new utilization (same volume, but potentially better arrangement)
        new_max_height = np.max(height_map.map)
        improvement_ratio = original_util  # Same volume, so util is same
        
        return True, new_positions, improvement_ratio
    
    def attempt_repack_load_balanced(self, placed_items, placed_positions, num_sections=4):
        """
        Repack untuk mengoptimalkan load distribution.
        
        Algoritma:
        1. Divide container menjadi N sections
        2. Calculate target weight per section
        3. Redistribute items untuk balance beban
        4. Prioritas: items dengan berat besar di tengah
        
        Args:
            placed_items: List of (length, width, height)
            placed_positions: List of (x, y, z)
            num_sections: Number of vertical sections
            
        Returns:
            tuple: (success, new_positions, load_balance_score)
        """
        if len(placed_items) == 0:
            return False, [], 1.0
        
        # Create new height map
        height_map = HeightMap(self.L, self.W, self.H)
        
        # Sort items by volume (descending)
        sorted_indices = sorted(range(len(placed_items)),
                               key=lambda i: placed_items[i][0] * placed_items[i][1] * placed_items[i][2],
                               reverse=True)
        
        # Calculate section boundaries
        section_width = self.W / num_sections
        
        # Try to place in load-balanced manner
        new_positions = []
        placed_count = 0
        
        for idx in sorted_indices:
            item_l, item_w, item_h = placed_items[idx]
            
            # Prefer middle sections for balance
            section_preference = [1, 2, 0, 3]  # Middle sections first
            
            placed = False
            for section_id in section_preference:
                if placed:
                    break
                
                # Calculate section bounds
                sec_y_min = int(section_id * section_width)
                sec_y_max = int((section_id + 1) * section_width)
                
                # Try to place in this section
                for y in range(max(0, sec_y_min), min(self.W - item_w + 1, sec_y_max)):
                    if placed:
                        break
                    
                    for x in range(self.L - item_l + 1):
                        z = height_map.max_height_in_region(x, y, item_l, item_w)
                        
                        if z + item_h <= self.H and self._can_place_at(height_map, x, y, item_l, item_w, item_h, z):
                            height_map.update_region_absolute(x, y, item_l, item_w, z + item_h)
                            new_positions.append((x, y, z))
                            placed_count += 1
                            placed = True
                            break
        
        if placed_count < len(placed_items):
            return False, [], 1.0
        
        # Calculate load balance score
        load_balance_score = self._calculate_load_balance(new_positions, placed_items)
        
        return True, new_positions, load_balance_score
    
    def attempt_repack_minimize_height(self, placed_items, placed_positions):
        """
        Repack untuk meminimalkan maksimum height.
        
        Tujuan: Free up floor space untuk items baru.
        
        Args:
            placed_items: List of (length, width, height)
            placed_positions: List of (x, y, z)
            
        Returns:
            tuple: (success, new_positions, height_reduction_ratio)
        """
        if len(placed_items) == 0:
            return False, [], 1.0
        
        # Calculate original max height
        original_max_height = max(z + placed_items[i][2] 
                                 for i, (x, y, z) in enumerate(placed_positions))
        
        # Create new height map
        height_map = HeightMap(self.L, self.W, self.H)
        
        # Sort items: tall items first (greedy to minimize height)
        sorted_indices = sorted(range(len(placed_items)),
                               key=lambda i: placed_items[i][2],
                               reverse=True)
        
        new_positions = []
        placed_count = 0
        
        for idx in sorted_indices:
            item_l, item_w, item_h = placed_items[idx]
            
            # Find position dengan minimum height impact
            best_pos = None
            best_height = float('inf')
            
            for y in range(self.W - item_w + 1):
                for x in range(self.L - item_l + 1):
                    z = height_map.max_height_in_region(x, y, item_l, item_w)
                    
                    if z + item_h <= self.H and self._can_place_at(height_map, x, y, item_l, item_w, item_h, z):
                        if z + item_h < best_height:
                            best_pos = (x, y, z)
                            best_height = z + item_h
            
            if best_pos is not None:
                x, y, z = best_pos
                height_map.update_region_absolute(x, y, item_l, item_w, z + item_h)
                new_positions.append(best_pos)
                placed_count += 1
        
        if placed_count < len(placed_items):
            return False, [], 1.0
        
        # Calculate height reduction
        new_max_height = np.max(height_map.map)
        height_reduction = original_max_height / max(new_max_height, 1)
        
        return True, new_positions, height_reduction
    
    def _can_place_at(self, height_map, x, y, item_l, item_w, item_h, z):
        """
        Check if item dapat ditempatkan pada posisi absolute z.
        
        Args:
            height_map: HeightMap instance
            x, y: Position
            item_l, item_w, item_h: Item dimensions
            z: Absolute z position (tidak relative)
            
        Returns:
            bool: True jika dapat ditempatkan
        """
        # Check boundary
        if x + item_l > self.L or y + item_w > self.W or z + item_h > self.H:
            return False
        
        # Check if region is free (all cells at z should be <= z)
        region = height_map.map[x:x+item_l, y:y+item_w]
        if np.any(region > z):
            # Region tidak free pada level z
            return False
        
        return True
    
    def _calculate_load_balance(self, positions, items):
        """Calculate load balance coefficient."""
        if not items:
            return 1.0
        
        # Divide container into quadrants
        mid_x = self.L / 2.0
        mid_y = self.W / 2.0
        
        quadrant_weights = [0.0, 0.0, 0.0, 0.0]
        
        for pos, item in zip(positions, items):
            x, y, z = pos
            l, w, h = item
            
            center_x = x + l / 2.0
            center_y = y + w / 2.0
            weight = l * w * h
            
            if center_x < mid_x and center_y < mid_y:
                quadrant_weights[0] += weight
            elif center_x >= mid_x and center_y < mid_y:
                quadrant_weights[1] += weight
            elif center_x < mid_x and center_y >= mid_y:
                quadrant_weights[2] += weight
            else:
                quadrant_weights[3] += weight
        
        # Calculate coefficient of variation
        total = sum(quadrant_weights)
        if total == 0:
            return 1.0
        
        avg = total / 4.0
        variance = sum((w - avg) ** 2 for w in quadrant_weights) / 4.0
        std_dev = np.sqrt(variance)
        cv = std_dev / avg if avg > 0 else 0.0
        
        # Load balance: inverse of coefficient of variation
        lb = 1.0 / (1.0 + cv)
        return lb
    
    def auto_repack(self, placed_items, placed_positions, strategy='load_balanced'):
        """
        Automatic repacking selection.
        
        Args:
            placed_items: List of (length, width, height)
            placed_positions: List of (x, y, z)
            strategy: 'blf' (bottom-left-fill), 
                     'load_balanced', 
                     'min_height'
            
        Returns:
            dict: {
                'success': bool,
                'new_positions': list,
                'metric': float,
                'strategy_used': str,
                'description': str
            }
        """
        if strategy == 'blf':
            success, positions, metric = self.attempt_repack_bottom_left_fill(
                placed_items, placed_positions
            )
            return {
                'success': success,
                'new_positions': positions,
                'metric': metric,
                'strategy_used': 'bottom_left_fill',
                'description': 'Packed from bottom-left corner'
            }
        
        elif strategy == 'load_balanced':
            success, positions, metric = self.attempt_repack_load_balanced(
                placed_items, placed_positions
            )
            return {
                'success': success,
                'new_positions': positions,
                'metric': metric,
                'strategy_used': 'load_balanced',
                'description': f'Load balanced (score: {metric:.4f})'
            }
        
        elif strategy == 'min_height':
            success, positions, metric = self.attempt_repack_minimize_height(
                placed_items, placed_positions
            )
            return {
                'success': success,
                'new_positions': positions,
                'metric': metric,
                'strategy_used': 'minimize_height',
                'description': f'Height reduced by {metric:.2f}x'
            }
        
        else:
            # Try all strategies, pick best
            results = []
            
            success1, pos1, metric1 = self.attempt_repack_bottom_left_fill(
                placed_items, placed_positions
            )
            if success1:
                results.append(('blf', pos1, metric1))
            
            success2, pos2, metric2 = self.attempt_repack_load_balanced(
                placed_items, placed_positions
            )
            if success2:
                results.append(('load_balanced', pos2, metric2))
            
            success3, pos3, metric3 = self.attempt_repack_minimize_height(
                placed_items, placed_positions
            )
            if success3:
                results.append(('min_height', pos3, metric3))
            
            if not results:
                return {
                    'success': False,
                    'new_positions': [],
                    'metric': 0.0,
                    'strategy_used': 'none',
                    'description': 'No valid repacking found'
                }
            
            # Select best result (prioritize load balance > height reduction > blf)
            best = max(results, key=lambda r: r[2])
            
            return {
                'success': True,
                'new_positions': best[1],
                'metric': best[2],
                'strategy_used': best[0],
                'description': f'Best strategy: {best[0]} (score: {best[2]:.4f})'
            }


def attempt_repack(env, strategy='load_balanced'):
    """
    Attempt to repack items dalam environment.
    
    Args:
        env: ContainerEnv instance
        strategy: 'blf', 'load_balanced', 'min_height', atau 'auto'
        
    Returns:
        dict: Repacking result dengan success, new_positions, dan metrics
    """
    repacker = Repacker(container_dims=(env.L, env.W, env.H))
    
    result = repacker.auto_repack(env.placed_items, env.placed_positions, strategy)
    
    if result['success']:
        # Update environment with new positions
        env.placed_positions = result['new_positions']
        # Rebuild height map
        env.height_map.reset()
        for (x, y, z), (l, w, h) in zip(result['new_positions'], env.placed_items):
            env.height_map.update_region_absolute(x, y, l, w, z + h)
    
    return result


if __name__ == "__main__":
    """Test repacking functionality"""
    
    print("=" * 70)
    print("REPACKING MECHANISM TEST")
    print("=" * 70)
    
    repacker = Repacker(container_dims=(59, 23, 23))
    
    # Test items
    items = [
        (10, 10, 10),  # volume 1000
        (8, 8, 8),     # volume 512
        (6, 6, 6),     # volume 216
        (5, 5, 5),     # volume 125
    ]
    
    positions = [
        (0, 0, 0),
        (10, 0, 0),
        (20, 0, 0),
        (30, 0, 0),
    ]
    
    print("\n1. Testing Bottom-Left-Fill Repacking")
    print("-" * 70)
    success1, pos1, metric1 = repacker.attempt_repack_bottom_left_fill(items, positions)
    print(f"Success: {success1}")
    print(f"New positions: {pos1}")
    print(f"Metric: {metric1:.4f}")
    
    print("\n2. Testing Load-Balanced Repacking")
    print("-" * 70)
    success2, pos2, metric2 = repacker.attempt_repack_load_balanced(items, positions)
    print(f"Success: {success2}")
    print(f"New positions: {pos2}")
    print(f"Load Balance Score: {metric2:.4f}")
    
    print("\n3. Testing Minimize Height Repacking")
    print("-" * 70)
    success3, pos3, metric3 = repacker.attempt_repack_minimize_height(items, positions)
    print(f"Success: {success3}")
    print(f"New positions: {pos3}")
    print(f"Height Reduction: {metric3:.2f}x")
    
    print("\n4. Testing Auto Repack")
    print("-" * 70)
    result = repacker.auto_repack(items, positions, strategy='auto')
    print(f"Result: {result}")
    
    print("\n" + "=" * 70)
    print("✓ Repacking tests completed!")
    print("=" * 70)