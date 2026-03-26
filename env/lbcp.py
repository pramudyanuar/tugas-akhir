"""
LBCP Module - Backward Compatibility Layer

Imports and re-exports LBCP classes from their new locations:
- StabilityValidator from env.stability_validator
- LBCPClusterer from env.lbcp_clusterer

New code should import directly:
  from env.stability_validator import StabilityValidator
  from env.lbcp_clusterer import LBCPClusterer

Legacy imports from this module still work:
  from env.lbcp import StabilityValidator, LBCPClusterer, is_stable
"""

import numpy as np
from .stability_validator import StabilityValidator
from .lbcp_clusterer import LBCPClusterer

__all__ = ['StabilityValidator', 'LBCPClusterer', 'is_stable', 
           'compute_support_cells', 'compute_convex_hull', 'is_cog_inside_polygon']


# Backward compatibility: module-level functions that delegate to StabilityValidator
def compute_support_cells(height_map, x, y, l, w, base_height):
    """Backward compatibility wrapper for StabilityValidator.compute_support_cells"""
    return StabilityValidator.compute_support_cells(height_map, x, y, l, w, base_height)

def compute_convex_hull(support_cells):
    """Backward compatibility wrapper for StabilityValidator.compute_convex_hull"""
    return StabilityValidator.compute_convex_hull(support_cells)

def is_cog_inside_polygon(hull_points, cog):
    """Backward compatibility wrapper for StabilityValidator.is_cog_inside_polygon"""
    return StabilityValidator.is_cog_inside_polygon(hull_points, cog)

def is_stable(height_map, x, y, l, w, h, max_height):
    """Backward compatibility wrapper for StabilityValidator.is_stable"""
    return StabilityValidator.is_stable(height_map, x, y, l, w, h, max_height)


if __name__ == "__main__":
    """Test cases untuk LBCP Stability"""
    
    # Test Case 1: Flat floor → stable
    print("=" * 60)
    print("Test Case 1: Flat floor → stable")
    print("=" * 60)
    height_map_flat = np.zeros((10, 10), dtype=np.int32)
    # Flat floor di z=0, item 3x3 dengan height 2
    result_flat = StabilityValidator.is_stable(height_map_flat, 2, 2, 3, 3, 2, 10)
    print(f"Flat floor (3x3 item on z=0): {result_flat}")
    print(f"Expected: True")
    assert result_flat == True, "Flat floor test failed!"
    print("✓ PASSED\n")

    # Test Case 2: Overhang kecil → unstable
    print("=" * 60)
    print("Test Case 2: Overhang kecil → unstable")
    print("=" * 60)
    height_map_overhang = np.zeros((15, 15), dtype=np.int32)
    # Create strong overhang: support cells hanya di satu sisi saja
    # Region akan menjadi 5x5, tapi support hanya di baris pertama
    height_map_overhang[3, 3:8] = 5    # Support hanya di 1 baris (y=3:8)
    height_map_overhang[4:8, 3:8] = 0  # Tidak ada support di baris lainnya
    result_overhang = StabilityValidator.is_stable(height_map_overhang, 3, 3, 5, 5, 2, 10)
    print(f"Overhang kecil (5x5 item with single row support): {result_overhang}")
    print(f"Expected: False (CoG outside support polygon/collinear)")
    assert result_overhang == False, "Overhang test failed!"
    print("✓ PASSED\n")

    # Test Case 3: Single support cell → unstable
    print("=" * 60)
    print("Test Case 3: Single support cell → unstable")
    print("=" * 60)
    height_map_single = np.zeros((10, 10), dtype=np.int32)
    # Minimal support cells (< 3)
    height_map_single[3, 3] = 5  # Only 1 support cell
    result_single = StabilityValidator.is_stable(height_map_single, 2, 2, 3, 3, 2, 10)
    print(f"Single support cell: {result_single}")
    print(f"Expected: False (need minimal 3 support cells)")
    assert result_single == False, "Single support cell test failed!"
    print("✓ PASSED\n")

    print("=" * 60)
    print("All LBCP Stability tests completed!")
    print("=" * 60)
