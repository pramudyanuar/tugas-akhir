"""
LBCP Module - Backward Compatibility Layer

Imports and re-exports LBCP classes from their new locations:
- StabilityValidator from src.core.stability_validator
- LBCPClusterer from src.core.lbcp_clusterer

New code should import directly:
  from src.core.stability_validator import StabilityValidator
  from src.core.lbcp_clusterer import LBCPClusterer

Legacy imports from this module still work:
  from src.core.lbcp import StabilityValidator, LBCPClusterer, is_stable
"""

from .stability_validator import StabilityValidator
from .lbcp_clusterer import LBCPClusterer

__all__ = [
    "StabilityValidator",
    "LBCPClusterer",
    "is_stable",
    "compute_support_cells",
    "compute_convex_hull",
    "is_cog_inside_polygon",
    "validate_structural_stability",
    "update_feasibility_map",
]


# Backward compatibility: module-level functions that delegate to StabilityValidator

def compute_support_cells(height_map, x, y, l, w, base_height):
    """Backward compatibility wrapper for StabilityValidator.compute_support_cells."""
    return StabilityValidator.compute_support_cells(height_map, x, y, l, w, base_height)


def compute_convex_hull(support_cells):
    """Backward compatibility wrapper for StabilityValidator.compute_convex_hull."""
    return StabilityValidator.compute_convex_hull(support_cells)


def is_cog_inside_polygon(hull_points, cog):
    """Backward compatibility wrapper for StabilityValidator.is_cog_inside_polygon."""
    return StabilityValidator.is_cog_inside_polygon(hull_points, cog)


def is_stable(height_map, x, y, l, w, h, max_height, strict_mode=True):
    """Backward compatibility wrapper for StabilityValidator.is_stable."""
    return StabilityValidator.is_stable(
        height_map,
        x,
        y,
        l,
        w,
        h,
        max_height,
        strict_mode=strict_mode,
    )


def validate_structural_stability(
    new_object,
    load_configuration,
    height_map_t,
    feasibility_map_t,
    cog_tolerance,
):
    """Backward compatibility wrapper for StabilityValidator.validate."""
    return StabilityValidator.validate(
        new_object,
        load_configuration,
        height_map_t,
        feasibility_map_t,
        cog_tolerance,
    )


def update_feasibility_map(feasibility_map_t, updated_support_polygon):
    """Backward compatibility wrapper for StabilityValidator.update_feasibility_map."""
    return StabilityValidator.update_feasibility_map(feasibility_map_t, updated_support_polygon)
