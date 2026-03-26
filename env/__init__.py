"""Environment Module for 3D Bin Packing

Main components:
- ContainerEnv: Main environment class
- HeightMap: Height map management
- CandidateGenerator: Action candidate generation
- StabilityValidator: Structural stability checking
- LBCPClusterer: Load-balanced clustering
"""

from .container_env import ContainerEnv
from .height_map import HeightMap
from .candidate_generator import CandidateGenerator
from .stability_validator import StabilityValidator
from .lbcp_clusterer import LBCPClusterer
from .lbcp import is_stable  # For backward compatibility

__all__ = [
    'ContainerEnv',
    'HeightMap',
    'CandidateGenerator',
    'StabilityValidator',
    'LBCPClusterer',
    'is_stable',
]
