"""
⚠️  DEPRECATED: Repack module has been removed.

This module has been deprecated and replaced by the new Algorithm implementations:
- Use HighLevelSearcher for complete hierarchical search with repacking
- Use RepackTrial directly for deadlock resolution

NEW IMPLEMENTATION: Use src.planning.high_level_search.HighLevelSearcher
For direct repack trials, use src.planning.repack_trial.RepackTrial

MIGRATION GUIDE:
  Old: from src.planning.repack import attempt_repack
  
  New: from src.planning.high_level_search import HighLevelSearcher
       searcher = HighLevelSearcher(env, use_repack=True)
       result = searcher.search(env_state)

For low-level repack operations:
  from src.planning.repack_trial import RepackTrial
  rt = RepackTrial(container_dims=(env.L, env.W, env.H))
  result = rt.attempt_repack(env_state)
"""

import warnings


def attempt_repack(env, strategy='load_balanced'):
    """
    ⚠️  DEPRECATED: This function is no longer available.
    
    This function has been removed as part of code cleanup.
    All repacking functionality is now handled by:
    
    1. HighLevelSearcher - for complete hierarchical search
    2. RepackTrial - for direct deadlock resolution
    
    Please update your code to use one of the new APIs above.
    
    Args:
        env: ContainerEnv instance
        strategy: (ignored)
        
    Raises:
        RuntimeError: Always raises to indicate the function is no longer available
    """
    raise RuntimeError(
        "attempt_repack() has been removed. "
        "Use HighLevelSearcher or RepackTrial instead. "
        "See src/planning/repack.py module docstring for migration guide."
    )


__all__ = ['attempt_repack']
