"""
High-Level Agent Module - Backward Compatibility Layer

This module re-exports HighLevelAgent from its new location.
New code should import directly from:
  - models.high_level_agent.HighLevelAgent

Legacy imports from this module are still supported:
  from rl.high_level_agent import HighLevelAgent
"""

from models.high_level_agent import HighLevelAgent

__all__ = ['HighLevelAgent']
