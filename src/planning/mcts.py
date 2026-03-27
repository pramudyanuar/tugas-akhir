"""
MCTS Module - Backward Compatibility Layer

This module re-exports MCTS classes from their new locations for backward compatibility.
New code should import directly from:
  - common.mcts_node.MCTSNode
  - agents.mcts.MCTS

Legacy imports from this module are still supported:
  from src.planning.mcts import MCTSNode, MCTS
"""

from src.common.mcts_node import MCTSNode
from src.learning.agents.mcts import MCTS

__all__ = ['MCTSNode', 'MCTS']

