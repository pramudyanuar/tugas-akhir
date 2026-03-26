"""
MCTS Module - Backward Compatibility Layer

This module re-exports MCTS classes from their new locations for backward compatibility.
New code should import directly from:
  - common.mcts_node.MCTSNode
  - agents.mcts.MCTS

Legacy imports from this module are still supported:
  from planning.mcts import MCTSNode, MCTS
"""

from common.mcts_node import MCTSNode
from agents.mcts import MCTS

__all__ = ['MCTSNode', 'MCTS']


if __name__ == "__main__":
    """Test cases untuk MCTS"""
    
    print("=" * 70)
    print("Test Case 1: MCTSNode initialization")
    print("=" * 70)
    
    from env.container_env import ContainerEnv
    
    env = ContainerEnv(seed=42)
    state, action_mask = env.reset()
    
    node = MCTSNode(state, untried_actions=[0, 1, 2, 3])
    
    print(f"Node created with state shape: {node.state.shape}")
    print(f"Untried actions: {node.untried_actions}")
    print(f"Visits: {node.visits}")
    print(f"Value: {node.value}")
    print("✓ PASSED\n")
    
    print("=" * 70)
    print("Test Case 2: UCB value calculation")
    print("=" * 70)
    
    root = MCTSNode(state, untried_actions=[0, 1, 2])
    root.visits = 10
    root.value = 50.0
    
    child1 = MCTSNode(state, parent=root, action=0)
    child1.visits = 3
    child1.value = 20.0
    
    child2 = MCTSNode(state, parent=root, action=1)
    child2.visits = 7
    child2.value = 30.0
    
    root.children[0] = child1
    root.children[1] = child2
    
    ucb1 = child1.ucb_value(c=1.4)
    ucb2 = child2.ucb_value(c=1.4)
    
    print(f"Child1 UCB: {ucb1:.4f} (visits=3, value=20)")
    print(f"Child2 UCB: {ucb2:.4f} (visits=7, value=30)")
    print(f"UCB values calculated successfully")
    print("✓ PASSED\n")
    
    print("=" * 70)
    print("Test Case 3: Backup propagation")
    print("=" * 70)
    
    root = MCTSNode(state, untried_actions=[0, 1])
    child = MCTSNode(state, parent=root, action=0)
    root.children[0] = child
    
    child.backup(10.0)
    
    print(f"Root visits after backup: {root.visits}")
    print(f"Root value after backup: {root.value}")
    print(f"Child visits after backup: {child.visits}")
    print(f"Child value after backup: {child.value}")
    assert root.visits == 1, "Root visits should be 1"
    assert root.value == 10.0, "Root value should be 10.0"
    print("✓ PASSED\n")
    
    print("=" * 70)
    print("Test Case 4: MCTS initialization")
    print("=" * 70)
    
    env = ContainerEnv(seed=123)
    mcts = MCTS(env, budget=50, c=1.4, gamma=0.99)
    
    print(f"MCTS created with budget: {mcts.budget}")
    print(f"Exploration constant c: {mcts.c}")
    print(f"Discount factor gamma: {mcts.gamma}")
    print("✓ PASSED\n")
    
    print("=" * 70)
    print("All MCTS tests completed!")
    print("=" * 70)
