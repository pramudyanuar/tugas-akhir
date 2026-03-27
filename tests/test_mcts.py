"""Test cases untuk MCTS module."""

import pytest
import numpy as np
from src.common.mcts_node import MCTSNode
from src.learning.agents.mcts import MCTS
from src.core.container_env import ContainerEnv


class TestMCTSNode:
    """Test suite for MCTSNode class."""

    @pytest.fixture
    def env(self):
        """Create environment for testing."""
        return ContainerEnv(seed=42)

    def test_node_initialization(self, env):
        """Test MCTSNode initialization."""
        state, _ = env.reset()
        
        node = MCTSNode(state, untried_actions=[0, 1, 2, 3])
        
        assert node.state is not None, "Node should have state"
        assert node.untried_actions == [0, 1, 2, 3], "Untried actions not set correctly"
        assert node.visits == 0, "Initial visits should be 0"
        assert node.value == 0.0, "Initial value should be 0.0"

    def test_node_parent_child_relationship(self, env):
        """Test parent-child relationship."""
        state, _ = env.reset()
        
        parent = MCTSNode(state, untried_actions=[0, 1, 2])
        child = MCTSNode(state, parent=parent, action=0)
        
        assert child.parent == parent, "Parent not set correctly"
        assert child.action == 0, "Action not set correctly"

    def test_ucb_value_calculation(self, env):
        """Test UCB value calculation."""
        state, _ = env.reset()
        
        root = MCTSNode(state, untried_actions=[0, 1, 2])
        root.visits = 10
        root.value = 50.0
        
        child1 = MCTSNode(state, parent=root, action=0)
        child1.visits = 3
        child1.value = 20.0
        
        child2 = MCTSNode(state, parent=root, action=1)
        child2.visits = 7
        child2.value = 30.0
        
        ucb1 = child1.ucb_value(c=1.4)
        ucb2 = child2.ucb_value(c=1.4)
        
        assert isinstance(ucb1, (int, float)), "UCB value should be numeric"
        assert isinstance(ucb2, (int, float)), "UCB value should be numeric"
        # UCB values should be different for different nodes
        assert ucb1 != ucb2, "Different nodes should have different UCB values"

    def test_backup_propagation(self, env):
        """Test backup propagation."""
        state, _ = env.reset()
        
        root = MCTSNode(state, untried_actions=[0, 1])
        child = MCTSNode(state, parent=root, action=0)
        
        # Initial state
        assert root.visits == 0, "Initial root visits should be 0"
        assert child.visits == 0, "Initial child visits should be 0"
        
        # Backup
        child.backup(10.0)
        
        # After backup
        assert child.visits == 1, "Child visits should be 1 after backup"
        assert child.value == 10.0, "Child value should be 10.0"
        assert root.visits == 1, "Root should be updated via backup"
        assert root.value == 10.0, "Root value should be updated via backup"

    def test_node_selection(self, env):
        """Test selecting best child node."""
        state, _ = env.reset()
        
        parent = MCTSNode(state, untried_actions=[0, 1, 2])
        
        child1 = MCTSNode(state, parent=parent, action=0)
        child1.visits = 10
        child1.value = 100.0
        
        child2 = MCTSNode(state, parent=parent, action=1)
        child2.visits = 5
        child2.value = 50.0
        
        parent.children[0] = child1
        parent.children[1] = child2
        
        # Best child should be based on UCB or visit count
        assert parent.children[0] is not None, "Child 1 should exist"
        assert parent.children[1] is not None, "Child 2 should exist"


class TestMCTS:
    """Test suite for MCTS algorithm."""

    @pytest.fixture
    def env(self):
        """Create environment for testing."""
        return ContainerEnv(seed=42)

    @pytest.fixture
    def mcts(self, env):
        """Create MCTS instance for testing."""
        return MCTS(env, budget=10)

    def test_mcts_initialization(self, mcts):
        """Test MCTS initialization."""
        assert mcts is not None, "MCTS should be initialized"
        assert mcts.budget == 10, "Budget should be set"
        assert mcts.c == 1.4, "Exploration constant should be set"
        assert mcts.gamma == 0.99, "Discount factor should be set"

    def test_mcts_search(self, mcts, env):
        """Test MCTS search."""
        state, action_mask = env.reset()
        
        # Run MCTS search - returns dict with search results
        result = mcts.search(state, action_mask)
        
        # Check that result is a dictionary with expected keys
        assert isinstance(result, dict), "Result should be a dictionary"
        assert 'best_action' in result, "Result should contain best_action"
        best_action = result['best_action']
        assert isinstance(best_action, (int, np.integer)), "Best action should be integer"
        assert best_action >= 0, "Action index should be non-negative"
