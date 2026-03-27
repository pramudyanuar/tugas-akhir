import unittest

from src.core.container_env import ContainerEnv
from src.planning.mcts import MCTS


class TestMCTSRearrangement(unittest.TestCase):
    def setUp(self):
        self.env = ContainerEnv(max_items=8, seed=42)
        self.env.reset(seed=42)

        # Build a deterministic placed state to allow unpack/repack simulation.
        self.env.placed_items = [
            (10, 5, 4),
            (8, 4, 3),
            (6, 3, 3),
        ]
        self.env.placed_positions = [
            (0, 0, 0),
            (10, 0, 0),
            (18, 0, 0),
        ]

        self.env.height_map.reset()
        for (x, y, z), (l, w, h) in zip(self.env.placed_positions, self.env.placed_items):
            self.env.height_map.update_region_absolute(x, y, l, w, z + h)

        self.mcts = MCTS(self.env, budget=8, c=1.4, gamma=0.99)

    def test_search_rearrangement_returns_expected_keys(self):
        failed_item = (5, 5, 3)
        result = self.mcts.search_rearrangement(
            failed_item=failed_item,
            max_unpack=2,
            apply_to_env=False,
        )

        self.assertIn('success', result)
        self.assertIn('best_sequence', result)
        self.assertIn('best_value', result)
        self.assertIn('tree_stats', result)
        self.assertIn('applied', result)

        self.assertIsInstance(result['best_sequence'], list)
        self.assertGreaterEqual(result['best_value'], -1.0)
        self.assertLessEqual(result['best_value'], 1.0)

    def test_search_rearrangement_apply_flag_updates_env_safely(self):
        failed_item = (4, 4, 2)
        result = self.mcts.search_rearrangement(
            failed_item=failed_item,
            max_unpack=3,
            apply_to_env=True,
        )

        # Applied state must still be physically bounded.
        for (x, y, z), (l, w, h) in zip(self.env.placed_positions, self.env.placed_items):
            self.assertLessEqual(x + l, self.env.L)
            self.assertLessEqual(y + w, self.env.W)
            self.assertLessEqual(z + h, self.env.H)

        self.assertIn('applied', result)


if __name__ == '__main__':
    unittest.main()
