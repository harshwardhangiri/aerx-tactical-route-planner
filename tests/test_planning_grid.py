"""
AerX Labs — Planning Grid Unit Tests
tests/test_planning_grid.py
"""

import unittest
import numpy as np

from src.planning_grid import PlanningGrid3D, CostWeights


class TestPlanningGrid3D(unittest.TestCase):

    def setUp(self):
        # 30x30 small terrain for ultra-fast unit testing
        self.terrain = np.zeros((30, 30), dtype=float)
        # Create a small hill in the center (peak ~150m, base ~100m)
        self.terrain += 100.0
        self.terrain[10:20, 10:20] += 50.0

        self.z_layers = np.array([110.0, 130.0, 160.0, 180.0, 220.0])
        self.grid = PlanningGrid3D(
            terrain=self.terrain,
            z_layers=self.z_layers,
            min_agl=15.0,
            max_agl=150.0,
            max_climb_slope=1.0,
            xy_scale=1.0
        )

    def test_node_feasibility(self):
        """Test AGL clearance constraints."""
        # Flat area: ground = 100m
        # z=110m -> AGL = 10m < min_agl (15m) -> infeasible
        self.assertFalse(self.grid.is_feasible((5, 5, 0)))

        # z=130m -> AGL = 30m >= 15m -> feasible
        self.assertTrue(self.grid.is_feasible((5, 5, 1)))

        # On the hill: ground = 150m
        # z=130m (below ground) -> infeasible
        self.assertFalse(self.grid.is_feasible((15, 15, 1)))
        # z=180m (AGL=30m) -> feasible
        self.assertTrue(self.grid.is_feasible((15, 15, 3)))

    def test_pos_node_conversions(self):
        """Test coordinate transformation consistency."""
        node = (10, 15, 2)
        pos = self.grid.node_to_pos(node)
        self.assertEqual(pos[0], 10.0)
        self.assertEqual(pos[1], 15.0)
        self.assertEqual(pos[2], self.z_layers[2])

        recovered_node = self.grid.pos_to_node(pos)
        self.assertEqual(recovered_node, node)

    def test_edge_cost_and_weights(self):
        """Test cost weighting under distance vs risk configurations."""
        u = (5, 5, 1)  # z = 130m
        v = (6, 5, 1)  # z = 130m (horizontal step, dist = 1.0)

        # Distance focused
        self.grid.weights = CostWeights.distance_focused()
        cost_dist = self.grid.compute_edge_cost(u, v)
        self.assertGreater(cost_dist, 0.0)
        self.assertLess(cost_dist, float("inf"))

        # Infeasible node edge must be inf
        infeasible_node = (5, 5, 0)
        cost_inf = self.grid.compute_edge_cost(u, infeasible_node)
        self.assertEqual(cost_inf, float("inf"))

    def test_neighbor_generation(self):
        """Test neighbor discovery and slope pruning."""
        node = (5, 5, 2)  # z = 160m
        neighbors = self.grid.get_neighbors(node)
        self.assertGreater(len(neighbors), 0)
        for neighbor, cost in neighbors:
            self.assertTrue(self.grid.is_feasible(neighbor))
            self.assertLess(cost, float("inf"))


if __name__ == "__main__":
    unittest.main()
