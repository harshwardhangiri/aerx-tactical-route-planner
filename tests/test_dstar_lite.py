"""
AerX Labs — D* Lite 3D Unit Tests
tests/test_dstar_lite.py
"""

import unittest
import numpy as np

from src.risk_field import HazardSite
from src.planning_grid import PlanningGrid3D, CostWeights
from src.dstar_lite import DStarLite3D


class TestDStarLite3D(unittest.TestCase):

    def setUp(self):
        # 30x30 test grid
        self.terrain = np.full((30, 30), 100.0)
        # Add a blocking hill in the middle (x: 10-20, y: 10-20)
        self.terrain[10:20, 10:20] = 200.0

        self.z_layers = np.array([120.0, 140.0, 160.0, 180.0, 220.0, 250.0])
        self.grid = PlanningGrid3D(
            terrain=self.terrain,
            z_layers=self.z_layers,
            min_agl=15.0,
            max_agl=200.0,
            max_climb_slope=1.0,
            xy_scale=1.0,
            weights=CostWeights.distance_focused()
        )
        self.planner = DStarLite3D(self.grid)

    def test_basic_path_finding(self):
        """Test finding a valid 3D route around an obstacle."""
        start = (5.0, 15.0, 140.0)
        goal = (25.0, 15.0, 140.0)

        path = self.planner.plan(start, goal)
        self.assertIsNotNone(path)
        self.assertGreater(len(path), 0)

        coords = self.planner.get_path_coordinates(path)
        self.assertEqual(coords.shape, (len(path), 3))
        # Start and goal nodes match
        self.assertEqual(path[0], self.grid.pos_to_node(start))
        self.assertEqual(path[-1], self.grid.pos_to_node(goal))

        # Check all waypoints on path are feasible
        for node in path:
            self.assertTrue(self.grid.is_feasible(node))

    def test_risk_aware_deviation(self):
        """Test that risk-focused weights guide the route away from high-intensity hazards."""
        # Flat terrain with a hazard in direct line of path
        flat_terrain = np.full((30, 30), 100.0)
        hazard = HazardSite(x=15.0, y=15.0, z=140.0, radius=8.0, intensity=1.0, requires_los=False)

        grid_dist = PlanningGrid3D(
            terrain=flat_terrain,
            z_layers=np.array([140.0, 180.0]),
            min_agl=10.0,
            hazards=[hazard],
            weights=CostWeights.distance_focused()
        )
        planner_dist = DStarLite3D(grid_dist)
        path_dist = planner_dist.plan((5.0, 15.0, 140.0), (25.0, 15.0, 140.0))

        grid_risk = PlanningGrid3D(
            terrain=flat_terrain,
            z_layers=np.array([140.0, 180.0]),
            min_agl=10.0,
            hazards=[hazard],
            weights=CostWeights.risk_focused()
        )
        planner_risk = DStarLite3D(grid_risk)
        path_risk = planner_risk.plan((5.0, 15.0, 140.0), (25.0, 15.0, 140.0))

        self.assertIsNotNone(path_dist)
        self.assertIsNotNone(path_risk)

        # Distance path goes straight through y=15
        y_dist = [node[1] for node in path_dist]
        self.assertTrue(all(y == 15 for y in y_dist))

        # Risk-aware path deviates away from hazard center (y != 15)
        y_risk = [node[1] for node in path_risk]
        self.assertTrue(any(y != 15 for y in y_risk))


if __name__ == "__main__":
    unittest.main()
