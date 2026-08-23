"""
AerX Labs — Route Metrics, Scoring, and Monte Carlo Unit Tests
tests/test_evaluation.py
"""

import unittest
import numpy as np

from src.risk_field import HazardSite
from src.route_metrics import compute_route_metrics
from src.scoring import compute_route_score
from src.monte_carlo import run_monte_carlo_evaluation


class TestEvaluationPipeline(unittest.TestCase):

    def setUp(self):
        self.terrain = np.full((50, 50), 100.0)
        self.hazards = [
            HazardSite(x=25.0, y=25.0, z=140.0, radius=20.0, intensity=1.0, requires_los=False)
        ]
        # Straight-line waypoints passing through the hazard
        self.straight_waypoints = np.array([
            [5.0, 25.0, 140.0],
            [15.0, 25.0, 140.0],
            [25.0, 25.0, 140.0],
            [35.0, 25.0, 140.0],
            [45.0, 25.0, 140.0]
        ])

        # Detour waypoints avoiding the hazard
        self.detour_waypoints = np.array([
            [5.0, 25.0, 140.0],
            [15.0, 5.0, 140.0],
            [25.0, 5.0, 140.0],
            [35.0, 5.0, 140.0],
            [45.0, 25.0, 140.0]
        ])

    def test_metrics_calculation(self):
        """Test calculation of distance, flight time, and integrated risk."""
        m_straight = compute_route_metrics(self.straight_waypoints, self.terrain, self.hazards)
        m_detour = compute_route_metrics(self.detour_waypoints, self.terrain, self.hazards)

        self.assertAlmostEqual(m_straight.total_distance, 40.0)
        self.assertGreater(m_detour.total_distance, 40.0)
        self.assertGreater(m_straight.integrated_risk, m_detour.integrated_risk)
        self.assertEqual(m_detour.integrated_risk, 0.0)

    def test_scoring_comparison(self):
        """Test that detour route achieves higher survivability score than straight through hazard."""
        m_straight = compute_route_metrics(self.straight_waypoints, self.terrain, self.hazards)
        m_detour = compute_route_metrics(self.detour_waypoints, self.terrain, self.hazards)

        score_straight = compute_route_score(m_straight, direct_euclidean_distance=40.0)
        score_detour = compute_route_score(m_detour, direct_euclidean_distance=40.0)

        # Straight route has low survivability due to hazard exposure
        self.assertLess(score_straight.survivability_score, 1.0)
        # Detour route has perfect survivability
        self.assertEqual(score_detour.survivability_score, 1.0)

    def test_monte_carlo_distribution(self):
        """Test Monte Carlo statistical distribution generation."""
        mc_res = run_monte_carlo_evaluation(
            waypoints=self.straight_waypoints,
            terrain=self.terrain,
            hazards=self.hazards,
            direct_euclidean_distance=40.0,
            num_trials=20
        )
        self.assertEqual(mc_res.num_trials, 20)
        self.assertTrue(0.0 <= mc_res.mean_composite_score <= 100.0)
        self.assertGreaterEqual(mc_res.p95_best_case_score, mc_res.p5_worst_case_score)


if __name__ == "__main__":
    unittest.main()
