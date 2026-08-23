"""
AerX Labs — Risk Field Unit Tests
tests/test_risk_field.py
"""

import unittest
import numpy as np

from src.terrain import create_demo_terrain
from src.risk_field import (
    HazardSite,
    compute_hazard_attenuation,
    compute_point_risk,
    compute_risk_slice,
    create_demo_hazards,
    Z_RISK_SCALE,
)


class TestRiskField(unittest.TestCase):

    def setUp(self):
        self.terrain = create_demo_terrain(nx=100, ny=100)
        self.hazards = create_demo_hazards(self.terrain)

    def test_out_of_range_risk_is_zero(self):
        """Points outside hazard radius must have zero risk."""
        hazard = HazardSite(x=10.0, y=10.0, z=200.0, radius=20.0, intensity=1.0)
        far_point = (50.0, 50.0, 200.0)  # distance ~ 56.5 > 20
        risk = compute_point_risk(far_point, [hazard], terrain=self.terrain, apply_los=False)
        self.assertEqual(risk, 0.0)

    def test_attenuation_models(self):
        """Verify quadratic, linear, and gaussian decay formulas."""
        r = 10.0
        # at d = 5 (half radius):
        # quadratic: (1 - 0.5)^2 = 0.25
        self.assertAlmostEqual(compute_hazard_attenuation(5.0, r, intensity=1.0, decay_type="quadratic"), 0.25)
        # linear: 1 - 0.5 = 0.5
        self.assertAlmostEqual(compute_hazard_attenuation(5.0, r, intensity=1.0, decay_type="linear"), 0.5)
        # gaussian: exp(-3 * 0.25) = exp(-0.75) ~ 0.472366
        self.assertAlmostEqual(compute_hazard_attenuation(5.0, r, intensity=1.0, decay_type="gaussian"), np.exp(-0.75), places=5)

    def test_point_risk_in_range_clear_los(self):
        """Points within range and high altitude (clear LOS) must receive expected risk.
        The vertical separation is de-weighted by Z_RISK_SCALE in the range gate, so
        the effective distance is 15*Z_RISK_SCALE."""
        hazard = HazardSite(x=50.0, y=50.0, z=150.0, radius=30.0, intensity=1.0, decay_type="linear")
        point = (50.0, 50.0, 165.0)  # dz = 15 -> effective 15*Z_RISK_SCALE
        eff_dist = 15.0 * Z_RISK_SCALE
        expected = 1.0 - eff_dist / 30.0  # linear decay at the effective distance
        risk = compute_point_risk(point, [hazard], terrain=self.terrain, apply_los=True)
        self.assertAlmostEqual(risk, expected, places=2)

    def test_terrain_masking_blocks_risk(self):
        """Points geometrically behind a mountain peak must have 0 risk when apply_los=True."""
        # Mountain 1 is centered at x~32.5, y~42.5 with height ~ 280m
        # Hazard at x=10, y=42.5, z=120m
        # Target point at x=80, y=42.5, z=120m (behind the mountain)
        hazard = HazardSite(x=10.0, y=42.5, z=120.0, radius=100.0, intensity=1.0, requires_los=True)
        target = (80.0, 42.5, 120.0)

        # Without LOS: should have risk due to proximity within radius 100
        risk_no_los = compute_point_risk(target, [hazard], terrain=self.terrain, apply_los=False)
        self.assertGreater(risk_no_los, 0.0)

        # With LOS: mountain blocks ray, risk should be masked to 0.0
        risk_with_los = compute_point_risk(target, [hazard], terrain=self.terrain, apply_los=True)
        self.assertEqual(risk_with_los, 0.0)

    def test_probabilistic_hazard_aggregation(self):
        """Multi-hazard risk combination must obey 1 - (1-R1)(1-R2) and remain in [0, 1]."""
        h1 = HazardSite(x=50.0, y=50.0, z=200.0, radius=50.0, intensity=0.6, decay_type="linear")
        h2 = HazardSite(x=50.0, y=50.0, z=200.0, radius=50.0, intensity=0.5, decay_type="linear")
        point = (50.0, 50.0, 200.0)  # at origin: R1 = 0.6, R2 = 0.5

        # Expected: 1 - (1 - 0.6)*(1 - 0.5) = 1 - (0.4 * 0.5) = 1 - 0.2 = 0.8
        combined_risk = compute_point_risk(point, [h1, h2], terrain=self.terrain, apply_los=False)
        self.assertAlmostEqual(combined_risk, 0.8, places=5)
        self.assertTrue(0.0 <= combined_risk <= 1.0)

    def test_risk_slice_shape(self):
        """Test compute_risk_slice produces correct 2D shape."""
        slice_2d = compute_risk_slice(self.terrain, self.hazards, altitude_asl=250.0, subsample_step=5)
        expected_shape = (20, 20)
        self.assertEqual(slice_2d.shape, expected_shape)
        self.assertTrue(np.all((slice_2d >= 0.0) & (slice_2d <= 1.0)))


if __name__ == "__main__":
    unittest.main()
