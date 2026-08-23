"""
AerX Labs — Route Metrics & Performance Evaluation
src/route_metrics.py

Computes quantitative mission metrics for 3D flight trajectories:
- Total 3D distance and 2D horizontal distance
- Estimated flight time (considering climb/descent speed adjustments)
- Energy / fuel consumption proxy
- Cumulative integrated spatial risk and peak risk exposure
- Minimum, maximum, and average Above Ground Level (AGL) clearance
- Maximum climb and descent gradients
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from src.los import terrain_height
from src.risk_field import HazardSite, compute_point_risk


@dataclass
class RouteMetrics:
    """Quantitative performance and risk metrics for an evaluated flight route."""
    total_distance: float              # Total 3D distance (m)
    horizontal_distance: float         # 2D ground projection distance (m)
    flight_time: float                 # Estimated flight duration (s)
    energy_proxy: float                # Normalized energy/fuel consumption proxy
    integrated_risk: float             # Total cumulative risk exposure along trajectory
    peak_risk: float                   # Highest single instantaneous risk encountered
    min_agl: float                     # Lowest terrain clearance (m)
    max_agl: float                     # Highest terrain clearance (m)
    mean_agl: float                    # Average terrain clearance (m)
    max_climb_angle_deg: float         # Steepest ascent angle (degrees)
    max_descent_angle_deg: float       # Steepest descent angle (degrees)
    waypoint_count: int                # Number of waypoints in trajectory


def compute_route_metrics(
    waypoints: np.ndarray,
    terrain: np.ndarray,
    hazards: Optional[List[HazardSite]] = None,
    nominal_speed: float = 30.0,         # Nominal flight speed (m/s)
    climb_speed: float = 15.0,           # Ascent speed (m/s)
    descent_speed: float = 25.0,         # Descent speed (m/s)
    base_power: float = 1.0,             # Base cruise power consumption rate
    climb_power_factor: float = 2.5,     # Power multiplier during climb
    xy_scale: float = 1.0,
    rcs_aniso: bool = False,             # apply heading-dependent anisotropic RCS
    los_samples: int = 30                # LOS ray samples per risk query (lower = faster)
) -> RouteMetrics:
    """
    Compute comprehensive flight performance and risk metrics for a 3D path.

    Parameters:
        waypoints: (N, 3) array of 3D positions [[x, y, z], ...]
        terrain: 2D elevation grid
        hazards: List of HazardSite instances
        nominal_speed: Cruise speed in m/s
        climb_speed: Speed during climb in m/s
        descent_speed: Speed during descent in m/s
        base_power: Normalized cruise power proxy
        climb_power_factor: Normalized climb power multiplier
        xy_scale: Distance scaling factor

    Returns:
        RouteMetrics dataclass instance.
    """
    waypoints = np.asarray(waypoints, dtype=float)
    N = len(waypoints)
    if N < 2:
        raise ValueError("Route must contain at least 2 waypoints to compute metrics.")

    hazards = hazards or []

    total_dist = 0.0
    horiz_dist = 0.0
    total_time = 0.0
    total_energy = 0.0
    integrated_risk = 0.0
    peak_risk = 0.0

    agl_list = []
    climb_angles = []
    descent_angles = []

    # Per-waypoint flight heading (for optional anisotropic RCS).
    def _heading(i):
        if not rcs_aniso:
            return None
        j = min(i + 1, N - 1)
        k = max(i - 1, 0)
        return (waypoints[j][:2] - waypoints[k][:2])

    # Per-waypoint risk computed ONCE (the risk field is the expensive part), then
    # reused for both the peak (max) and the integrated (trapezoidal) exposure —
    # ~3x fewer LOS-sampled evaluations than sampling per waypoint and per segment.
    if hazards:
        wp_risk = [
            compute_point_risk(tuple(waypoints[i]), hazards, terrain=terrain,
                               apply_los=True, los_samples=los_samples, heading=_heading(i))
            for i in range(N)
        ]
        peak_risk = max(wp_risk)
    else:
        wp_risk = None

    for i in range(N):
        x, y, z = waypoints[i]
        ground = terrain_height(terrain, x / xy_scale, y / xy_scale)
        agl_list.append(z - ground)

    for i in range(N - 1):
        p1 = waypoints[i]
        p2 = waypoints[i + 1]

        seg_vec = p2 - p1
        seg_dist = float(np.linalg.norm(seg_vec))
        seg_horiz = float(np.hypot(seg_vec[0], seg_vec[1]))
        dz = float(seg_vec[2])

        total_dist += seg_dist
        horiz_dist += seg_horiz

        # Gradient / angle calculation
        if seg_horiz > 1e-6:
            slope_angle = np.degrees(np.arctan2(dz, seg_horiz))
            if dz > 0:
                climb_angles.append(slope_angle)
            elif dz < 0:
                descent_angles.append(abs(slope_angle))

        # Travel time calculation with vertical rate adjustments
        if dz > 0:
            v_effective = climb_speed
            p_effective = base_power * climb_power_factor
        elif dz < 0:
            v_effective = descent_speed
            p_effective = base_power * 0.8  # Slight power reduction during descent
        else:
            v_effective = nominal_speed
            p_effective = base_power

        t_seg = seg_dist / v_effective
        total_time += t_seg
        total_energy += p_effective * t_seg

        # Segment integrated risk (trapezoidal, reusing the per-waypoint risk).
        if hazards:
            integrated_risk += 0.5 * (wp_risk[i] + wp_risk[i + 1]) * seg_dist

    max_climb = float(max(climb_angles)) if climb_angles else 0.0
    max_descent = float(max(descent_angles)) if descent_angles else 0.0

    return RouteMetrics(
        total_distance=float(total_dist),
        horizontal_distance=float(horiz_dist),
        flight_time=float(total_time),
        energy_proxy=float(total_energy),
        integrated_risk=float(integrated_risk),
        peak_risk=float(peak_risk),
        min_agl=float(min(agl_list)),
        max_agl=float(max(agl_list)),
        mean_agl=float(np.mean(agl_list)),
        max_climb_angle_deg=max_climb,
        max_descent_angle_deg=max_descent,
        waypoint_count=N
    )
