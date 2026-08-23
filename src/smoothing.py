"""
AerX Labs — Trajectory Smoothing & Local Refinement
src/smoothing.py

Refines discrete grid waypoints into continuous, flyable flight trajectories:
- Risk-aware greedy line-of-sight waypoint pruning (removes stair-steps WITHOUT
  shortcutting through threat zones the global planner deliberately avoided)
- Cubic / B-spline interpolation
- Terrain collision & minimum AGL clearance validation
- Post-smoothing risk guard so the spline never pushes into higher exposure

Design note:
    The earlier implementation pruned purely on terrain line-of-sight. On open
    terrain that straightened every risk-avoiding detour back into a direct line
    through the hazards, collapsing distinct route profiles into one identical
    path. Pruning here additionally samples the spatial risk field and refuses
    any shortcut whose peak exposure exceeds what the spanned waypoints already
    incurred (plus a small tolerance).
"""

from typing import List, Optional
import numpy as np
from scipy.interpolate import splprep, splev

from src.los import check_los, terrain_height
from src.risk_field import HazardSite, compute_point_risk


def _segment_peak_risk(
    p1: np.ndarray,
    p2: np.ndarray,
    terrain: np.ndarray,
    hazards: List[HazardSite],
    samples: int = 16,
    risk_sampler=None,
) -> float:
    """Highest instantaneous spatial risk sampled along the straight segment p1->p2."""
    if not hazards and risk_sampler is None:
        return 0.0
    ts = np.linspace(0.0, 1.0, samples)
    peak = 0.0
    for t in ts:
        pt = p1 + t * (p2 - p1)
        r = _risk_at(pt, hazards, terrain, risk_sampler)
        if r > peak:
            peak = r
    return float(peak)


def _risk_at(pt, hazards, terrain, risk_sampler):
    """Fast cached risk (trilinear grid lookup) when a sampler is given, else a
    live LOS-sampled computation."""
    if risk_sampler is not None:
        return float(risk_sampler(pt[0], pt[1], pt[2]))
    return compute_point_risk(pt, hazards, terrain=terrain, apply_los=True, los_samples=18)


def prune_waypoints(
    waypoints: np.ndarray,
    terrain: np.ndarray,
    min_agl: float = 15.0,
    xy_scale: float = 1.0,
    hazards: Optional[List[HazardSite]] = None,
    risk_tolerance: float = 0.08,
    risk_sampler=None,
) -> np.ndarray:
    """
    Risk-aware greedy line-of-sight path pruning.

    Removes jagged grid stair-steps while guaranteeing that every accepted
    shortcut:
      * keeps clear terrain line-of-sight,
      * preserves the minimum AGL clearance, and
      * does NOT route through more threat exposure than the original waypoints
        it replaces (within `risk_tolerance`).

    Parameters:
        waypoints:     (N, 3) planned path.
        terrain:       2D elevation grid.
        min_agl:       Hard terrain-clearance floor (m).
        xy_scale:      Metres per grid cell.
        hazards:       Active hazard sites (enables risk-aware pruning).
        risk_tolerance: Allowed exposure increase over the spanned waypoints.
    """
    waypoints = np.asarray(waypoints, dtype=float)
    N = len(waypoints)
    if N <= 2:
        return waypoints

    hazards = hazards or []

    # Precompute per-waypoint risk so a shortcut can be compared against the
    # exposure the planner already accepted across the span it would replace.
    if hazards or risk_sampler is not None:
        wp_risk = np.array([_risk_at(wp, hazards, terrain, risk_sampler) for wp in waypoints])
    else:
        wp_risk = np.zeros(N)

    pruned = [waypoints[0]]
    curr_idx = 0

    while curr_idx < N - 1:
        furthest_valid = curr_idx + 1
        for next_idx in range(N - 1, curr_idx, -1):
            p1 = waypoints[curr_idx]
            p2 = waypoints[next_idx]

            # 1. Terrain line-of-sight must stay clear.
            visible, _ = check_los(terrain, p1, p2, samples=40)
            if not visible:
                continue

            # 2. Minimum AGL clearance along the shortcut.
            safe = True
            for t in np.linspace(0, 1, 20):
                pt = p1 + t * (p2 - p1)
                ground = terrain_height(terrain, pt[0] / xy_scale, pt[1] / xy_scale)
                if pt[2] - ground < min_agl:
                    safe = False
                    break
            if not safe:
                continue

            # 3. Risk guard: the shortcut may not exceed the exposure the
            #    spanned waypoints already accepted (plus a small tolerance).
            if hazards or risk_sampler is not None:
                allowed = float(wp_risk[curr_idx:next_idx + 1].max()) + risk_tolerance
                if _segment_peak_risk(p1, p2, terrain, hazards, risk_sampler=risk_sampler) > allowed:
                    continue

            furthest_valid = next_idx
            break

        pruned.append(waypoints[furthest_valid])
        curr_idx = furthest_valid

    return np.array(pruned, dtype=float)


def smooth_trajectory(
    waypoints: np.ndarray,
    terrain: np.ndarray,
    num_points: int = 150,
    min_agl: float = 15.0,
    xy_scale: float = 1.0,
    hazards: Optional[List[HazardSite]] = None,
    smoothing: float = 2.0,
    risk_sampler=None,
) -> np.ndarray:
    """
    Smooth a sequence of 3D waypoints into a continuous flyable trajectory using
    a parametric B-spline, with terrain-clearance enforcement.

    Pass `risk_sampler` (e.g. PlanningGrid3D.risk_at) for a fast cached risk
    lookup in the risk-aware pruning; otherwise a live LOS computation is used.

    Falls back to a linear interpolation of the pruned waypoints when there are
    too few control points for a spline.
    """
    pruned = prune_waypoints(
        waypoints, terrain, min_agl=min_agl, xy_scale=xy_scale, hazards=hazards,
        risk_sampler=risk_sampler
    )

    if len(pruned) < 4:
        # Linear interpolation between the pruned control points.
        t_orig = np.linspace(0, 1, len(pruned))
        t_new = np.linspace(0, 1, num_points)
        smoothed = np.zeros((num_points, 3), dtype=float)
        for d in range(3):
            smoothed[:, d] = np.interp(t_new, t_orig, pruned[:, d])
    else:
        try:
            tck, _ = splprep(
                [pruned[:, 0], pruned[:, 1], pruned[:, 2]],
                s=smoothing, k=min(3, len(pruned) - 1)
            )
            u_new = np.linspace(0, 1, num_points)
            x_s, y_s, z_s = splev(u_new, tck)
            smoothed = np.column_stack([x_s, y_s, z_s])
        except Exception:
            return pruned

    # Enforce minimum terrain clearance on every smoothed sample.
    for i in range(len(smoothed)):
        ground = terrain_height(terrain, smoothed[i, 0] / xy_scale, smoothed[i, 1] / xy_scale)
        if smoothed[i, 2] < ground + min_agl:
            smoothed[i, 2] = ground + min_agl

    return smoothed
