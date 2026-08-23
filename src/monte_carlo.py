"""
AerX Labs — Monte Carlo Robustness & Uncertainty Evaluation
src/monte_carlo.py

Evaluates candidate route sensitivity and survivability robustness under
stochastic operational uncertainties:
- Flight trajectory tracking noise (position perturbations)
- Threat / hazard location uncertainty (sensor location drift)
- Hazard emission intensity fluctuations
- Wind / travel time perturbations
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from src.risk_field import HazardSite
from src.los import terrain_height
from src.route_metrics import compute_route_metrics
from src.scoring import compute_route_score, evaluate_round_trip, compute_mission_score


def _perturb_waypoints(waypoints: np.ndarray, terrain: np.ndarray, pos_noise_std: float,
                       min_agl: float = 15.0) -> np.ndarray:
    """Add tracking noise to the interior waypoints, keeping start/end fixed and
    enforcing the clearance floor so noise can't manufacture terrain crashes."""
    wp = np.asarray(waypoints, dtype=float).copy()
    n = len(wp)
    if n > 2:
        wp[1:-1, :2] += np.random.normal(0, pos_noise_std, size=(n - 2, 2))
        wp[1:-1, 2:] += np.random.normal(0, pos_noise_std * 0.5, size=(n - 2, 1))
        for i in range(1, n - 1):
            g = terrain_height(terrain, wp[i, 0], wp[i, 1])
            if wp[i, 2] < g + min_agl:
                wp[i, 2] = g + min_agl
    return wp


def _perturb_hazards(hazards: List[HazardSite], pos_std: float, intensity_std: float) -> List[HazardSite]:
    """Draw one perturbed realization of the threat sites (location + strength)."""
    out = []
    for h in hazards:
        out.append(HazardSite(
            x=h.x + np.random.normal(0, pos_std),
            y=h.y + np.random.normal(0, pos_std),
            z=h.z, mast_height=h.mast_height, radius=h.radius,
            intensity=float(np.clip(h.intensity + np.random.normal(0, intensity_std), 0.1, 1.0)),
            requires_los=h.requires_los, decay_type=h.decay_type, name=h.name,
            threat_type=h.threat_type, min_detect_alt=h.min_detect_alt,
            azimuth_center=h.azimuth_center, azimuth_width=h.azimuth_width,
            elev_angle_min=h.elev_angle_min, elev_angle_max=h.elev_angle_max,
        ))
    return out


@dataclass
class MonteCarloResults:
    """Statistical summary of Monte Carlo uncertainty trials."""
    num_trials: int
    mean_composite_score: float
    std_composite_score: float
    p5_worst_case_score: float       # 5th percentile (robust lower bound)
    p50_median_score: float          # 50th percentile (median)
    p95_best_case_score: float       # 95th percentile
    mean_survivability: float
    std_survivability: float
    mean_integrated_risk: float
    std_integrated_risk: float
    all_composite_scores: np.ndarray
    all_survivability_scores: np.ndarray


def run_monte_carlo_evaluation(
    waypoints: np.ndarray,
    terrain: np.ndarray,
    hazards: List[HazardSite],
    direct_euclidean_distance: float,
    num_trials: int = 100,
    pos_noise_std: float = 1.5,       # Waypoint position tracking noise (meters)
    hazard_pos_noise_std: float = 2.0, # Hazard location estimation uncertainty (meters)
    hazard_intensity_noise_std: float = 0.15, # Hazard intensity variation (+/- 15%)
    random_seed: Optional[int] = 42
) -> MonteCarloResults:
    """
    Run Monte Carlo stochastic evaluation on a planned flight route.

    Parameters:
        waypoints: (N, 3) planned waypoint coordinates
        terrain: 2D elevation grid
        hazards: Nominal hazard sites
        direct_euclidean_distance: Baseline Euclidean distance
        num_trials: Number of stochastic simulation iterations
        pos_noise_std: Standard deviation of aircraft position perturbation (m)
        hazard_pos_noise_std: Uncertainty in hazard position (m)
        hazard_intensity_noise_std: Standard deviation of hazard intensity fluctuation
        random_seed: Random seed for repeatability

    Returns:
        MonteCarloResults dataclass with empirical statistics.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    waypoints = np.asarray(waypoints, dtype=float)
    N = len(waypoints)

    composite_scores = np.zeros(num_trials, dtype=float)
    surv_scores = np.zeros(num_trials, dtype=float)
    integrated_risks = np.zeros(num_trials, dtype=float)

    for trial in range(num_trials):
        # 1. Perturb waypoints (simulate flight execution tracking errors)
        # Keep start and end fixed
        perturbed_waypoints = waypoints.copy()
        if N > 2:
            noise_xy = np.random.normal(0, pos_noise_std, size=(N - 2, 2))
            noise_z = np.random.normal(0, pos_noise_std * 0.5, size=(N - 2, 1))
            perturbed_waypoints[1:-1, :2] += noise_xy
            perturbed_waypoints[1:-1, 2:] += noise_z
            # Enforce safety clearance floor so tracking noise does not generate artificial terrain crashes
            for pt_idx in range(1, N - 1):
                px = perturbed_waypoints[pt_idx, 0]
                py = perturbed_waypoints[pt_idx, 1]
                g_z = terrain_height(terrain, px, py)
                if perturbed_waypoints[pt_idx, 2] < g_z + 15.0:
                    perturbed_waypoints[pt_idx, 2] = g_z + 15.0

        # 2. Perturb hazards (simulate sensor location drift and intensity variance)
        perturbed_hazards = []
        for h in hazards:
            dx = np.random.normal(0, hazard_pos_noise_std)
            dy = np.random.normal(0, hazard_pos_noise_std)
            d_intensity = np.random.normal(0, hazard_intensity_noise_std)
            
            p_intensity = float(np.clip(h.intensity + d_intensity, 0.1, 1.0))
            perturbed_hazards.append(
                HazardSite(
                    x=h.x + dx,
                    y=h.y + dy,
                    z=h.z,
                    mast_height=h.mast_height,
                    radius=h.radius,
                    intensity=p_intensity,
                    requires_los=h.requires_los,
                    decay_type=h.decay_type,
                    name=h.name
                )
            )

        # 3. Evaluate metrics and scores for this realization
        metrics = compute_route_metrics(
            waypoints=perturbed_waypoints,
            terrain=terrain,
            hazards=perturbed_hazards
        )
        score_breakdown = compute_route_score(
            metrics=metrics,
            direct_euclidean_distance=direct_euclidean_distance
        )

        composite_scores[trial] = score_breakdown.composite_score
        surv_scores[trial] = score_breakdown.survivability_score
        integrated_risks[trial] = metrics.integrated_risk

    return MonteCarloResults(
        num_trials=num_trials,
        mean_composite_score=float(np.mean(composite_scores)),
        std_composite_score=float(np.std(composite_scores)),
        p5_worst_case_score=float(np.percentile(composite_scores, 5)),
        p50_median_score=float(np.percentile(composite_scores, 50)),
        p95_best_case_score=float(np.percentile(composite_scores, 95)),
        mean_survivability=float(np.mean(surv_scores)),
        std_survivability=float(np.std(surv_scores)),
        mean_integrated_risk=float(np.mean(integrated_risks)),
        std_integrated_risk=float(np.std(integrated_risks)),
        all_composite_scores=composite_scores,
        all_survivability_scores=surv_scores
    )


def run_sortie_monte_carlo(
    ingress_waypoints: np.ndarray,
    egress_waypoints: Optional[np.ndarray],
    terrain: np.ndarray,
    hazards: List[HazardSite],
    direct_euclidean_distance: float,
    nominal_fuel_factor: float = 1.0,
    num_trials: int = 50,
    pos_noise_std: float = 1.2,
    hazard_pos_noise_std: float = 1.5,
    hazard_intensity_noise_std: float = 0.12,
    random_seed: Optional[int] = 42,
) -> MonteCarloResults:
    """
    Full-sortie Monte Carlo: perturb BOTH the ingress and egress routes and the
    (shared) threat sites, then score the round trip as a mission. It tests the
    genuinely uncertain quantity — SURVIVABILITY under threat-location and
    tracking uncertainty — while holding the planned fuel verdict fixed (fuel is a
    deterministic property of the planned route, not something tracking noise
    should re-roll), applied via `nominal_fuel_factor`. So the reported composite
    is the full-sortie mission score, and its spread reflects survivability
    robustness across the whole ingress + egress mission.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Monte Carlo is statistical, so coarsen the geometry: sample a subset of the
    # waypoints and use fewer LOS rays per query. This cuts the (dominant) risk-field
    # cost several-fold with negligible effect on the composite distribution.
    def _thin(w):
        w = np.asarray(w, dtype=float)
        if len(w) <= 45:
            return w
        idx = np.linspace(0, len(w) - 1, 45).round().astype(int)
        return w[idx]

    ingress_waypoints = _thin(ingress_waypoints)
    has_egress = egress_waypoints is not None
    if has_egress:
        egress_waypoints = _thin(egress_waypoints)
    MC_LOS = 12

    composite = np.zeros(num_trials)
    surv = np.zeros(num_trials)
    risks = np.zeros(num_trials)

    for t in range(num_trials):
        # Both legs fly through the SAME perturbed threat picture this trial.
        pert_haz = _perturb_hazards(hazards, hazard_pos_noise_std, hazard_intensity_noise_std)

        ing = compute_route_metrics(_perturb_waypoints(ingress_waypoints, terrain, pos_noise_std),
                                    terrain, pert_haz, los_samples=MC_LOS)
        eg = (compute_route_metrics(_perturb_waypoints(egress_waypoints, terrain, pos_noise_std),
                                    terrain, pert_haz, los_samples=MC_LOS) if has_egress else None)

        # Fuel held feasible here (deterministic from the plan); the planned fuel
        # penalty is applied uniformly via nominal_fuel_factor.
        rt = evaluate_round_trip(ing, eg, direct_euclidean_distance, fuel_budget=float("inf"))
        ms = compute_mission_score(rt, direct_euclidean_distance)

        composite[t] = ms.composite_score * nominal_fuel_factor
        surv[t] = rt.survivability
        risks[t] = rt.total_integrated_risk

    return MonteCarloResults(
        num_trials=num_trials,
        mean_composite_score=float(np.mean(composite)),
        std_composite_score=float(np.std(composite)),
        p5_worst_case_score=float(np.percentile(composite, 5)),
        p50_median_score=float(np.percentile(composite, 50)),
        p95_best_case_score=float(np.percentile(composite, 95)),
        mean_survivability=float(np.mean(surv)),
        std_survivability=float(np.std(surv)),
        mean_integrated_risk=float(np.mean(risks)),
        std_integrated_risk=float(np.std(risks)),
        all_composite_scores=composite,
        all_survivability_scores=surv,
    )
