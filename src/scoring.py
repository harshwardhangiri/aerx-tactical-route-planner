"""
AerX Labs — Multi-Objective Survivability & Mission Scoring
src/scoring.py

Computes normalized mission quality and survivability scores for candidate routes:
- Simulated Survivability Probability: S_survive = exp(-lambda_risk * R_integrated)
- Mission Efficiency Score: Normalized distance / time penalty
- Terrain Feasibility & Clearance Score
- Composite Overall Score in [0, 100]
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from src.route_metrics import RouteMetrics


@dataclass
class RoundTripEvaluation:
    """
    Full-sortie (ingress + egress) evaluation: survive the whole mission AND get
    home on one tank of fuel.
    """
    total_distance: float          # ingress + egress 3D distance (m)
    total_time: float              # ingress + egress flight time (s)
    total_energy: float            # ingress + egress energy/fuel proxy
    total_integrated_risk: float   # cumulative exposure over both legs
    survivability: float           # exp(-lambda * (R_ingress + R_egress)) in (0, 1]
    min_agl: float                 # tightest clearance over the whole sortie
    fuel_budget: float             # onboard fuel budget (energy units)
    fuel_used: float               # = total_energy
    fuel_margin_pct: float         # (budget - used)/budget * 100
    fuel_ok: bool                  # True if the round trip fits the budget
    has_egress: bool               # False if only the ingress leg was available


def evaluate_round_trip(
    ingress: RouteMetrics,
    egress: Optional[RouteMetrics],
    direct_euclidean_distance: float,
    nominal_speed: float = 30.0,
    fuel_budget: Optional[float] = None,
    risk_sensitivity: float = 0.05,
) -> RoundTripEvaluation:
    """
    Combine the ingress and egress legs into one sortie evaluation.

    Survivability is the probability of surviving BOTH legs:
        S = exp(-lambda * (R_ingress + R_egress))

    Fuel is a fixed onboard budget the whole round trip must fit within. Pass an
    explicit `fuel_budget` (energy units); when None, a provisional budget is set
    and the fuel verdict is deferred (the caller sizes the real budget once all
    candidate routes are known — typically a reserve over the most efficient one).
    A route whose combined energy exceeds the budget is flagged fuel-infeasible.
    """
    if egress is not None:
        total_energy = ingress.energy_proxy + egress.energy_proxy
        total_risk = ingress.integrated_risk + egress.integrated_risk
        total_dist = ingress.total_distance + egress.total_distance
        total_time = ingress.flight_time + egress.flight_time
        min_agl = min(ingress.min_agl, egress.min_agl)
        has_egress = True
    else:
        total_energy = ingress.energy_proxy
        total_risk = ingress.integrated_risk
        total_dist = ingress.total_distance
        total_time = ingress.flight_time
        min_agl = ingress.min_agl
        has_egress = False

    surv = float(np.clip(np.exp(-risk_sensitivity * total_risk), 0.0, 1.0))

    if fuel_budget is None:
        budget = 0.0
        margin = 0.0
        ok = True  # provisional (verdict deferred until the real budget is set)
    else:
        budget = float(fuel_budget)
        margin = (budget - total_energy) / budget * 100.0 if budget > 0 else 0.0
        ok = total_energy <= budget

    return RoundTripEvaluation(
        total_distance=float(total_dist),
        total_time=float(total_time),
        total_energy=float(total_energy),
        total_integrated_risk=float(total_risk),
        survivability=surv,
        min_agl=float(min_agl),
        fuel_budget=budget,
        fuel_used=float(total_energy),
        fuel_margin_pct=float(margin),
        fuel_ok=ok,
        has_egress=has_egress,
    )


def apply_fuel_budget(round_trips: List["RoundTripEvaluation"], reserve_factor: float = 1.15) -> float:
    """
    Size one shared onboard fuel budget for a set of candidate round trips and
    stamp the fuel verdict onto each. The budget is `reserve_factor` times the
    energy of the most fuel-efficient candidate — enough for the efficient sortie
    plus a reserve; routes needing meaningfully more energy (e.g. long low-risk
    detours) are then flagged fuel-infeasible. Returns the chosen budget.
    """
    if not round_trips:
        return 0.0
    min_energy = min(rt.total_energy for rt in round_trips)
    budget = reserve_factor * min_energy
    for rt in round_trips:
        rt.fuel_budget = budget
        rt.fuel_used = rt.total_energy
        rt.fuel_margin_pct = (budget - rt.total_energy) / budget * 100.0 if budget > 0 else 0.0
        rt.fuel_ok = rt.total_energy <= budget
    return budget


@dataclass
class ScoreBreakdown:
    """Detailed breakdown of normalized sub-scores and final aggregate score."""
    survivability_score: float     # [0, 1] based on exponential integrated risk
    distance_efficiency: float     # [0, 1] compared to straight-line Euclidean distance
    time_efficiency: float         # [0, 1] normalized time score
    clearance_score: float         # [0, 1] safety clearance score
    composite_score: float         # Overall score scaled to [0, 100]
    rating_label: str              # Qualitative rating ("Optimal", "High Risk", etc.)


@dataclass
class MissionScore:
    """Full-sortie (ingress + egress) mission score, fuel feasibility folded in."""
    survivability: float          # round-trip survival probability [0, 1]
    distance_efficiency: float    # round-trip distance vs direct there-and-back
    time_efficiency: float
    clearance_score: float
    fuel_ok: bool
    fuel_factor: float            # multiplier applied when the sortie is over budget
    composite_score: float        # [0, 100] mission score (fuel penalty applied)
    rating_label: str


def compute_mission_score(
    round_trip,
    direct_euclidean_distance: float,
    min_required_agl: float = 15.0,
    w_survive: float = 0.50,
    w_distance: float = 0.25,
    w_time: float = 0.15,
    w_clearance: float = 0.10,
    nominal_speed: float = 30.0,
) -> MissionScore:
    """
    Score the WHOLE sortie: fly in, survive, and get home on one tank.

    It reuses the same weighted structure as the single-leg score, but on
    round-trip quantities, and applies a fuel-feasibility factor: a route the
    aircraft cannot complete on its fuel budget is a mission failure, so its
    composite is cut in half and it is labelled FUEL-INFEASIBLE. This lets the
    headline number rank a route that is BOTH survivable AND able to return above
    one that is merely survivable.
    """
    rt = round_trip
    direct_rt = 2.0 * direct_euclidean_distance

    s_survive = rt.survivability
    s_dist = float(direct_rt / max(rt.total_distance, direct_rt))
    nominal_time = direct_rt / nominal_speed
    s_time = float(nominal_time / max(rt.total_time, nominal_time))
    if rt.min_agl < min_required_agl:
        s_clear = 0.0
    else:
        s_clear = float(np.clip((rt.min_agl - min_required_agl) / 20.0, 0.2, 1.0))

    base = 100.0 * (w_survive * s_survive + w_distance * s_dist
                    + w_time * s_time + w_clearance * s_clear)
    fuel_factor = 1.0 if rt.fuel_ok else 0.5
    composite = base * fuel_factor

    if not rt.fuel_ok:
        rating = "FUEL-INFEASIBLE: cannot complete round trip"
    elif s_survive > 0.85 and composite >= 70.0:
        rating = "EXCELLENT: survivable round trip on budget"
    elif s_survive > 0.60:
        rating = "GOOD: viable sortie"
    elif s_survive <= 0.40:
        rating = "HIGH RISK: severe exposure"
    else:
        rating = "SUB-OPTIMAL"

    return MissionScore(
        survivability=float(s_survive),
        distance_efficiency=s_dist,
        time_efficiency=s_time,
        clearance_score=s_clear,
        fuel_ok=bool(rt.fuel_ok),
        fuel_factor=float(fuel_factor),
        composite_score=float(composite),
        rating_label=rating,
    )


def calculate_survivability_score(integrated_risk: float, risk_sensitivity: float = 0.05) -> float:
    """
    Calculate simulated mission survivability score:
      S_survive = exp(-risk_sensitivity * R_integrated)

    Returns:
        float in (0, 1.0], where 1.0 indicates zero threat exposure.
    """
    score = np.exp(-float(risk_sensitivity) * float(integrated_risk))
    return float(np.clip(score, 0.0, 1.0))


def compute_route_score(
    metrics: RouteMetrics,
    direct_euclidean_distance: float,
    min_required_agl: float = 15.0,
    risk_sensitivity: float = 0.05,
    w_survive: float = 0.50,
    w_distance: float = 0.25,
    w_time: float = 0.15,
    w_clearance: float = 0.10
) -> ScoreBreakdown:
    """
    Compute normalized multi-objective mission score.

    Parameters:
        metrics: RouteMetrics instance
        direct_euclidean_distance: Direct line-of-sight distance between start and goal
        min_required_agl: Minimum required safety clearance
        risk_sensitivity: Scaling factor for exponential risk decay
        w_*: Relative scoring weights (sum to 1.0)

    Returns:
        ScoreBreakdown instance.
    """
    # 1. Survivability score
    s_survive = calculate_survivability_score(metrics.integrated_risk, risk_sensitivity)

    # 2. Distance efficiency score (ratio of straight-line to actual path, <= 1.0)
    s_dist = float(direct_euclidean_distance / max(metrics.total_distance, direct_euclidean_distance))

    # 3. Time efficiency score
    nominal_min_time = direct_euclidean_distance / 30.0  # at nominal 30 m/s
    s_time = float(nominal_min_time / max(metrics.flight_time, nominal_min_time))

    # 4. Clearance safety score (penalize flying too close to minimum AGL)
    if metrics.min_agl < min_required_agl:
        s_clearance = 0.0  # Violation of hard safety limit
    else:
        # Optimal clearance zone: 25m - 80m AGL
        excess_margin = metrics.min_agl - min_required_agl
        s_clearance = float(np.clip(excess_margin / 20.0, 0.2, 1.0))

    # Composite weighted score (0 to 100)
    composite = 100.0 * (
        w_survive * s_survive
        + w_distance * s_dist
        + w_time * s_time
        + w_clearance * s_clearance
    )

    # Rating qualitative assessment
    if s_clearance == 0.0:
        rating = "CRITICAL: Terrain Clearance Violation"
    elif s_survive > 0.85 and composite >= 80.0:
        rating = "EXCELLENT: Low Risk & High Efficiency"
    elif s_survive > 0.70 and composite >= 65.0:
        rating = "GOOD: Balanced Tactical Profile"
    elif s_survive <= 0.40:
        rating = "HIGH RISK: Severe Threat Exposure"
    else:
        rating = "SUB-OPTIMAL: Length / Risk Penalty"

    return ScoreBreakdown(
        survivability_score=float(s_survive),
        distance_efficiency=float(s_dist),
        time_efficiency=float(s_time),
        clearance_score=float(s_clearance),
        composite_score=float(composite),
        rating_label=rating
    )
