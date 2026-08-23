"""
AerX Labs — Advanced Aerospace Systems Challenge 2026
Threat-Aware Tactical Route Planning Engine

Single entry point. Every mission runs on a REAL Copernicus GLO-30 DEM and is
planned as a full sortie (ingress + egress, scored on one fuel budget):

    python main.py everest                      # Mount Everest / Khumbu
    python main.py ghats                         # Western Ghats / Anamalai
    python main.py region --lat 34.1 --lon 74.8  # any point on Earth

Each run writes a fresh, timestamped folder under results\\<region>\\<stamp>\\
so nothing is ever overwritten. See run_full_suite() for the outputs produced.
"""

import argparse
import os
from datetime import datetime

# Configure the Matplotlib backend BEFORE pyplot is imported anywhere.
# Every run renders headlessly to PNG/HTML — no interactive Tk window.
from src.render_env import configure_matplotlib
configure_matplotlib(interactive=False)

import numpy as np

from src.risk_field import load_mission_scenario, set_custom_dem
from src.planning_grid import CostWeights, PlanningGrid3D
from src.dstar_lite import DStarLite3D
from src.smoothing import smooth_trajectory
from src.route_metrics import compute_route_metrics
from src.scoring import compute_route_score, evaluate_round_trip
from src.terrain import download_copernicus_tile, find_relief_window
from src.visualization import render_tactical_dashboard, render_mission_scorecard
from src.analysis import pareto_sweep, render_pareto, replanning_demo, render_replanning
from src.web_export import export_web_3d
from src.cesium_export import export_cesium


# Global scenario result cache for instant switching
_SCENARIO_CACHE = {}

# Local-refinement mode ("none" = B-spline smoothing, "rrt" = risk-aware RRT*).
# Set from the --refine CLI flag before solving.
REFINE_MODE = "none"


def solve_mission_scenario(scenario_key: str = "dem", scorecard_dir: str = "results/figures"):
    """
    Computes (or retrieves from cache) the complete path planning, metrics,
    and Monte Carlo analysis for a given tactical scenario.

    Returns:
        (terrain, hazards, start_pos, goal_pos, routes_data, title, eval_altitude)
    """
    if scenario_key in _SCENARIO_CACHE:
        return _SCENARIO_CACHE[scenario_key]

    print("\n" + "=" * 80)
    print(f"  SOLVING MISSION SCENARIO: {scenario_key.upper()}")
    print("=" * 80)

    # 1. Load Scenario Environment
    terrain, hazards, start_pos, goal_pos, z_layers, title, desc = load_mission_scenario(scenario_key)
    ny, nx = terrain.shape

    # Baseline "direct" distance between the grid-snapped start and goal actually
    # used by the planner, so every route's Detour % is >= 0 (a flyable path can
    # never be shorter than the straight chord it spans).
    def _snap(p):
        ix = int(np.clip(round(p[0]), 0, nx - 1))
        iy = int(np.clip(round(p[1]), 0, ny - 1))
        iz = int(np.argmin(np.abs(z_layers - p[2])))
        return np.array([ix, iy, z_layers[iz]], dtype=float)

    direct_dist = float(np.linalg.norm(_snap(goal_pos) - _snap(start_pos)))

    print(f"\n[Step 1] Profile: {title}")
    print(f"  Description       : {desc}")
    print(f"  Grid Dimensions   : {nx} x {ny}")
    print(f"  Elevation Range   : {terrain.min():.1f} m to {terrain.max():.1f} m ASL")
    print(f"  Start (Ingress)   : {start_pos} m ASL | Target: {goal_pos} m ASL (Line: {direct_dist:.1f} m)")

    # 2. Threat sites
    print(f"\n[Step 2] Active Threat & Sensor Sites ({len(hazards)} sites):")
    for idx, h in enumerate(hazards, 1):
        pos = h.get_position(terrain)
        print(f"  [{idx}] {h.name:<32} | Pos: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}m) | Radius: {h.radius:.0f}m")

    # 3. 3D Planning Grid.
    # High-relief real DEMs need a terrain-following corridor: a much larger AGL
    # ceiling (so valley-to-ridge routing stays feasible) and a slightly higher
    # clearance floor. The synthetic scenarios (relief ~200 m) keep the tight NOE
    # band. Detected automatically from the terrain relief.
    relief = float(terrain.max() - terrain.min())
    if relief > 400.0:
        grid_min_agl, grid_max_agl, grid_climb = 25.0, relief + 120.0, 1.0
    else:
        grid_min_agl, grid_max_agl, grid_climb = 15.0, 220.0, 0.85
    print("\n[Step 3] Initializing 3D Layered Planning Grid...", flush=True)
    base_grid = PlanningGrid3D(
        terrain=terrain,
        z_layers=z_layers,
        min_agl=grid_min_agl,
        max_agl=grid_max_agl,
        max_climb_slope=grid_climb,
        hazards=hazards,
        weights=CostWeights.balanced()
    )

    # 4. Multi-Objective Path Optimization
    print("\n[Step 4] Optimizing Multi-Objective 3D D* Lite Route Profiles...", flush=True)
    profiles = [
        ("Distance-focused", CostWeights.distance_focused()),
        ("Risk-focused",     CostWeights.risk_focused()),
        ("Balanced",         CostWeights.balanced())
    ]

    routes_data = {}

    for name, weights in profiles:
        print(f"\n  --- Profile: {name} ---", flush=True)
        base_grid.weights = weights
        planner = DStarLite3D(base_grid)

        print("    [>] Computing D* Lite global trajectory...", end="", flush=True)
        discrete_path = planner.plan(start_pos, goal_pos)
        print(" [OK]", flush=True)

        if discrete_path is None:
            print(f"    [!] Warning: No feasible path for {name}", flush=True)
            continue

        raw_coords = planner.get_path_coordinates(discrete_path)
        smooth_coords = smooth_trajectory(
            raw_coords, terrain, num_points=120, min_agl=15.0, hazards=hazards,
            risk_sampler=base_grid.risk_at
        )

        # Optional risk-aware RRT* local refinement of the global route.
        if REFINE_MODE == "rrt":
            from src.rrt_star import refine_with_rrt_star, RRTStarConfig
            print("    [>] Refining with risk-aware RRT*...", end="", flush=True)
            refined = refine_with_rrt_star(
                raw_coords, terrain, hazards,
                RRTStarConfig(max_iterations=900), risk_sampler=base_grid.risk_at
            )
            if refined is not None and len(refined) >= 2:
                smooth_coords = smooth_trajectory(
                    refined, terrain, num_points=120, min_agl=15.0, hazards=hazards,
                    risk_sampler=base_grid.risk_at
                )
            print(" [OK]", flush=True)

        # Calculate metrics
        metrics = compute_route_metrics(smooth_coords, terrain, hazards)
        score = compute_route_score(metrics, direct_euclidean_distance=direct_dist)

        # Egress route: fly the mission back out (target -> base). Because the
        # edge cost is direction-dependent (climb costs more than descent), the
        # egress path can differ from the ingress path — the ingress/egress
        # asymmetry the concept document calls for.
        print("    [>] Computing egress (return) trajectory...", end="", flush=True)
        egress_planner = DStarLite3D(base_grid)
        egress_discrete = egress_planner.plan(goal_pos, start_pos)
        egress_smooth = None
        egress_metrics = None
        if egress_discrete is not None:
            egress_raw = egress_planner.get_path_coordinates(egress_discrete)
            egress_smooth = smooth_trajectory(
                egress_raw, terrain, num_points=120, min_agl=15.0, hazards=hazards,
                risk_sampler=base_grid.risk_at
            )
            egress_metrics = compute_route_metrics(egress_smooth, terrain, hazards)
            print(" [OK]", flush=True)
        else:
            print(" [!] no egress path", flush=True)

        # Full-sortie (round-trip) evaluation: survive both legs AND get home on
        # one fuel budget.
        round_trip = evaluate_round_trip(metrics, egress_metrics, direct_euclidean_distance=direct_dist)

        routes_data[name] = {
            "raw_coords": raw_coords,
            "smooth_coords": smooth_coords,
            "metrics": metrics,
            "score": score,
            "egress_smooth_coords": egress_smooth,
            "egress_metrics": egress_metrics,
            "round_trip": round_trip,
        }

        print(f"    Ingress Distance : {metrics.total_distance:.1f} m (Detour: +{(metrics.total_distance - direct_dist)/direct_dist*100:.1f}%)", flush=True)
        if egress_metrics is not None:
            print(f"    Egress Distance  : {egress_metrics.total_distance:.1f} m | Egress Risk: {egress_metrics.integrated_risk:.2f}", flush=True)
        print(f"    Round-trip       : {round_trip.total_distance:.1f} m | Survivability(I+E): {round_trip.survivability*100:.1f}%", flush=True)

    # 4b. Size one shared onboard fuel budget for the whole sortie (ingress +
    #     egress), stamp each route's fuel verdict, and score the full mission.
    from src.scoring import apply_fuel_budget, compute_mission_score
    from src.monte_carlo import run_sortie_monte_carlo
    round_trips = [r["round_trip"] for r in routes_data.values()]
    fuel_budget = apply_fuel_budget(round_trips)
    for r in routes_data.values():
        r["mission_score"] = compute_mission_score(r["round_trip"], direct_euclidean_distance=direct_dist)

    # 4c. Full-sortie Monte-Carlo robustness (perturb both legs + shared threats,
    #     score the mission against the fixed fuel budget).
    for name, r in routes_data.items():
        print(f"    [>] Sortie Monte-Carlo robustness ({name})...", end="", flush=True)
        r["mc"] = run_sortie_monte_carlo(
            ingress_waypoints=r["smooth_coords"],
            egress_waypoints=r.get("egress_smooth_coords"),
            terrain=terrain, hazards=hazards,
            direct_euclidean_distance=direct_dist,
            nominal_fuel_factor=r["mission_score"].fuel_factor, num_trials=28,
        )
        print(" [OK]", flush=True)

    # 5. Output Summary Table (round-trip / full-sortie view).
    print("\n" + "=" * 104)
    print(f"          CANDIDATE ROUTE EVALUATION MATRIX ({scenario_key.upper()})  —  full sortie (ingress + egress)")
    print(f"          Onboard fuel budget: {fuel_budget:.1f} energy units (round trip must fit within this)")
    print("=" * 104)
    print(f"{'Route Profile':<18} | {'RT Dist':<8} | {'RT Time':<8} | {'RT Risk':<8} | {'Surv I+E':<9} | {'Fuel Use':<9} | {'Fuel':<11} | {'Mission':<8}")
    print("-" * 104)
    for name, r in routes_data.items():
        rt = r["round_trip"]
        ms = r["mission_score"]
        fuel_str = f"{rt.fuel_margin_pct:+.0f}% " + ("OK" if rt.fuel_ok else "OVER")
        print(
            f"{name:<18} | {rt.total_distance:<8.1f} | {rt.total_time:<8.1f} | "
            f"{rt.total_integrated_risk:<8.2f} | {rt.survivability*100:<8.1f}% | "
            f"{rt.fuel_used:<9.1f} | {fuel_str:<11} | {ms.composite_score:<8.1f}"
        )
    print("=" * 104)

    # Export Infographic Scorecard
    scorecard_save_path = f"{scorecard_dir}/mission_scorecard_and_metrics_guide_{scenario_key}.png"
    render_mission_scorecard(
        routes_data=routes_data,
        scenario_title=title,
        direct_dist=direct_dist,
        save_path=scorecard_save_path
    )

    eval_alt = float(z_layers[len(z_layers) // 3])
    result_tuple = (terrain, hazards, start_pos, goal_pos, routes_data, title, eval_alt)
    _SCENARIO_CACHE[scenario_key] = result_tuple
    return result_tuple


# --------------------------------------------------------------------------- #
#  Full analysis suite (one region -> one timestamped folder)
# --------------------------------------------------------------------------- #

# CLI keyword -> (engine scenario key, filename prefix, human name, (lon, lat)).
# lon/lat georeference the CesiumJS globe scene for the region's DEM crop.
REGIONS = {
    "everest": ("dem",   "everest", "Mount Everest / Khumbu",     (86.4, 27.25)),
    "ghats":   ("ghats", "ghats",   "Western Ghats / Anamalai",   (77.13, 10.42)),
}


def run_full_suite(scenario, pfx, human, out, geo, quick=False):
    """
    Solve one mission and render its product set into folder `out`:

        <pfx>_scorecard.png    mission metrics + survivability scorecard
        <pfx>_dashboard.png    3D tactical dashboard (terrain + threats + routes)
      (full run also adds:)
        <pfx>_pareto.png       risk-weight Pareto trade-off sweep
        <pfx>_replanning.png   D* Lite pop-up-threat incremental repair demo
        <pfx>_3d.html          interactive Plotly 3D view
        <pfx>_cesium.html      CesiumJS georeferenced globe flythrough

    `quick=True` stops after the scorecard + dashboard for fast iteration.
    """
    os.makedirs(out, exist_ok=True)
    print("\n" + "=" * 78)
    print(f"  AerX — {human} (real Copernicus GLO-30 DEM)")
    print(f"  Output folder: {out}")
    print("=" * 78)

    # Fresh solve (drop any cached result so re-runs never reuse stale terrain).
    _SCENARIO_CACHE.pop(scenario, None)
    terrain, hazards, start, goal, routes, title, eval_alt = solve_mission_scenario(
        scenario, scorecard_dir=out)
    produced = os.path.join(out, f"mission_scorecard_and_metrics_guide_{scenario}.png")
    if os.path.isfile(produced):
        os.replace(produced, os.path.join(out, f"{pfx}_scorecard.png"))

    print(f"\n[render] {pfx}: tactical dashboard ...", flush=True)
    render_tactical_dashboard(
        terrain=terrain, hazards=hazards, start_pos=start, goal_pos=goal,
        routes_data=routes, scenario_title=title, eval_altitude=eval_alt,
        save_path=os.path.join(out, f"{pfx}_dashboard.png"), show_interactive=False,
    )

    if quick:
        print(f"\n[quick] scorecard + dashboard only. Done.\n  {os.path.abspath(out)}")
        return out

    print(f"[render] {pfx}: Pareto risk-weight sweep ...", flush=True)
    render_pareto(pareto_sweep(scenario), os.path.join(out, f"{pfx}_pareto.png"))
    print(f"[render] {pfx}: incremental-replanning demo ...", flush=True)
    render_replanning(replanning_demo(scenario), os.path.join(out, f"{pfx}_replanning.png"))
    print(f"[render] {pfx}: Plotly 3D + CesiumJS globe ...", flush=True)
    export_web_3d(terrain, hazards, start, goal, routes, title,
                  os.path.join(out, f"{pfx}_3d.html"))
    lon0, lat0 = geo
    export_cesium(terrain, hazards, start, goal, routes, title,
                  os.path.join(out, f"{pfx}_cesium.html"), lon0=lon0, lat0=lat0)

    print("\n" + "=" * 78)
    print(f"  DONE — {human}. All products saved in:")
    print(f"    {os.path.abspath(out)}")
    print("=" * 78)
    return out


def _stamped_out(label):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join("results", label, stamp)


def run_region_preset(key, quick=False):
    """Run one of the built-in real-DEM regions (everest / ghats)."""
    scenario, pfx, human, geo = REGIONS[key]
    return run_full_suite(scenario, pfx, human, _stamped_out(pfx), geo, quick=quick)


def run_custom_region(lat, lon, label=None, quick=False):
    """Download the Copernicus tile for any lat/lon and run the full suite."""
    label = label or f"lat{lat:.2f}_lon{lon:.2f}".replace(".", "p").replace("-", "m")
    print("=" * 78)
    print(f"  AerX — custom region  lat {lat}, lon {lon}")
    print("=" * 78)
    tile = download_copernicus_tile(lat, lon)
    row0, col0 = find_relief_window(tile)
    title = f"Custom Region (lat {lat:.3f}, lon {lon:.3f})"
    desc = (f"Real Copernicus GLO-30 DEM near lat {lat:.3f}, lon {lon:.3f}, "
            f"normalized to planning units — generic hilltop radar + valley SAM.")
    set_custom_dem(tile, row0, col0, title, desc)
    return run_full_suite("custom", label, title, _stamped_out(label),
                          (float(lon), float(lat)), quick=quick)


def main():
    parser = argparse.ArgumentParser(
        description="AerX Labs — Threat-Aware Tactical Route Planning (real Copernicus DEMs).")
    parser.add_argument(
        "target",
        choices=["everest", "ghats", "region"],
        help="everest / ghats = built-in real DEM regions; "
             "region = any lat/lon (requires --lat and --lon).")
    parser.add_argument("--lat", type=float, help="Latitude for 'region' (degrees).")
    parser.add_argument("--lon", type=float, help="Longitude for 'region' (degrees).")
    parser.add_argument("--label", default=None, help="Folder/filename label for 'region'.")
    parser.add_argument("--quick", action="store_true",
                        help="Scorecard + dashboard only (skip Pareto, replanning, Plotly, Cesium).")
    parser.add_argument("--refine", choices=["none", "rrt"], default="none",
                        help="Local refinement: 'none' (B-spline) or 'rrt' (risk-aware RRT*).")
    args = parser.parse_args()

    global REFINE_MODE
    REFINE_MODE = args.refine

    print("=" * 78)
    print("      AerX Labs — Threat-Aware Tactical Mission Planning Engine")
    print("              Advanced Aerospace Systems Challenge 2026")
    print("=" * 78)

    if args.target == "region":
        if args.lat is None or args.lon is None:
            parser.error("region requires --lat and --lon, e.g. "
                         "python main.py region --lat 34.1 --lon 74.8 --label kashmir")
        run_custom_region(args.lat, args.lon, args.label, quick=args.quick)
    else:
        run_region_preset(args.target, quick=args.quick)

    print("\n[COMPLETE] Mission planning session finished.")


if __name__ == "__main__":
    main()
