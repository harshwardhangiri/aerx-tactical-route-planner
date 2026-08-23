"""
AerX Labs — Trade-off & Replanning Analysis
src/analysis.py

Two research-grade studies that the concept document and the reference papers
call for but the base pipeline did not provide:

1. Pareto sensitivity sweep — sweeps the risk weight w_risk and traces the
   distance-vs-exposure trade-off front. This turns the "which weight?" question
   into a defensible curve instead of three hand-picked presets (the sensitivity
   analysis the master context explicitly asks for).

2. Incremental-replanning demo — plans a route, injects a pop-up threat on it,
   then repairs the route with D* Lite's incremental update and contrasts the
   node-expansion cost against a full re-plan. This is the concrete justification
   for choosing D* Lite over one-shot A*.
"""

import os
import time
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt

from src.risk_field import HazardSite, load_mission_scenario, compute_risk_slice
from src.planning_grid import CostWeights, PlanningGrid3D
from src.dstar_lite import DStarLite3D
from src.smoothing import smooth_trajectory
from src.route_metrics import compute_route_metrics
from src.scoring import compute_route_score

# Shared theme.
from src.visualization import (
    BG_DEEP, BG_PANEL, TEXT_HI, TEXT_LO, ACCENT, PANEL_EDGE, GRID_INK,
    _TERRAIN_CMAP, _RISK_CMAP, _hillshade_rgb, stamp_figure,
)


# --------------------------------------------------------------------------- #
#  Pareto sensitivity sweep
# --------------------------------------------------------------------------- #
def pareto_sweep(scenario_key: str, risk_weights: Optional[List[float]] = None) -> Dict:
    """Plan a route for each risk weight and record its distance/exposure/score."""
    if risk_weights is None:
        risk_weights = [0.0, 1.0, 2.0, 4.0, 6.0, 10.0]

    terrain, hazards, start, goal, z_layers, title, _ = load_mission_scenario(scenario_key)
    grid = PlanningGrid3D(terrain=terrain, z_layers=z_layers, min_agl=15.0, max_agl=220.0,
                          max_climb_slope=0.85, hazards=hazards)

    def _snap(p):
        return grid.node_to_pos(grid.pos_to_node(p))
    direct = float(np.linalg.norm(_snap(goal) - _snap(start)))

    rows = []
    for wr in risk_weights:
        grid.weights = CostWeights(w_distance=1.0, w_risk=wr, w_energy=1.0, w_altitude=0.3)
        path = DStarLite3D(grid).plan(start, goal)
        if path is None:
            continue
        coords = np.array([grid.node_to_pos(n) for n in path])
        smooth = smooth_trajectory(coords, terrain, num_points=120, min_agl=15.0, hazards=hazards,
                                   risk_sampler=grid.risk_at)
        m = compute_route_metrics(smooth, terrain, hazards)
        s = compute_route_score(m, direct_euclidean_distance=direct)
        rows.append({
            "w_risk": wr,
            "distance": m.total_distance,
            "detour_pct": (m.total_distance - direct) / direct * 100.0,
            "integrated_risk": m.integrated_risk,
            "survivability": s.survivability_score * 100.0,
            "composite": s.composite_score,
        })
    return {"scenario": scenario_key, "title": title, "direct": direct, "rows": rows}


def render_pareto(sweep: Dict, save_path: str) -> None:
    """Render the distance-vs-exposure trade-off front from a pareto_sweep result."""
    rows = sweep["rows"]
    if not rows:
        return
    wr = np.array([r["w_risk"] for r in rows])
    dist = np.array([r["distance"] for r in rows])
    risk = np.array([r["integrated_risk"] for r in rows])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6), facecolor=BG_DEEP)
    fig.suptitle(f"Risk-Weight Sensitivity & Distance–Exposure Trade-off  |  {sweep['title']}",
                 color=ACCENT, fontsize=13, weight="bold")

    # Left: the Pareto front (distance vs integrated risk), colored by w_risk.
    axL.set_facecolor(BG_PANEL)
    sc = axL.scatter(dist, risk, c=wr, cmap="viridis", s=90, edgecolors="white", linewidth=0.8, zorder=5)
    axL.plot(dist, risk, color=TEXT_LO, lw=1.0, ls="--", alpha=0.5, zorder=4)
    for r in rows:
        axL.annotate(f"w={r['w_risk']:g}", (r["distance"], r["integrated_risk"]),
                     textcoords="offset points", xytext=(6, 4), fontsize=7, color=TEXT_LO)
    axL.set_xlabel("Total flight distance (m)", color=TEXT_LO, fontsize=9)
    axL.set_ylabel("Integrated threat exposure", color=TEXT_LO, fontsize=9)
    axL.set_title("Pareto front: buying safety with distance", color=TEXT_HI, fontsize=10, weight="bold")
    cb = fig.colorbar(sc, ax=axL, shrink=0.85)
    cb.set_label("risk weight  w_risk", color=TEXT_LO, fontsize=8)
    cb.ax.tick_params(colors=TEXT_LO, labelsize=7)

    # Right: how each objective responds to w_risk.
    axR.set_facecolor(BG_PANEL)
    axR.plot(wr, risk, "-o", color="#FF3D6E", label="Integrated risk", lw=2)
    axR.set_xlabel("risk weight  w_risk", color=TEXT_LO, fontsize=9)
    axR.set_ylabel("Integrated threat exposure", color="#FF3D6E", fontsize=9)
    axR.tick_params(axis="y", colors="#FF3D6E")
    ax2 = axR.twinx()
    ax2.plot(wr, dist, "-s", color="#22D3EE", label="Distance", lw=2)
    ax2.set_ylabel("Distance (m)", color="#22D3EE", fontsize=9)
    ax2.tick_params(axis="y", colors="#22D3EE")
    axR.set_title("Diminishing returns of increasing w_risk", color=TEXT_HI, fontsize=10, weight="bold")

    for ax in (axL, axR):
        ax.tick_params(colors=TEXT_LO, labelsize=7.5)
        for sp in ax.spines.values():
            sp.set_color(PANEL_EDGE)
        ax.grid(True, ls=":", alpha=0.25, color=GRID_INK)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    stamp_figure(fig, os.path.basename(save_path))
    fig.savefig(save_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor="none")
    print(f"  [OK] Saved Pareto sensitivity study: {save_path}", flush=True)
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  Incremental-replanning demo
# --------------------------------------------------------------------------- #
def replanning_demo(scenario_key: str, popup: Optional[HazardSite] = None) -> Dict:
    """
    Plan a route, inject a pop-up threat onto it, then repair the route with D*
    Lite's incremental update; compare the expansion cost against a full re-plan.
    """
    terrain, hazards, start, goal, z_layers, title, _ = load_mission_scenario(scenario_key)
    grid = PlanningGrid3D(terrain=terrain, z_layers=z_layers, min_agl=15.0, max_agl=220.0,
                          max_climb_slope=0.85, hazards=hazards, weights=CostWeights.balanced())

    planner = DStarLite3D(grid)
    path0 = planner.plan(start, goal)
    coords0 = np.array([grid.node_to_pos(n) for n in path0])
    smooth0 = smooth_trajectory(coords0, terrain, num_points=120, min_agl=15.0, hazards=hazards,
                                risk_sampler=grid.risk_at)

    # Default pop-up threat: drop it on the middle of the current route.
    if popup is None:
        mid = coords0[len(coords0) // 2]
        popup = HazardSite(x=float(mid[0]), y=float(mid[1]), z=None, mast_height=12.0,
                           radius=42.0, intensity=0.97, requires_los=True,
                           decay_type="radar", name="Pop-up Threat")

    affected = grid.add_hazard(popup)

    # Incremental repair (reuse search tree).
    t0 = time.time()
    path_inc = planner.replan_after_hazard_change(start, affected_nodes=affected)
    t_inc = time.time() - t0
    exp_inc = planner.last_expansions

    # Full re-plan from scratch on the same updated grid, for the expansion-count
    # comparison (we only need its timing/expansions, not the path itself).
    fresh = DStarLite3D(grid)
    t0 = time.time()
    fresh.plan(start, goal)
    t_full = time.time() - t0
    exp_full = fresh.last_expansions

    smooth_inc = (smooth_trajectory(np.array([grid.node_to_pos(n) for n in path_inc]),
                                    terrain, num_points=120, min_agl=15.0, hazards=grid.hazards,
                                    risk_sampler=grid.risk_at)
                  if path_inc else None)

    return {
        "scenario": scenario_key, "title": title,
        "terrain": terrain, "hazards": hazards, "popup": popup,
        "start": start, "goal": goal,
        "route_before": smooth0, "route_after": smooth_inc,
        "affected": len(affected),
        "exp_incremental": exp_inc, "time_incremental": t_inc,
        "exp_full": exp_full, "time_full": t_full,
    }


def render_replanning(demo: Dict, save_path: str) -> None:
    """Before/after map of an incremental re-plan around a pop-up threat."""
    terrain = demo["terrain"]
    ny, nx = terrain.shape
    hazards = demo["hazards"]
    popup = demo["popup"]
    start, goal = demo["start"], demo["goal"]

    fig, ax = plt.subplots(figsize=(11, 9.4), facecolor=BG_DEEP)
    fig.subplots_adjust(top=0.88, bottom=0.07, left=0.08, right=0.97)
    ax.set_facecolor(BG_PANEL)
    extent = [0, nx - 1, 0, ny - 1]
    ax.imshow(_hillshade_rgb(terrain, _TERRAIN_CMAP, vert=4.0), origin="lower", extent=extent, zorder=1)

    eval_alt = float(np.mean([p[2] for p in demo["route_before"]]))
    risk = compute_risk_slice(terrain, hazards + [popup], altitude_asl=eval_alt, apply_los=True, subsample_step=1)
    ax.imshow(np.ma.masked_less(risk, 0.02), origin="lower", extent=extent, cmap=_RISK_CMAP,
              alpha=0.7, vmin=0, vmax=1, zorder=2)

    import matplotlib.patches as patches
    for h in hazards:
        p = h.get_position(terrain)
        ax.scatter(p[0], p[1], color="#FF3D6E", s=80, marker="^", edgecolors="white", zorder=8)
    pp = popup.get_position(terrain)
    ax.scatter(pp[0], pp[1], color="#FFD23F", s=220, marker="X", edgecolors="black", linewidth=1.4,
               zorder=10, label="Pop-up threat")
    ax.add_patch(patches.Circle((pp[0], pp[1]), popup.radius, color="#FFD23F", fill=False, ls="--", lw=1.5, alpha=0.8))

    rb = demo["route_before"]
    ax.plot(rb[:, 0], rb[:, 1], color="#8892A6", lw=2.4, ls="--", label="Original route", zorder=6)
    if demo["route_after"] is not None:
        ra = demo["route_after"]
        ax.plot(ra[:, 0], ra[:, 1], color="#39FF9E", lw=2.8, label="Incrementally re-planned", zorder=7)

    ax.scatter(start[0], start[1], color="#2E9BFF", s=130, marker="o", edgecolors="white", zorder=9)
    ax.scatter(goal[0], goal[1], color="#FFD23F", s=180, marker="*", edgecolors="black", zorder=9)

    speedup = demo["exp_full"] / max(demo["exp_incremental"], 1)
    ax.set_title(f"D* Lite Incremental Re-planning — {demo['title']}",
                 color=TEXT_HI, fontsize=11, weight="bold", pad=10)
    fig.text(
        0.5, 0.925,
        f"Pop-up threat repaired by touching {demo['affected']} nodes    "
        f"incremental {demo['exp_incremental']} expansions ({demo['time_incremental']*1e3:.0f} ms)  vs  "
        f"full re-plan {demo['exp_full']} ({demo['time_full']*1e3:.0f} ms)  =  {speedup:.1f}x fewer",
        ha="center", va="top", color=ACCENT, fontsize=9)
    ax.set_xlabel("X (m)", color=TEXT_LO); ax.set_ylabel("Y (m)", color=TEXT_LO)
    ax.tick_params(colors=TEXT_LO, labelsize=7)
    ax.legend(loc="upper left", fontsize=8.5, facecolor=BG_PANEL, edgecolor=PANEL_EDGE, labelcolor=TEXT_HI)
    for sp in ax.spines.values():
        sp.set_color(PANEL_EDGE)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    stamp_figure(fig, os.path.basename(save_path))
    fig.savefig(save_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor="none")
    print(f"  [OK] Saved incremental-replanning demo: {save_path}", flush=True)
    plt.close(fig)
