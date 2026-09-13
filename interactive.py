"""
AerX Labs — Interactive Radar-Placement Mission Planner
interactive.py

A click-to-place desktop tool for the real Copernicus DEM regions. Open a region,
drop / remove radar and SAM sites anywhere on the terrain, pick an aircraft, and
re-plan the survivable ingress + egress route on demand.

Run it (from D:\\Projects\\AerXlabs\\AerX_Prototype):

    .venv\\Scripts\\python.exe interactive.py ghats
    .venv\\Scripts\\python.exe interactive.py everest
    .venv\\Scripts\\python.exe interactive.py region --lat 34.1 --lon 74.8

Controls (shown on screen):
    Left-click   add a radar at the cursor        Right-click  remove the nearest site
    Enter/Space  re-plan the route                c            clear all sites
    r            reset to the auto-placed sites    a            cycle aircraft
    + / -        grow / shrink the next radar      s            toggle SAM / radar for new sites
"""

import argparse

# Interactive GUI backend BEFORE pyplot is imported.
from src.render_env import configure_matplotlib
configure_matplotlib(interactive=True)

import numpy as np                                  # noqa: E402
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.colors import LightSource           # noqa: E402

from src.risk_field import (                         # noqa: E402
    load_mission_scenario, set_custom_dem, get_last_region_scale,
    compute_risk_slice, HazardSite,
)
from src.terrain import download_copernicus_tile, find_relief_window   # noqa: E402
from src.planning_grid import PlanningGrid3D, CostWeights              # noqa: E402
from src.dstar_lite import DStarLite3D                                 # noqa: E402
from src.smoothing import smooth_trajectory                           # noqa: E402
from src.route_metrics import compute_route_metrics                   # noqa: E402
from src.aircraft import (get_aircraft, RegionScale, realize_sortie,          # noqa: E402
                          list_aircraft, kinematic_limits_for_grid)
from src.rrt_star import refine_with_rrt_star, RRTStarConfig                  # noqa: E402

RISK_SENS = 0.05  # survivability sensitivity lambda (matches the engine)


def load_region(target, lat=None, lon=None):
    """Return (scenario_key, terrain, hazards, start, goal, z_layers, title)."""
    if target == "everest":
        return ("dem",) + load_mission_scenario("dem")[:6]
    if target == "ghats":
        return ("ghats",) + load_mission_scenario("ghats")[:6]
    if target == "region":
        if lat is None or lon is None:
            raise SystemExit("region needs --lat and --lon")
        tile = download_copernicus_tile(lat, lon)
        row0, col0 = find_relief_window(tile)
        set_custom_dem(tile, row0, col0,
                       f"Custom ({lat:.3f}, {lon:.3f})",
                       "Interactive custom region.")
        return ("custom",) + load_mission_scenario("custom")[:6]
    raise SystemExit(f"unknown target '{target}'")


class InteractivePlanner:
    def __init__(self, terrain, hazards, start, goal, z_layers, title, aircraft_key="scout_heli"):
        self.terrain = np.asarray(terrain, dtype=float)
        self.ny, self.nx = self.terrain.shape
        self.auto_hazards = [self._clone(h) for h in hazards]
        self.hazards = [self._clone(h) for h in hazards]
        self.start = np.asarray(start, dtype=float)
        self.goal = np.asarray(goal, dtype=float)
        self.z_layers = np.asarray(z_layers, dtype=float)
        self.title = title
        self.scale = RegionScale(**get_last_region_scale())
        self.aircraft_keys = list_aircraft()
        self.aircraft = get_aircraft(aircraft_key)
        self.eval_alt = float(self.z_layers[len(self.z_layers) // 3])

        self.new_radius = 0.55 * max(self.nx, self.ny)
        self.new_is_sam = False
        self.routes = None          # dict with ingress/egress after a solve
        self.info = "Left-click to add a radar, then press Enter to plan."

        self.fig, self.ax = plt.subplots(figsize=(12.5, 8.4))
        try:
            self.fig.canvas.manager.set_window_title("AerX — Interactive Radar Planner")
        except Exception:
            pass
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._render()

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _clone(h):
        return HazardSite(x=h.x, y=h.y, z=h.z, mast_height=h.mast_height, radius=h.radius,
                          intensity=h.intensity, requires_los=h.requires_los,
                          decay_type=h.decay_type, name=h.name, threat_type=h.threat_type,
                          min_detect_alt=h.min_detect_alt, azimuth_center=h.azimuth_center,
                          azimuth_width=h.azimuth_width, elev_angle_min=h.elev_angle_min,
                          elev_angle_max=h.elev_angle_max)

    def _grid_params(self):
        relief = float(self.terrain.max() - self.terrain.min())
        if relief > 400.0:
            return 25.0, relief + 120.0, 1.0
        return 15.0, 220.0, 0.85

    def _add_radar(self, x, y):
        n = sum(1 for h in self.hazards)
        if self.new_is_sam:
            self.hazards.append(HazardSite(
                x=x, y=y, z=None, mast_height=8.0, radius=self.new_radius * 1.1,
                intensity=0.9, requires_los=True, decay_type="radar", threat_type="sam",
                min_detect_alt=float(self.terrain.min()) + 90.0, name=f"SAM {n+1}"))
        else:
            self.hazards.append(HazardSite(
                x=x, y=y, z=None, mast_height=12.0, radius=self.new_radius,
                intensity=0.95, requires_los=True, decay_type="radar", threat_type="radar",
                name=f"Radar {n+1}"))

    def _remove_nearest(self, x, y):
        if not self.hazards:
            return
        d = [(h.x - x) ** 2 + (h.y - y) ** 2 for h in self.hazards]
        self.hazards.pop(int(np.argmin(d)))

    # ------------------------------------------------------------------- events
    def _on_click(self, event):
        if event.inaxes is not self.ax or event.xdata is None:
            return
        x = float(np.clip(event.xdata, 0, self.nx - 1))
        y = float(np.clip(event.ydata, 0, self.ny - 1))
        if event.button == 1:
            self._add_radar(x, y)
            self.info = f"Added {'SAM' if self.new_is_sam else 'radar'} at ({x:.0f}, {y:.0f}). Enter to re-plan."
        elif event.button == 3:
            self._remove_nearest(x, y)
            self.info = "Removed nearest site. Enter to re-plan."
        self.routes = None
        self._render()

    def _on_key(self, event):
        k = (event.key or "").lower()
        if k in ("enter", " "):
            self._solve()
        elif k == "c":
            self.hazards = []; self.routes = None
            self.info = "Cleared all sites."; self._render()
        elif k == "r":
            self.hazards = [self._clone(h) for h in self.auto_hazards]; self.routes = None
            self.info = "Reset to auto-placed sites."; self._render()
        elif k == "a":
            i = (self.aircraft_keys.index(self.aircraft.key) + 1) % len(self.aircraft_keys)
            self.aircraft = get_aircraft(self.aircraft_keys[i])
            self.info = f"Aircraft: {self.aircraft.name}"; self._render()
        elif k in ("+", "="):
            self.new_radius *= 1.15; self.info = f"New-radar radius: {self.new_radius:.0f}"; self._render()
        elif k == "-":
            self.new_radius /= 1.15; self.info = f"New-radar radius: {self.new_radius:.0f}"; self._render()
        elif k == "s":
            self.new_is_sam = not self.new_is_sam
            self.info = f"New sites are now: {'SAM' if self.new_is_sam else 'radar'}"; self._render()

    # -------------------------------------------------------------------- solve
    def _solve(self):
        self.info = "Planning… (this takes a few seconds)"
        self._render()
        self.fig.canvas.draw(); self.fig.canvas.flush_events()

        min_agl, max_agl, climb = self._grid_params()
        grid = PlanningGrid3D(terrain=self.terrain, z_layers=self.z_layers, min_agl=min_agl,
                              max_agl=max_agl, max_climb_slope=climb, hazards=self.hazards,
                              weights=CostWeights.balanced())
        ing = self._plan_leg(grid, self.start, self.goal)
        egr = self._plan_leg(grid, self.goal, self.start)
        if ing is None:
            self.routes = None
            self.info = "No feasible route with these threats — move/remove some sites."
            self._render(); return

        legs = [ing] + ([egr] if egr is not None else [])
        rs = realize_sortie(legs, self.terrain, self.scale, self.aircraft)
        ri = compute_route_metrics(ing, self.terrain, self.hazards).integrated_risk
        re = compute_route_metrics(egr, self.terrain, self.hazards).integrated_risk if egr is not None else 0.0
        surv = float(np.clip(np.exp(-RISK_SENS * (ri + re)), 0.0, 1.0))
        self.routes = {"ingress": ing, "egress": egr}
        self.info = (f"{self.aircraft.name}: {rs.distance_km:.1f} km · {rs.time_min:.1f} min · "
                     f"{rs.fuel_kg:.0f} kg ({rs.fuel_pct:.0f}%) · survivability {surv*100:.0f}% · "
                     f"min AGL {rs.min_agl_m:.0f} m")
        self._render()

    def _plan_leg(self, grid, a, b):
        planner = DStarLite3D(grid)
        path = planner.plan(a, b)
        if path is None:
            return None
        coords = planner.get_path_coordinates(path)
        # Aircraft-aware RRT* refinement -> flyable trajectory for the chosen airframe.
        c = RRTStarConfig(max_iterations=600)
        c.max_turn_deg, c.max_climb_rate = kinematic_limits_for_grid(self.aircraft, self.scale, c.step_size)
        refined = refine_with_rrt_star(coords, self.terrain, self.hazards, c, risk_sampler=grid.risk_at)
        if refined is not None and len(refined) >= 2:
            coords = refined
        return smooth_trajectory(coords, self.terrain, num_points=120, min_agl=15.0,
                                 hazards=self.hazards, risk_sampler=grid.risk_at)

    # ------------------------------------------------------------------- render
    def _render(self):
        self.ax.clear()
        # Shaded terrain backdrop.
        ls = LightSource(azdeg=315, altdeg=45)
        shaded = ls.shade(self.terrain, cmap=plt.cm.terrain, vert_exag=2.0, blend_mode="soft")
        self.ax.imshow(shaded, origin="lower", extent=[0, self.nx, 0, self.ny])
        # Live risk heatmap (coarse, LOS-masked) as a translucent overlay.
        try:
            risk = compute_risk_slice(self.terrain, self.hazards, self.eval_alt,
                                      apply_los=True, subsample_step=2, los_samples=10)
            self.ax.imshow(risk, origin="lower", extent=[0, self.nx, 0, self.ny],
                           cmap="Reds", alpha=0.45, vmin=0, vmax=1)
        except Exception:
            pass
        # Threats.
        for h in self.hazards:
            col = "#FF3D6E" if h.threat_type != "sam" else "#FFB020"
            self.ax.scatter([h.x], [h.y], s=70, marker="X", c=col, edgecolors="white",
                            linewidths=1.2, zorder=6)
            circ = plt.Circle((h.x, h.y), h.radius, color=col, fill=False, ls="--",
                              lw=0.9, alpha=0.55, zorder=5)
            self.ax.add_patch(circ)
        # Routes.
        if self.routes:
            ing = self.routes["ingress"]
            self.ax.plot(ing[:, 0], ing[:, 1], "-", color="#39FF9E", lw=2.6, zorder=8, label="ingress")
            if self.routes.get("egress") is not None:
                eg = self.routes["egress"]
                self.ax.plot(eg[:, 0], eg[:, 1], "--", color="#3DE1FF", lw=2.2, zorder=8, label="egress")
            self.ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
        # Start / goal.
        self.ax.scatter([self.start[0]], [self.start[1]], s=110, marker="o", c="#2E9BFF",
                        edgecolors="white", zorder=9)
        self.ax.scatter([self.goal[0]], [self.goal[1]], s=130, marker="*", c="#FFD23F",
                        edgecolors="black", zorder=9)
        self.ax.annotate("BASE", (self.start[0], self.start[1]), color="white", fontsize=8,
                         xytext=(4, 4), textcoords="offset points")
        self.ax.annotate("TARGET", (self.goal[0], self.goal[1]), color="#FFD23F", fontsize=8,
                         xytext=(4, 4), textcoords="offset points")

        self.ax.set_xlim(0, self.nx); self.ax.set_ylim(0, self.ny)
        self.ax.set_title(self.title, fontsize=11)
        self.fig.suptitle(self.info, y=0.985, fontsize=10, color="#0A5E75")
        controls = ("L-click add · R-click remove · Enter plan · c clear · r reset · "
                    f"a aircraft [{self.aircraft.name}] · +/- radius[{self.new_radius:.0f}] · "
                    f"s type[{'SAM' if self.new_is_sam else 'radar'}]")
        self.ax.set_xlabel(controls, fontsize=8)
        self.fig.canvas.draw_idle()

    def run(self):
        plt.show()


def main():
    ap = argparse.ArgumentParser(description="AerX interactive radar-placement planner.")
    ap.add_argument("target", choices=["everest", "ghats", "region"])
    ap.add_argument("--lat", type=float)
    ap.add_argument("--lon", type=float)
    ap.add_argument("--aircraft", default="scout_heli")
    args = ap.parse_args()

    print("Loading region… (first run of a new region downloads the DEM tile)")
    scen, terrain, hazards, start, goal, z_layers, title = load_region(args.target, args.lat, args.lon)
    print(f"Loaded: {title}")
    print("Opening interactive window — place radars, press Enter to plan.")
    InteractivePlanner(terrain, hazards, start, goal, z_layers, title, args.aircraft).run()


if __name__ == "__main__":
    main()
