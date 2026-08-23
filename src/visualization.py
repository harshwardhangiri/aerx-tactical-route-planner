"""
AerX Labs — Tactical Visualization & Mission Scorecard Engine
src/visualization.py

Renders the four-panel tactical mission dashboard and the standalone
performance scorecard.

Panels
------
1. 3D battlefield: elevation-shaded terrain surface with a projected contour
   "map" on the floor, glowing dual-layer flight tubes, ground-track shadows +
   vertical drop-lines for depth, threat masts and projected radar-coverage
   rings, and start/goal markers.
2. 2D threat field: hill-shaded terrain under a translucent risk heatmap with
   detection rings, terrain-masked radar shadows and the three route tracks.
3. Nap-of-the-Earth profile: each route on its own along-track distance axis,
   the AGL clearance band, the minimum-clearance floor, and threat-exposure
   shading along the balanced route.
4. Monte-Carlo robustness: mean composite score with P5-P95 whiskers over
   benchmark survivability zones, plus the raw sample swarm.

Public API (unchanged): InteractiveMissionDashboard, render_tactical_dashboard,
render_mission_scorecard.
"""

import os
from datetime import datetime
from typing import Callable, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LightSource, LinearSegmentedColormap
from matplotlib.widgets import Button

from src.los import terrain_height
from src.risk_field import compute_risk_slice, compute_point_risk


# --------------------------------------------------------------------------- #
#  Theme
# --------------------------------------------------------------------------- #
BG_DEEP = "#0B1020"       # figure background
BG_PANEL = "#0E1730"      # axes background
GRID_INK = "#24304F"
TEXT_HI = "#E8EEF7"
TEXT_LO = "#9FB0C8"
ACCENT = "#38E1FF"
PANEL_EDGE = "#2C3A5E"

PROFILE_STYLES = {
    "Distance-focused": {"color": "#22D3EE", "glow": "#0A4A57", "z": 11},
    "Risk-focused":     {"color": "#FF3D6E", "glow": "#4A0A1C", "z": 12},
    "Balanced":         {"color": "#39FF9E", "glow": "#0A4A2C", "z": 13},
}

# Warm cartographic elevation ramp (low = teal lowland, high = pale ridge).
_TERRAIN_CMAP = LinearSegmentedColormap.from_list(
    "aerx_terrain",
    ["#123C4A", "#1E5F52", "#3E7A3A", "#7E8F3C", "#B79A55", "#D9C6A0", "#F2ECDD"],
)
# Risk overlay ramp (transparent safe -> hot threat), alpha baked per level.
_RISK_CMAP = LinearSegmentedColormap.from_list(
    "aerx_risk", ["#122A2A", "#F6C445", "#FF7A2F", "#E01E4F"]
)


def stamp_figure(fig, note: str = "") -> None:
    """Stamp a figure with its generation timestamp so a saved PNG is never
    mistaken for a stale one from a previous run."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    label = f"generated {ts}" + (f"  ·  {note}" if note else "")
    fig.text(0.995, 0.006, label, ha="right", va="bottom",
             color=TEXT_LO, fontsize=7, family="monospace", alpha=0.75)


def _style_axis(ax, title: str) -> None:
    """Apply the shared dark 2D panel styling."""
    ax.set_facecolor(BG_PANEL)
    ax.set_title(title, color=TEXT_HI, fontsize=10.5, weight="bold", pad=8)
    for spine in ax.spines.values():
        spine.set_color(PANEL_EDGE)
    ax.tick_params(colors=TEXT_LO, labelsize=7)


def _hillshade_rgb(terrain: np.ndarray, cmap, azdeg=315, altdeg=45, vert=6.0) -> np.ndarray:
    """Return an RGB image of the terrain shaded by a synthetic light source."""
    ls = LightSource(azdeg=azdeg, altdeg=altdeg)
    return ls.shade(
        terrain, cmap=cmap, blend_mode="soft",
        vert_exag=vert, dx=1, dy=1,
    )


# Small per-profile lateral offset (grid units) so overlapping routes render as
# distinct parallel tracks instead of one hiding another. When two profiles pick
# the same route (common when balanced and risk-focused converge), this is the
# only reason all three stay visible.
_ROUTE_OFFSET = {"Distance-focused": 1.3, "Balanced": 0.0, "Risk-focused": -1.3}


def _offset_xy(coords: np.ndarray, d: float) -> np.ndarray:
    """Shift a polyline sideways by `d` (perpendicular to its local direction)."""
    c = np.asarray(coords, dtype=float).copy()
    if len(c) < 2 or abs(d) < 1e-9:
        return c
    tang = np.gradient(c[:, :2], axis=0)
    normal = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
    mag = np.linalg.norm(normal, axis=1, keepdims=True)
    mag[mag < 1e-9] = 1.0
    c[:, :2] = c[:, :2] + d * (normal / mag)
    return c


def _cumulative_distance(coords: np.ndarray) -> np.ndarray:
    """Along-track cumulative distance for an (N,3) polyline."""
    d = np.zeros(len(coords))
    if len(coords) > 1:
        seg = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        d[1:] = np.cumsum(seg)
    return d


# --------------------------------------------------------------------------- #
#  3D battlefield panel
# --------------------------------------------------------------------------- #
def _render_panel_3d(ax, terrain, hazards, start_pos, goal_pos, routes_data):
    ny, nx = terrain.shape
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny))

    z_floor = float(terrain.min()) - 12.0
    z_ceil = max(float(terrain.max()),
                 max((r["smooth_coords"][:, 2].max() for r in routes_data.values()),
                     default=terrain.max())) + 20.0

    ax.set_facecolor(BG_PANEL)
    ax.set_box_aspect((1.0, 1.0, 0.55))
    ax.view_init(elev=34, azim=-58)

    # Shaded elevation surface.
    rgb = _hillshade_rgb(terrain, _TERRAIN_CMAP)
    ax.plot_surface(
        X, Y, terrain, facecolors=rgb, rstride=2, cstride=2,
        linewidth=0, antialiased=False, shade=False, alpha=0.97, zorder=1,
    )

    # Filled contour "map" projected on the floor for readability.
    ax.contourf(X, Y, terrain, levels=14, zdir="z", offset=z_floor,
                cmap=_TERRAIN_CMAP, alpha=0.35)

    # Threats: floor coverage ring, mast, apex marker.
    theta = np.linspace(0, 2 * np.pi, 60)
    for h in hazards:
        pos = h.get_position(terrain)
        ring_x = pos[0] + h.radius * np.cos(theta)
        ring_y = pos[1] + h.radius * np.sin(theta)
        ax.plot(ring_x, ring_y, z_floor, color="#FF3D6E", lw=1.3, alpha=0.75, zorder=3)
        ax.plot([pos[0], pos[0]], [pos[1], pos[1]], [z_floor, pos[2]],
                color="#FF6B8A", lw=1.6, alpha=0.9, zorder=4)
        ax.scatter(pos[0], pos[1], pos[2], color="#FF3D6E", s=90, marker="^",
                   edgecolors="white", linewidth=1.0, depthshade=False, zorder=6)

    # Flight tubes with ground-track shadow + vertical depth stems. A small
    # per-profile lateral offset keeps all three visible even when two routes
    # coincide.
    for name, r in routes_data.items():
        c = _offset_xy(r["smooth_coords"], _ROUTE_OFFSET.get(name, 0.0))
        style = PROFILE_STYLES.get(name, {"color": "white", "glow": "black", "z": 10})
        # Ground-track shadow.
        ax.plot(c[:, 0], c[:, 1], z_floor, color="#05070D", lw=2.4, alpha=0.4, zorder=2)
        # Vertical drop-lines every few samples (depth cue).
        for k in range(0, len(c), 12):
            ax.plot([c[k, 0], c[k, 0]], [c[k, 1], c[k, 1]], [z_floor, c[k, 2]],
                    color=style["color"], lw=0.6, alpha=0.18, zorder=5)
        # Dark contrast tube + glowing core (ingress = solid).
        ax.plot(c[:, 0], c[:, 1], c[:, 2], color=style["glow"], lw=5.5,
                solid_capstyle="round", zorder=style["z"])
        ax.plot(c[:, 0], c[:, 1], c[:, 2], color=style["color"], lw=2.8,
                solid_capstyle="round", label=name, zorder=style["z"] + 1)

        # Egress (return) route = dashed, same colour.
        eg = r.get("egress_smooth_coords")
        if eg is not None:
            eg = _offset_xy(eg, _ROUTE_OFFSET.get(name, 0.0))
            ax.plot(eg[:, 0], eg[:, 1], eg[:, 2], color=style["color"], lw=1.7,
                    ls=(0, (5, 3)), alpha=0.9, zorder=style["z"] + 1)

    ax.scatter(*start_pos, color="#2E9BFF", s=150, marker="o",
               edgecolors="white", linewidth=1.4, depthshade=False, zorder=20)
    ax.scatter(*goal_pos, color="#FFD23F", s=230, marker="*",
               edgecolors="black", linewidth=1.2, depthshade=False, zorder=20)

    ax.set_xlim(0, nx - 1)
    ax.set_ylim(0, ny - 1)
    ax.set_zlim(z_floor, z_ceil)
    ax.set_title("3D Battlefield — Terrain, Radar Coverage & Flight Tubes (drag to orbit)",
                 color=TEXT_HI, fontsize=10.5, weight="bold", pad=0)
    ax.set_xlabel("X (m)", color=TEXT_LO, fontsize=8, labelpad=-4)
    ax.set_ylabel("Y (m)", color=TEXT_LO, fontsize=8, labelpad=-4)
    ax.set_zlabel("Alt ASL (m)", color=TEXT_LO, fontsize=8, labelpad=-4)
    ax.tick_params(colors=TEXT_LO, labelsize=6.5, pad=-2)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_facecolor((0.05, 0.08, 0.16, 0.75))
        pane.pane.set_edgecolor(GRID_INK)
        pane._axinfo["grid"]["color"] = (0.2, 0.26, 0.4, 0.5)
    ax.legend(loc="upper left", fontsize=7.5, facecolor=BG_PANEL,
              edgecolor=PANEL_EDGE, labelcolor=TEXT_HI, framealpha=0.85)
    if any(r.get("egress_smooth_coords") is not None for r in routes_data.values()):
        ax.text2D(0.02, 0.80, "solid = ingress   dashed = egress",
                  transform=ax.transAxes, color=TEXT_LO, fontsize=7, style="italic")


# --------------------------------------------------------------------------- #
#  2D threat-field panel
# --------------------------------------------------------------------------- #
def _render_panel_map(ax, fig, terrain, hazards, start_pos, goal_pos,
                      routes_data, eval_altitude):
    ny, nx = terrain.shape
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny))
    extent = [0, nx - 1, 0, ny - 1]

    _style_axis(ax, f"2D Threat Field & Terrain Radar Shadows  (eval ASL = {eval_altitude:.0f} m)")

    # Hill-shaded terrain base.
    rgb = _hillshade_rgb(terrain, _TERRAIN_CMAP, vert=4.0)
    ax.imshow(rgb, origin="lower", extent=extent, zorder=1)

    # Elevation contours for orientation.
    ct = ax.contour(X, Y, terrain, levels=8, colors="#0B1020", alpha=0.25, linewidths=0.6, zorder=2)
    ax.clabel(ct, inline=True, fontsize=5.5, fmt="%d", colors="#0B1020")

    # Risk heatmap (terrain-masked) with transparency where safe.
    risk_slice = compute_risk_slice(terrain, hazards, altitude_asl=eval_altitude,
                                    apply_los=True, subsample_step=1)
    risk_masked = np.ma.masked_less(risk_slice, 0.02)
    risk_im = ax.imshow(risk_masked, origin="lower", extent=extent, cmap=_RISK_CMAP,
                        alpha=0.78, vmin=0.0, vmax=1.0, zorder=3)

    # Threats + detection rings.
    for h in hazards:
        pos = h.get_position(terrain)
        ax.scatter(pos[0], pos[1], color="#FF3D6E", s=95, marker="^",
                   edgecolors="white", linewidth=1.1, zorder=12)
        ax.add_patch(patches.Circle((pos[0], pos[1]), h.radius, color="#FF6B8A",
                                    fill=False, ls="--", lw=1.2, alpha=0.85, zorder=11))
        ax.annotate(h.name, (pos[0], pos[1]), textcoords="offset points", xytext=(6, 6),
                    fontsize=6.5, weight="bold", color="#FFD8E1",
                    bbox=dict(boxstyle="round,pad=0.2", fc=BG_PANEL, ec="none", alpha=0.75), zorder=13)

    # Route tracks (small per-profile offset so coincident routes stay distinct).
    # Ingress = solid, egress (return) = dashed, same colour.
    for name, r in routes_data.items():
        c = _offset_xy(r["smooth_coords"], _ROUTE_OFFSET.get(name, 0.0))
        style = PROFILE_STYLES.get(name, {"color": "white"})
        ax.plot(c[:, 0], c[:, 1], color="#05070D", lw=3.6, zorder=9)
        ax.plot(c[:, 0], c[:, 1], color=style["color"], lw=1.9, label=name, zorder=10)
        eg = r.get("egress_smooth_coords")
        if eg is not None:
            eg = _offset_xy(eg, _ROUTE_OFFSET.get(name, 0.0))
            ax.plot(eg[:, 0], eg[:, 1], color=style["color"], lw=1.4, ls=(0, (5, 3)),
                    alpha=0.9, zorder=10)

    ax.scatter(start_pos[0], start_pos[1], color="#2E9BFF", s=120, marker="o",
               edgecolors="white", linewidth=1.4, zorder=14, label="Start")
    ax.scatter(goal_pos[0], goal_pos[1], color="#FFD23F", s=170, marker="*",
               edgecolors="black", linewidth=1.2, zorder=14, label="Target")

    ax.set_xlim(0, nx - 1)
    ax.set_ylim(0, ny - 1)
    ax.set_xlabel("X (m)", color=TEXT_LO, fontsize=8)
    ax.set_ylabel("Y (m)", color=TEXT_LO, fontsize=8)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=6.8, ncol=2, facecolor=BG_PANEL,
              edgecolor=PANEL_EDGE, labelcolor=TEXT_HI, framealpha=0.85)
    if any(r.get("egress_smooth_coords") is not None for r in routes_data.values()):
        ax.text(0.02, 0.02, "solid = ingress   dashed = egress", transform=ax.transAxes,
                color=TEXT_HI, fontsize=7, style="italic",
                bbox=dict(boxstyle="round,pad=0.25", fc=BG_PANEL, ec="none", alpha=0.7))
    cb = fig.colorbar(risk_im, ax=ax, shrink=0.72, pad=0.02)
    cb.set_label("Threat exposure [0, 1]", color=TEXT_LO, fontsize=7.5)
    cb.ax.tick_params(colors=TEXT_LO, labelsize=6.5)
    cb.outline.set_edgecolor(PANEL_EDGE)


# --------------------------------------------------------------------------- #
#  Nap-of-the-Earth profile panel
# --------------------------------------------------------------------------- #
def _render_panel_profile(ax, terrain, hazards, routes_data, min_agl=15.0):
    _style_axis(ax, "Nap-of-the-Earth Altitude Profile & Threat Exposure")

    bal_r = routes_data.get("Balanced", list(routes_data.values())[0])
    cb = bal_r["smooth_coords"]
    d_bal = _cumulative_distance(cb)
    ground = np.array([terrain_height(terrain, p[0], p[1]) for p in cb])
    exposure = np.array([compute_point_risk(p, hazards, terrain=terrain, apply_los=True) for p in cb])

    # Ground surface + AGL clearance band for the balanced route.
    ax.fill_between(d_bal, z_bottom := (ground.min() - 15), ground,
                    color="#241A12", zorder=1)
    ax.plot(d_bal, ground, color="#7A5A3A", lw=1.4, zorder=2, label="Ground surface")
    ax.fill_between(d_bal, ground, cb[:, 2], color=PROFILE_STYLES["Balanced"]["color"],
                    alpha=0.10, zorder=2, label="Balanced AGL clearance")
    ax.plot(d_bal, ground + min_agl, color="#FFB020", lw=1.0, ls=":",
            alpha=0.8, zorder=3, label=f"Min AGL floor ({min_agl:.0f} m)")

    # Each route on its own along-track distance (a tiny altitude nudge keeps
    # coincident profiles from hiding one another).
    _z_nudge = {"Distance-focused": 1.6, "Balanced": 0.0, "Risk-focused": -1.6}
    for name, r in routes_data.items():
        c = r["smooth_coords"]
        style = PROFILE_STYLES.get(name, {"color": "white"})
        ax.plot(_cumulative_distance(c), c[:, 2] + _z_nudge.get(name, 0.0), color=style["color"],
                lw=2.2, label=f"{name}", zorder=6)

    # Threat exposure corridor along the balanced route.
    exposed = exposure > 0.05
    if np.any(exposed):
        ax.fill_between(d_bal, ground, cb[:, 2] + 15.0, where=exposed,
                        color="#FF3D6E", alpha=0.22, zorder=4,
                        label="Threat exposure (balanced)")

    ax.set_xlim(0, max(d_bal.max(), 1))
    ax.set_ylim(z_bottom, None)
    ax.set_xlabel("Along-track distance (m)", color=TEXT_LO, fontsize=8)
    ax.set_ylabel("Altitude ASL (m)", color=TEXT_LO, fontsize=8)
    ax.grid(True, ls=":", alpha=0.25, color=GRID_INK)
    ax.legend(loc="upper right", fontsize=6.6, ncol=2, facecolor=BG_PANEL,
              edgecolor=PANEL_EDGE, labelcolor=TEXT_HI, framealpha=0.85)


# --------------------------------------------------------------------------- #
#  Monte-Carlo robustness panel
# --------------------------------------------------------------------------- #
def _render_panel_montecarlo(ax, routes_data):
    _style_axis(ax, "Monte-Carlo Robustness — Full Sortie Mission Score (ingress + egress)")

    ax.axhspan(75, 100, color="#39FF9E", alpha=0.08)
    ax.axhspan(50, 75, color="#FFD23F", alpha=0.08)
    ax.axhspan(0, 50, color="#FF3D6E", alpha=0.10)
    ax.text(0.985, 87, "OPTIMAL", transform=ax.get_yaxis_transform(), ha="right",
            va="center", color="#39FF9E", fontsize=6.5, weight="bold", alpha=0.8)
    ax.text(0.985, 62, "MODERATE", transform=ax.get_yaxis_transform(), ha="right",
            va="center", color="#FFD23F", fontsize=6.5, weight="bold", alpha=0.8)
    ax.text(0.985, 25, "HIGH THREAT", transform=ax.get_yaxis_transform(), ha="right",
            va="center", color="#FF3D6E", fontsize=6.5, weight="bold", alpha=0.8)

    names = list(routes_data.keys())
    for idx, name in enumerate(names):
        mc = routes_data[name]["mc"]
        style = PROFILE_STYLES.get(name, {"color": "white"})
        color = style["color"]

        # Raw sample swarm (jittered) for distribution context.
        samples = np.asarray(mc.all_composite_scores)
        jitter = idx + np.random.uniform(-0.16, 0.16, size=len(samples))
        ax.scatter(jitter, samples, s=8, color=color, alpha=0.20, zorder=4, edgecolors="none")

        ax.bar(idx, mc.mean_composite_score, width=0.5, color=color, alpha=0.45,
               edgecolor=color, linewidth=1.3, zorder=5)
        err_low = max(0, mc.mean_composite_score - mc.p5_worst_case_score)
        err_high = max(0, mc.p95_best_case_score - mc.mean_composite_score)
        ax.errorbar(idx, mc.mean_composite_score, yerr=[[err_low], [err_high]], fmt="o",
                    color="white", ecolor="white", elinewidth=1.8, capsize=6,
                    capthick=1.6, markersize=4, zorder=8)
        ax.text(idx, min(mc.p95_best_case_score + err_high + 4, 112),
                f"{mc.mean_composite_score:.1f}\nSurv {mc.mean_survivability*100:.0f}%",
                ha="center", va="bottom", color=color, fontsize=7.5, weight="bold",
                bbox=dict(boxstyle="round,pad=0.22", fc=BG_PANEL, ec=color, lw=1.0, alpha=0.9),
                zorder=9)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, color=TEXT_HI, fontsize=8, weight="bold")
    ax.set_ylabel("Composite mission score [0-100]", color=TEXT_LO, fontsize=8)
    ax.set_ylim(0, 122)
    ax.set_xlim(-0.6, len(names) - 0.4)
    ax.grid(True, ls=":", alpha=0.22, color=GRID_INK, axis="y")


# --------------------------------------------------------------------------- #
#  Interactive dashboard
# --------------------------------------------------------------------------- #
class InteractiveMissionDashboard:
    """Interactive 4-panel tactical dashboard with scenario switching."""

    def __init__(
        self,
        scenario_runner_callback: Callable[[str], Tuple],
        initial_scenario: str = "mountain",
        interactive_controls: bool = True,
    ):
        self.runner_callback = scenario_runner_callback
        self.current_scenario = initial_scenario
        self.interactive_controls = interactive_controls

        self.fig = plt.figure(figsize=(19, 10.2), facecolor=BG_DEEP)
        try:
            self.fig.canvas.manager.set_window_title(
                "AerX Labs — Threat-Aware Tactical Mission Planning System")
        except Exception:
            pass

        # The scenario-switcher toolbar only makes sense in the live window; a
        # saved PNG is a single scenario, so static renders omit it.
        self.buttons = {}
        if interactive_controls:
            self._setup_scenario_buttons()
            self._setup_keyboard_shortcuts()
        self.update_scenario(initial_scenario)

    def _setup_scenario_buttons(self):
        scenarios = [
            ("mountain", "1  Mountain Pass"),
            ("canyon", "2  Deep Canyon"),
            ("archipelago", "3  Archipelago"),
            ("rolling", "4  Rolling Hills"),
        ]
        btn_w, btn_h, start_x, spacing = 0.15, 0.032, 0.17, 0.165
        for idx, (key, label) in enumerate(scenarios):
            ax_btn = self.fig.add_axes([start_x + idx * spacing, 0.952, btn_w, btn_h])
            btn = Button(ax_btn, label, color="#16223E", hovercolor="#28406E")
            btn.label.set_color(TEXT_HI)
            btn.label.set_fontsize(9)
            btn.label.set_weight("bold")
            btn.on_clicked((lambda sc=key: (lambda e: self.update_scenario(sc)))())
            self.buttons[key] = (btn, ax_btn)

    def _setup_keyboard_shortcuts(self):
        key_map = {"1": "mountain", "2": "canyon", "3": "archipelago", "4": "rolling"}

        def on_key(event):
            if event.key in key_map:
                self.update_scenario(key_map[event.key])

        self.fig.canvas.mpl_connect("key_press_event", on_key)

    def update_scenario(self, scenario_key: str):
        self.current_scenario = scenario_key
        for key, (btn, _) in self.buttons.items():
            active = key == scenario_key
            btn.color = ACCENT if active else "#16223E"
            btn.label.set_color(BG_DEEP if active else TEXT_HI)

        (terrain, hazards, start_pos, goal_pos, routes_data,
         scenario_title, eval_altitude) = self.runner_callback(scenario_key)

        for ax in list(self.fig.axes):
            if not any(ax is b_ax for _, (_, b_ax) in self.buttons.items()):
                ax.remove()

        self._render_panels(terrain, hazards, start_pos, goal_pos,
                            routes_data, scenario_title, eval_altitude)
        self.fig.canvas.draw_idle()

    def _render_panels(self, terrain, hazards, start_pos, goal_pos,
                       routes_data, scenario_title, eval_altitude):
        self.fig.text(0.012, 0.965, "AERX LABS", color=ACCENT, fontsize=13, weight="bold")
        self.fig.text(0.012, 0.945, "TACTICAL ENGINE", color=TEXT_LO, fontsize=8, weight="bold")
        # Scenario title sits on its own band BELOW the toolbar buttons.
        self.fig.text(0.5, 0.912, scenario_title.upper(), ha="center", color=TEXT_HI,
                      fontsize=12.5, weight="bold")

        ax1 = self.fig.add_subplot(221, projection="3d")
        ax2 = self.fig.add_subplot(222)
        ax3 = self.fig.add_subplot(223)
        ax4 = self.fig.add_subplot(224)

        _render_panel_3d(ax1, terrain, hazards, start_pos, goal_pos, routes_data)
        _render_panel_map(ax2, self.fig, terrain, hazards, start_pos, goal_pos,
                          routes_data, eval_altitude)
        _render_panel_profile(ax3, terrain, hazards, routes_data)
        _render_panel_montecarlo(ax4, routes_data)

        self.fig.subplots_adjust(top=0.885, bottom=0.07, left=0.045, right=0.965,
                                 hspace=0.30, wspace=0.17)


def render_tactical_dashboard(
    terrain, hazards, start_pos, goal_pos, routes_data,
    scenario_title="Mission Scenario", eval_altitude=180.0,
    save_path="results/figures/aerx_mission_dashboard.png", show_interactive=True,
):
    """Standalone wrapper that builds, saves and/or shows the dashboard."""
    def dummy_runner(_):
        return (terrain, hazards, start_pos, goal_pos, routes_data,
                scenario_title, eval_altitude)

    dashboard = InteractiveMissionDashboard(
        dummy_runner, initial_scenario="mountain", interactive_controls=False)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        stamp_figure(dashboard.fig, os.path.basename(save_path))
        dashboard.fig.savefig(save_path, dpi=160,
                              facecolor=dashboard.fig.get_facecolor(), edgecolor="none")
        print(f"  [OK] Saved Tactical Dashboard: {save_path}", flush=True)
    if show_interactive:
        plt.show()
    else:
        plt.close(dashboard.fig)


# --------------------------------------------------------------------------- #
#  Mission scorecard (table + summary bars + glossary)
# --------------------------------------------------------------------------- #
def render_mission_scorecard(routes_data, scenario_title, direct_dist,
                             save_path="results/figures/mission_scorecard_and_metrics_guide.png"):
    """High-resolution scorecard: comparison table, summary bars and glossary."""
    if not routes_data:
        print("  [!] No feasible routes to score — skipping scorecard.", flush=True)
        return
    fig = plt.figure(figsize=(18, 11), facecolor=BG_DEEP)

    fig.text(0.5, 0.955, "AERX LABS — TACTICAL MISSION PERFORMANCE SCORECARD",
             ha="center", color=ACCENT, fontsize=16, weight="bold")
    fig.text(0.5, 0.925, f"Scenario: {scenario_title}   |   Baseline chord distance: {direct_dist:.1f} m",
             ha="center", color=TEXT_LO, fontsize=11)

    # Top table = INGRESS-leg detail; the full-sortie mission score is the
    # separate round-trip table below (labels disambiguate the two composites).
    headers = ["Route Profile\n(ingress leg)", "Total Dist\n(m)", "Detour\n(%)", "Flight Time\n(s)",
               "Energy\n(norm)", "Int. Risk", "Peak Risk", "Min AGL\n(m)",
               "Surv.\n(ingress)", "Ingress\nScore", "MC P5-P95"]

    cell_data, cell_colors = [], []
    for name, r in routes_data.items():
        m, s, mc = r["metrics"], r["score"], r["mc"]
        detour = ((m.total_distance - direct_dist) / direct_dist) * 100.0
        cell_data.append([
            name, f"{m.total_distance:.1f}", f"+{detour:.1f}%", f"{m.flight_time:.1f}",
            f"{m.energy_proxy:.1f}", f"{m.integrated_risk:.2f}", f"{m.peak_risk:.3f}",
            f"{m.min_agl:.1f}", f"{s.survivability_score*100:.1f}", f"{s.composite_score:.1f}",
            f"[{mc.p5_worst_case_score:.0f}-{mc.p95_best_case_score:.0f}]",
        ])
        cell_colors.append(["#16223E" if name != "Balanced" else "#1B3350"] * len(headers))

    ax_table = fig.add_axes([0.05, 0.60, 0.90, 0.28])
    ax_table.axis("off")
    table = ax_table.table(cellText=cell_data, colLabels=headers, cellColours=cell_colors,
                           colColours=["#28406E"] * len(headers), loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 2.3)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(PANEL_EDGE)
        cell.set_linewidth(1.0)
        if row == 0:
            cell.set_text_props(color=TEXT_HI, weight="bold")
        else:
            pname = cell_data[row - 1][0]
            tcolor = PROFILE_STYLES.get(pname, {"color": "white"})["color"] if col == 0 else TEXT_HI
            cell.set_text_props(color=tcolor, weight="bold" if col in (0, 8, 9) else "normal")

    # Summary bar chart: full-sortie MISSION score + round-trip survivability.
    ax_bar = fig.add_axes([0.07, 0.34, 0.40, 0.20], facecolor=BG_PANEL)
    names = list(routes_data.keys())
    xs = np.arange(len(names))

    def _mission_comp(n):
        ms = routes_data[n].get("mission_score")
        return ms.composite_score if ms is not None else routes_data[n]["score"].composite_score

    def _rt_surv(n):
        rt = routes_data[n].get("round_trip")
        return (rt.survivability * 100) if rt is not None else routes_data[n]["score"].survivability_score * 100

    comp = [_mission_comp(n) for n in names]
    surv = [_rt_surv(n) for n in names]
    colors = [PROFILE_STYLES.get(n, {"color": "white"})["color"] for n in names]
    ax_bar.bar(xs - 0.2, comp, width=0.38, color=colors, alpha=0.9, label="Mission /100")
    ax_bar.bar(xs + 0.2, surv, width=0.38, color=colors, alpha=0.4, label="Survivability I+E %")
    for x, c, s in zip(xs, comp, surv):
        ax_bar.text(x - 0.2, c + 1.5, f"{c:.0f}", ha="center", color=TEXT_HI, fontsize=8, weight="bold")
        ax_bar.text(x + 0.2, s + 1.5, f"{s:.0f}", ha="center", color=TEXT_LO, fontsize=8)
    ax_bar.set_xticks(xs)
    ax_bar.set_xticklabels(names, color=TEXT_HI, fontsize=8, weight="bold")
    ax_bar.set_ylim(0, 115)
    ax_bar.set_title("Full-Sortie Mission Score vs Round-Trip Survivability",
                     color=TEXT_HI, fontsize=10, weight="bold")
    ax_bar.tick_params(colors=TEXT_LO, labelsize=7)
    for sp in ax_bar.spines.values():
        sp.set_color(PANEL_EDGE)
    ax_bar.legend(fontsize=7, facecolor=BG_PANEL, edgecolor=PANEL_EDGE, labelcolor=TEXT_HI)

    # Full-sortie (ingress + egress) round-trip summary, incl. the fuel budget.
    rts = {n: routes_data[n].get("round_trip") for n in names}
    if any(rt is not None for rt in rts.values()):
        ax_rt = fig.add_axes([0.05, 0.06, 0.42, 0.22])
        ax_rt.axis("off")
        budget = next((rt.fuel_budget for rt in rts.values() if rt is not None), 0.0)
        ax_rt.text(0.0, 1.02, "FULL SORTIE  ·  ingress + egress (go and get home)",
                   color=ACCENT, fontsize=10.5, weight="bold", transform=ax_rt.transAxes)
        ax_rt.text(0.0, 0.90, f"Onboard fuel budget: {budget:.1f} energy units  ·  the whole "
                   f"round trip must fit within it", color=TEXT_LO, fontsize=8.5, transform=ax_rt.transAxes)
        head = ["Route Profile", "RT Dist (m)", "Surv. I+E (%)", "Fuel used", "Fuel", "Mission /100"]
        rows = []
        for n in names:
            rt = rts[n]
            if rt is None:
                continue
            ms = routes_data[n].get("mission_score")
            status = f"{rt.fuel_margin_pct:+.0f}%  " + ("OK" if rt.fuel_ok else "OVER")
            rows.append([n, f"{rt.total_distance:.0f}", f"{rt.survivability*100:.1f}",
                         f"{rt.fuel_used:.1f} / {rt.fuel_budget:.1f}", status,
                         f"{ms.composite_score:.1f}" if ms is not None else "-"])
        t = ax_rt.table(cellText=rows, colLabels=head, colColours=["#28406E"] * len(head),
                        cellColours=[["#16223E"] * len(head) for _ in rows],
                        loc="center", cellLoc="center", bbox=[0, 0.0, 1, 0.78])
        t.auto_set_font_size(False)
        t.set_fontsize(9)
        for (row, col), cell in t.get_celld().items():
            cell.set_edgecolor(PANEL_EDGE)
            if row == 0:
                cell.set_text_props(color=TEXT_HI, weight="bold")
            else:
                pname = rows[row - 1][0]
                if col == 0:
                    cell.set_text_props(color=PROFILE_STYLES.get(pname, {"color": "white"})["color"], weight="bold")
                elif col == 4:  # fuel status: green OK / red OVER
                    ok = "OVER" not in rows[row - 1][4]
                    cell.set_text_props(color="#39FF9E" if ok else "#FF3D6E", weight="bold")
                elif col == 5:
                    cell.set_text_props(color=TEXT_HI, weight="bold")
                else:
                    cell.set_text_props(color=TEXT_HI)

    # Glossary.
    ax_guide = fig.add_axes([0.50, 0.05, 0.46, 0.50], facecolor=BG_PANEL)
    ax_guide.axis("off")
    ax_guide.add_patch(patches.FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02",
                       fc=BG_PANEL, ec=PANEL_EDGE, lw=1.4))
    ax_guide.text(0.05, 0.95, "ENGINEERING METRICS GUIDE", color=ACCENT, fontsize=12, weight="bold")
    definitions = [
        ("Total 3D Distance", "Cumulative 3D trajectory length vs direct chord."),
        ("Detour %", "Extra distance flown: (D_actual - D_chord)/D_chord * 100."),
        ("Flight Time", "Duration at 30 m/s cruise with climb/descent adjustments."),
        ("Energy (norm)", "Cruise power plus a 2.5x climb-power penalty."),
        ("Integrated Risk", "Cumulative threat integral along path: ∫R(x,y,z) dt."),
        ("Peak Risk", "Highest instantaneous exposure [0,1] on the route."),
        ("Min AGL", "Lowest above-ground clearance; hard ground-collision floor."),
        ("Survivability %", "S = exp(-λ·R_int); penalizes cumulative exposure."),
        ("Composite /100", "Survivability 50% · Distance 25% · Time 15% · Clearance 10%."),
        ("MC P5-P95", "Worst/best-case band under sensor drift & tracking noise."),
    ]
    for idx, (term, desc) in enumerate(definitions):
        y = 0.87 - idx * 0.085
        ax_guide.text(0.05, y, f"▸ {term}", color="#5BC0BE", fontsize=9, weight="bold")
        ax_guide.text(0.07, y - 0.035, desc, color=TEXT_LO, fontsize=8)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    stamp_figure(fig, os.path.basename(save_path))
    fig.savefig(save_path, dpi=160, facecolor=fig.get_facecolor(), edgecolor="none")
    print(f"  [OK] Saved Mission Scorecard & Metrics Guide: {save_path}", flush=True)
    plt.close(fig)
