"""
AerX Labs — Interactive Browser 3D Export
src/web_export.py

Exports a mission to a single self-contained HTML file with an interactive,
rotatable/zoomable 3D scene (terrain surface + flight tubes + threats), which
the user opens in any browser.

The concept document names CesiumJS as the eventual browser-based 3D viewer.
CesiumJS needs an Ion access token and streamed world terrain, which is
impractical for an offline research prototype driven by a synthetic/loaded DEM.
This module delivers the same capability — browser-based interactive 3D of the
actual planned mission — using Plotly.js (loaded from its CDN), so the file is
small and needs no account, token, or build step. Only an internet connection is
required the first time the page loads the Plotly script.
"""

import json
import os
from typing import Dict, List, Tuple

import numpy as np

from src.los import terrain_height
from src.risk_field import HazardSite

_PROFILE_WEB = {
    "Distance-focused": "#22D3EE",
    "Risk-focused": "#FF3D6E",
    "Balanced": "#39FF9E",
}


def export_web_3d(
    terrain: np.ndarray,
    hazards: List[HazardSite],
    start_pos: Tuple[float, float, float],
    goal_pos: Tuple[float, float, float],
    routes_data: Dict[str, dict],
    scenario_title: str,
    save_path: str,
    stride: int = 2,
) -> str:
    """Write an interactive Plotly 3D HTML file for the mission. Returns the path."""
    ny, nx = terrain.shape
    xs = list(range(0, nx, stride))
    ys = list(range(0, ny, stride))
    z_sub = terrain[::stride, ::stride]

    traces = []

    # Terrain surface.
    traces.append({
        "type": "surface", "x": xs, "y": ys, "z": z_sub.tolist(),
        "colorscale": "Earth", "opacity": 0.95, "showscale": False,
        "contours": {"z": {"show": True, "usecolormap": True, "project": {"z": True}}},
        "name": "Terrain", "hoverinfo": "skip",
    })

    # Flight tubes: ingress = solid, egress (return) = dashed.
    for name, r in routes_data.items():
        c = np.asarray(r["smooth_coords"])
        color = _PROFILE_WEB.get(name, "#FFFFFF")
        traces.append({
            "type": "scatter3d", "mode": "lines",
            "x": c[:, 0].tolist(), "y": c[:, 1].tolist(), "z": c[:, 2].tolist(),
            "line": {"color": color, "width": 7}, "name": f"{name} (ingress)",
        })
        eg = r.get("egress_smooth_coords")
        if eg is not None:
            eg = np.asarray(eg)
            traces.append({
                "type": "scatter3d", "mode": "lines",
                "x": eg[:, 0].tolist(), "y": eg[:, 1].tolist(), "z": eg[:, 2].tolist(),
                "line": {"color": color, "width": 4, "dash": "dash"},
                "name": f"{name} (egress)",
            })

    # Threats: mast + apex marker.
    for h in hazards:
        p = h.get_position(terrain)
        g = terrain_height(terrain, h.x, h.y)
        traces.append({
            "type": "scatter3d", "mode": "lines",
            "x": [h.x, h.x], "y": [h.y, h.y], "z": [g, p[2]],
            "line": {"color": "#FF3D6E", "width": 4}, "showlegend": False, "hoverinfo": "skip",
        })
        traces.append({
            "type": "scatter3d", "mode": "markers",
            "x": [p[0]], "y": [p[1]], "z": [p[2]],
            "marker": {"color": "#FF3D6E", "size": 5, "symbol": "diamond"},
            "name": h.name,
        })

    # Start / goal.
    traces.append({
        "type": "scatter3d", "mode": "markers",
        "x": [start_pos[0]], "y": [start_pos[1]], "z": [start_pos[2]],
        "marker": {"color": "#2E9BFF", "size": 6}, "name": "Start",
    })
    traces.append({
        "type": "scatter3d", "mode": "markers",
        "x": [goal_pos[0]], "y": [goal_pos[1]], "z": [goal_pos[2]],
        "marker": {"color": "#FFD23F", "size": 7, "symbol": "diamond"}, "name": "Target",
    })

    layout = {
        "title": {"text": f"AerX Labs — {scenario_title}", "font": {"color": "#E8EEF7", "size": 18}},
        "paper_bgcolor": "#0B1020", "plot_bgcolor": "#0B1020",
        "font": {"color": "#9FB0C8"},
        "scene": {
            "xaxis": {"title": "X (m)", "backgroundcolor": "#0E1730", "gridcolor": "#24304F"},
            "yaxis": {"title": "Y (m)", "backgroundcolor": "#0E1730", "gridcolor": "#24304F"},
            "zaxis": {"title": "Alt ASL (m)", "backgroundcolor": "#0E1730", "gridcolor": "#24304F"},
            "aspectratio": {"x": 1, "y": 1, "z": 0.45},
            "camera": {"eye": {"x": 1.6, "y": -1.6, "z": 1.1}},
        },
        "legend": {"font": {"color": "#E8EEF7"}, "bgcolor": "#0E1730"},
        "margin": {"l": 0, "r": 0, "t": 50, "b": 0},
    }

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"/>
<title>AerX Labs 3D — {scenario_title}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>html,body{{margin:0;height:100%;background:#0B1020;font-family:system-ui,sans-serif}}
#viz{{width:100vw;height:100vh}}</style></head>
<body><div id="viz"></div>
<script>
const data = {json.dumps(traces)};
const layout = {json.dumps(layout)};
Plotly.newPlot('viz', data, layout, {{responsive:true, displaylogo:false}});
</script></body></html>"""

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  [OK] Saved interactive 3D web view: {save_path}", flush=True)
    return save_path
