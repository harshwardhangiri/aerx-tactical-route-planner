"""
AerX Labs — CesiumJS 3D Mission Viewer Export
src/cesium_export.py

Writes a self-contained CesiumJS web page for a mission: a real 3D globe with the
DEM terrain rendered as an elevation-coloured surface, the ingress (solid) and
egress (dashed) flight tubes, translucent threat-coverage volumes, and an
animated aircraft that flies the balanced route — with CesiumJS's built-in
timeline / play-pause / scrub controls.

No Cesium Ion account or token is needed: the scene is built entirely from our
own data in a local East-North-Up frame anchored on the globe (imagery disabled),
so nothing is streamed. CesiumJS itself loads from its public CDN, so the page
needs an internet connection the first time it opens.

Because the planner works in normalized units (see the units note in the brief),
elevation is shown with a vertical exaggeration so the terrain reads as
mountainous; routes and threats use the same frame, so their relative geometry is
correct.
"""

import json
import os
from typing import Dict, List, Tuple

import numpy as np

from src.los import terrain_height
from src.risk_field import HazardSite

CESIUM_VERSION = "1.118"
# jsDelivr serves the npm package directly with CORS and no redirects, which
# avoids the "cross-origin worker redirect" refusal seen with the cesium.com CDN.
_CESIUM_BASE = f"https://cdn.jsdelivr.net/npm/cesium@{CESIUM_VERSION}/Build/Cesium/"

_PROFILE = {
    "Distance-focused": "#22D3EE",
    "Risk-focused": "#FF3D6E",
    "Balanced": "#39FF9E",
}


def export_cesium(
    terrain: np.ndarray,
    hazards: List[HazardSite],
    start_pos: Tuple[float, float, float],
    goal_pos: Tuple[float, float, float],
    routes_data: Dict[str, dict],
    scenario_title: str,
    save_path: str,
    lon0: float = 77.1,
    lat0: float = 10.4,
    cell_m: float = 150.0,
    vert_exag: float = 40.0,
    stride: int = 1,
    cruise_speed: float = 30.0,
) -> str:
    """Write a CesiumJS mission viewer HTML file. Returns the path."""
    ny, nx = terrain.shape
    zmin = float(terrain.min())

    # Terrain surface as a colour-coded point field (subsampled by `stride`).
    tz = terrain[::stride, ::stride]
    sy, sx = tz.shape
    zrange = float(terrain.max() - terrain.min()) or 1.0
    terrain_pts = []
    for jj in range(sy):
        for ii in range(sx):
            i = ii * stride
            j = jj * stride
            z = float(tz[jj, ii])
            terrain_pts.append([i, j, round(z, 2), round((z - zmin) / zrange, 3)])

    # Routes: ingress (solid) + egress (dashed) per profile.
    routes = {}
    for name, r in routes_data.items():
        ing = np.asarray(r["smooth_coords"], dtype=float)
        entry = {"color": _PROFILE.get(name, "#FFFFFF"),
                 "ingress": [[round(p[0], 2), round(p[1], 2), round(p[2], 2)] for p in ing]}
        eg = r.get("egress_smooth_coords")
        if eg is not None:
            eg = np.asarray(eg, dtype=float)
            entry["egress"] = [[round(p[0], 2), round(p[1], 2), round(p[2], 2)] for p in eg]
        routes[name] = entry

    # Threats: position + detection radius + type + name.
    threats = []
    for h in hazards:
        p = h.get_position(terrain)
        threats.append({
            "x": round(float(p[0]), 2), "y": round(float(p[1]), 2), "z": round(float(p[2]), 2),
            "ground": round(terrain_height(terrain, h.x, h.y), 2),
            "radius": float(h.radius), "name": h.name, "type": h.threat_type,
        })

    data = {
        "title": scenario_title,
        "nx": nx, "ny": ny, "zmin": zmin,
        "lon0": lon0, "lat0": lat0, "cell_m": cell_m, "vert_exag": vert_exag,
        "cruise_speed": cruise_speed,
        "terrain": terrain_pts,
        "routes": routes,
        "threats": threats,
        "start": [round(float(start_pos[0]), 2), round(float(start_pos[1]), 2), round(float(start_pos[2]), 2)],
        "goal": [round(float(goal_pos[0]), 2), round(float(goal_pos[1]), 2), round(float(goal_pos[2]), 2)],
    }

    html = _TEMPLATE.replace("__CESIUM_BASE__", _CESIUM_BASE).replace(
        "__CESIUM_VERSION__", CESIUM_VERSION).replace(
        "__TITLE__", scenario_title).replace(
        "__DATA__", json.dumps(data))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  [OK] Saved CesiumJS 3D viewer: {save_path}", flush=True)
    return save_path


_TEMPLATE = r"""<!doctype html>
<html><head><meta charset="utf-8"/>
<title>AerX Cesium — __TITLE__</title>
<script>window.CESIUM_BASE_URL = "__CESIUM_BASE__";</script>
<script src="__CESIUM_BASE__Cesium.js"></script>
<link href="__CESIUM_BASE__Widgets/widgets.css" rel="stylesheet">
<style>
  html,body,#c{margin:0;height:100%;width:100%;overflow:hidden;background:#0A0F1E;font-family:system-ui,sans-serif}
  #hud{position:absolute;top:10px;left:10px;z-index:5;background:rgba(14,23,48,.86);border:1px solid #25334F;
    border-radius:8px;padding:12px 14px;color:#D9E2F1;max-width:340px;font-size:13px}
  #hud h1{margin:0 0 6px;font-size:14px;color:#3DE1FF;letter-spacing:.02em}
  #hud .row{display:flex;align-items:center;gap:8px;margin:3px 0;color:#9FB0C8}
  #hud .sw{width:22px;height:3px;border-radius:2px;display:inline-block}
  #hud .dash{background:repeating-linear-gradient(90deg,currentColor 0 5px,transparent 5px 9px);height:3px}
  #hud .note{margin-top:8px;color:#5E6E8C;font-size:11px;line-height:1.5}
  #err{position:absolute;inset:0;display:none;align-items:center;justify-content:center;color:#FF3D6E;
    background:#0A0F1E;font-size:16px;padding:40px;text-align:center;z-index:10}
</style></head>
<body>
<div id="c"></div>
<div id="hud">
  <h1>AerX — __TITLE__</h1>
  <div class="row"><span class="sw" style="background:#22D3EE"></span>Distance-focused</div>
  <div class="row"><span class="sw" style="background:#FF3D6E"></span>Risk-focused</div>
  <div class="row"><span class="sw" style="background:#39FF9E"></span>Balanced</div>
  <div class="row"><span class="sw dash" style="color:#9FB0C8"></span>dashed = egress (return)</div>
  <div class="row"><span class="sw" style="background:rgba(255,61,110,.5)"></span>threat site &amp; range (representative dome)</div>
  <div class="note"><b>Left-drag to orbit</b> · scroll to zoom · middle-drag to pan.
    Use the timeline / play button (bottom) to fly the balanced sortie; click the aircraft
    to chase it. Heights are vertically exaggerated.</div>
</div>
<div id="err">CesiumJS failed to load — check your internet connection and reopen this file.</div>
<div id="filewarn" style="position:absolute;inset:0;display:none;flex-direction:column;align-items:center;
  justify-content:center;background:#0A0F1E;color:#D9E2F1;text-align:center;padding:40px;z-index:20;font-size:15px;line-height:1.6">
  <div style="color:#FFC24B;font-size:20px;font-weight:600;margin-bottom:14px">Open this viewer over a local server</div>
  <div style="max-width:560px;color:#9FB0C8">Cesium needs to be served over http (its 3D workers can't load from a
  <code>file://</code> page). From the project folder run:</div>
  <div style="font-family:monospace;background:#111B33;border:1px solid #25334F;border-radius:6px;
    padding:12px 16px;margin:16px 0;color:#3DE1FF">.venv\Scripts\python.exe serve_viewer.py</div>
  <div style="max-width:560px;color:#9FB0C8">— it starts a local server and opens the latest viewer automatically.</div>
</div>
<script>
const DATA = __DATA__;
if (location.protocol === "file:") { document.getElementById("filewarn").style.display = "flex"; }
try {
  if (location.protocol === "file:") throw new Error("file protocol — serve over http");
  if (typeof Cesium === "undefined") throw new Error("Cesium not loaded");
  Cesium.Ion.defaultAccessToken = undefined;

  const viewer = new Cesium.Viewer("c", {
    baseLayer: false, baseLayerPicker: false, geocoder: false, homeButton: false,
    sceneModePicker: false, navigationHelpButton: false, selectionIndicator: false,
    infoBox: false, timeline: true, animation: true, shouldAnimate: false,
    terrainProvider: new Cesium.EllipsoidTerrainProvider(),
  });
  const scene = viewer.scene;
  scene.globe.baseColor = Cesium.Color.fromCssColorString("#0A0F1E");
  scene.skyBox && (scene.skyBox.show = false);
  scene.backgroundColor = Cesium.Color.fromCssColorString("#05070D");
  scene.globe.showGroundAtmosphere = false;

  // --- Camera controls tuned for inspecting a small local model ---
  // Cesium's default maps LEFT-drag to *panning the globe*, which on a compact
  // local scene feels like "nothing rotates" (you'd have to middle-drag or
  // Ctrl+left-drag to tilt). Remap so LEFT-drag ORBITS the terrain (heading +
  // pitch around the point under the cursor), wheel / right-drag zoom, and
  // middle-drag pans. Collision detection off so you can orbit in close freely.
  const camCtl = scene.screenSpaceCameraController;
  camCtl.tiltEventTypes   = [Cesium.CameraEventType.LEFT_DRAG, Cesium.CameraEventType.PINCH];
  camCtl.zoomEventTypes   = [Cesium.CameraEventType.WHEEL, Cesium.CameraEventType.RIGHT_DRAG, Cesium.CameraEventType.PINCH];
  camCtl.rotateEventTypes = [Cesium.CameraEventType.MIDDLE_DRAG];
  camCtl.enableCollisionDetection = false;
  camCtl.minimumZoomDistance = 25;

  // Local East-North-Up frame anchored on the globe.
  const origin = Cesium.Cartesian3.fromDegrees(DATA.lon0, DATA.lat0, 0);
  const enu = Cesium.Transforms.eastNorthUpToFixedFrame(origin);
  const CM = DATA.cell_m, VE = DATA.vert_exag, Z0 = DATA.zmin;
  function toWorld(i, j, z) {
    const local = new Cesium.Cartesian3(i * CM, j * CM, (z - Z0) * VE + 5);
    return Cesium.Matrix4.multiplyByPoint(enu, local, new Cesium.Cartesian3());
  }
  // Elevation colour ramp (teal lowland -> pale ridge).
  const RAMP = [[18,60,74],[30,95,82],[62,122,58],[126,143,60],[183,154,85],[217,198,160],[242,236,221]];
  function elevColor(t) {
    t = Math.max(0, Math.min(1, t)) * (RAMP.length - 1);
    const a = RAMP[Math.floor(t)], b = RAMP[Math.min(RAMP.length - 1, Math.ceil(t))], f = t - Math.floor(t);
    return new Cesium.Color((a[0]+(b[0]-a[0])*f)/255,(a[1]+(b[1]-a[1])*f)/255,(a[2]+(b[2]-a[2])*f)/255,1);
  }

  // Terrain surface as a colour-coded point cloud.
  const pts = scene.primitives.add(new Cesium.PointPrimitiveCollection());
  for (const p of DATA.terrain) {
    pts.add({ position: toWorld(p[0], p[1], p[2]), color: elevColor(p[3]), pixelSize: 6 });
  }

  // Flight routes: ingress solid, egress dashed.
  for (const name in DATA.routes) {
    const r = DATA.routes[name];
    const col = Cesium.Color.fromCssColorString(r.color);
    // depthFailMaterial keeps the route visible (dimmed) where terrain occludes it.
    viewer.entities.add({ polyline: {
      positions: r.ingress.map(p => toWorld(p[0], p[1], p[2])),
      width: 6, arcType: Cesium.ArcType.NONE,
      material: col, depthFailMaterial: col.withAlpha(0.45) } });
    if (r.egress) {
      viewer.entities.add({ polyline: {
        positions: r.egress.map(p => toWorld(p[0], p[1], p[2])),
        width: 4, arcType: Cesium.ArcType.NONE,
        material: new Cesium.PolylineDashMaterialProperty({ color: col, dashLength: 14 }),
        depthFailMaterial: new Cesium.PolylineDashMaterialProperty({ color: col.withAlpha(0.4), dashLength: 14 }) } });
    }
  }

  // Threat sites + translucent detection-coverage volumes.
  for (const h of DATA.threats) {
    const apex = toWorld(h.x, h.y, h.z);
    viewer.entities.add({ position: apex,
      point: { pixelSize: 11, color: Cesium.Color.fromCssColorString("#FF3D6E"),
               outlineColor: Cesium.Color.WHITE, outlineWidth: 1.5,
               disableDepthTestDistance: Number.POSITIVE_INFINITY },
      label: { text: h.name, font: "12px sans-serif", fillColor: Cesium.Color.fromCssColorString("#FFD8E1"),
               pixelOffset: new Cesium.Cartesian2(0, -16), showBackground: true,
               backgroundColor: Cesium.Color.fromCssColorString("#111B33").withAlpha(0.8),
               disableDepthTestDistance: Number.POSITIVE_INFINITY } });
    // Coverage dome: the true max range is large relative to this crop (that's
    // what lets a ridge radar threaten the valley), so a literal sphere would
    // swallow the map. Draw a capped, representative dome instead — it marks the
    // site and its rough reach; the exact field is in the 2D risk heatmap.
    const vr = Math.min(h.radius, 0.28 * Math.max(DATA.nx, DATA.ny));
    viewer.entities.add({ position: apex,
      ellipsoid: { radii: new Cesium.Cartesian3(vr * CM, vr * CM, vr * VE * 0.9),
        material: Cesium.Color.fromCssColorString("#FF3D6E").withAlpha(0.12),
        outline: true, outlineColor: Cesium.Color.fromCssColorString("#FF3D6E").withAlpha(0.45) } });
    // Mast to the ground.
    viewer.entities.add({ polyline: { positions: [toWorld(h.x, h.y, (h.ground)), apex],
      width: 2, material: Cesium.Color.fromCssColorString("#FF6B8A") } });
  }

  // Start & target markers.
  const DEPTHOFF = Number.POSITIVE_INFINITY;
  viewer.entities.add({ position: toWorld(DATA.start[0], DATA.start[1], DATA.start[2]),
    point: { pixelSize: 12, color: Cesium.Color.fromCssColorString("#2E9BFF"), outlineColor: Cesium.Color.WHITE, outlineWidth: 2, disableDepthTestDistance: DEPTHOFF },
    label: { text: "BASE", font: "12px sans-serif", fillColor: Cesium.Color.WHITE, pixelOffset: new Cesium.Cartesian2(0, -18), disableDepthTestDistance: DEPTHOFF } });
  viewer.entities.add({ position: toWorld(DATA.goal[0], DATA.goal[1], DATA.goal[2]),
    point: { pixelSize: 14, color: Cesium.Color.fromCssColorString("#FFD23F"), outlineColor: Cesium.Color.BLACK, outlineWidth: 2, disableDepthTestDistance: DEPTHOFF },
    label: { text: "TARGET", font: "12px sans-serif", fillColor: Cesium.Color.fromCssColorString("#FFD23F"), pixelOffset: new Cesium.Cartesian2(0, -18), disableDepthTestDistance: DEPTHOFF } });

  // Animated aircraft flying the full balanced sortie (ingress + egress).
  const bal = DATA.routes["Balanced"] || DATA.routes[Object.keys(DATA.routes)[0]];
  let flight = bal.ingress.slice();
  if (bal.egress) flight = flight.concat(bal.egress);
  const startT = Cesium.JulianDate.fromIso8601("2026-01-01T00:00:00Z");
  const posProp = new Cesium.SampledPositionProperty();
  let t = 0;
  posProp.addSample(startT, toWorld(flight[0][0], flight[0][1], flight[0][2]));
  for (let k = 1; k < flight.length; k++) {
    const a = flight[k - 1], b = flight[k];
    const seg = Math.hypot(b[0]-a[0], b[1]-a[1], b[2]-a[2]) * CM / DATA.cruise_speed;
    t += Math.max(seg, 0.05);
    posProp.addSample(Cesium.JulianDate.addSeconds(startT, t, new Cesium.JulianDate()),
                      toWorld(b[0], b[1], b[2]));
  }
  const stopT = Cesium.JulianDate.addSeconds(startT, t, new Cesium.JulianDate());
  viewer.clock.startTime = startT.clone();
  viewer.clock.stopTime = stopT.clone();
  viewer.clock.currentTime = startT.clone();
  viewer.clock.clockRange = Cesium.ClockRange.LOOP_STOP;
  viewer.clock.multiplier = 12;
  viewer.timeline.zoomTo(startT, stopT);

  viewer.entities.add({
    position: posProp,
    orientation: new Cesium.VelocityOrientationProperty(posProp),
    point: { pixelSize: 13, color: Cesium.Color.fromCssColorString("#39FF9E"), outlineColor: Cesium.Color.WHITE, outlineWidth: 2, disableDepthTestDistance: DEPTHOFF },
    path: { resolution: 1, leadTime: 0, trailTime: 30, width: 4,
            material: new Cesium.PolylineGlowMaterialProperty({ glowPower: 0.25, color: Cesium.Color.fromCssColorString("#39FF9E") }) },
  });

  // Frame the battlefield: a bounding sphere over the whole terrain footprint,
  // viewed obliquely. (Computed from the terrain corners so the full map fits.)
  const corners = [toWorld(0, 0, Z0), toWorld(DATA.nx, 0, Z0), toWorld(0, DATA.ny, Z0),
                   toWorld(DATA.nx, DATA.ny, Z0), toWorld(DATA.nx / 2, DATA.ny / 2, DATA.zmin + 300)];
  const bs = Cesium.BoundingSphere.fromPoints(corners);
  viewer.camera.flyToBoundingSphere(bs, {
    duration: 0,
    offset: new Cesium.HeadingPitchRange(Cesium.Math.toRadians(-40), Cesium.Math.toRadians(-32), bs.radius * 2.4),
  });
  // Hide the Ion credit lightbox link (we use no Ion assets).
  try { viewer.cesiumWidget.creditContainer.style.display = "none"; } catch (e) {}
} catch (e) {
  if (location.protocol !== "file:") document.getElementById("err").style.display = "flex";
  console.error(e);
}
</script></body></html>
"""
