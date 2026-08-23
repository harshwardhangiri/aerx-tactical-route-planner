# AerX Labs — Threat-Aware Tactical Route Planning Engine

Research-grade terrain-aware, risk-sensitive 3D UAV/rotary-wing mission planning simulator developed for the **AerX Labs Advanced Aerospace Systems Challenge 2026**.

---

## 📖 The brief — what this is, in one read

> A conventional planner finds the **shortest** path. This engine plans a **complete survivable sortie** — fly in, survive radar exposure by hiding in terrain, and get home on one tank of fuel — over **real elevation data**, and scores every candidate route.

**The core idea.** Terrain is one layer; radar visibility is another. The planner fuses them into a single cost and searches for the route that best balances distance, threat exposure, fuel, terrain clearance, and time — for the *whole round trip* (ingress **and** egress), not just the way in.

**The pipeline** (one vertical slice: *data → algorithm → score → visualization*):

1. **Terrain** — a real Copernicus GLO-30 DEM (Everest, Western Ghats, or any lat/lon), height by bilinear interpolation.
2. **Line-of-sight masking** — if terrain blocks the radar→aircraft line, the aircraft is in the radar's shadow and undetectable there. *This is why flying low in a valley beats flying straight.*
3. **Threat field** — each radar is a 3D detection volume; detectability follows the radar equation `M ∝ 1/R⁴`, gated by range, LOS, sector, cone, and minimum altitude, then combined across all radars into a single risk field `R(x,y,z) = 1 − ∏(1−Rᵢ)`.
4. **3D planning grid + D\* Lite** — airspace becomes `(x,y,altitude)` nodes with a weighted cost (distance + exposure + climb + clearance); D\* Lite finds the min-cost ingress and egress routes and *repairs* them cheaply when a threat pops up.
5. **Local refinement** — risk-aware B-spline smoothing or risk-aware RRT\* with helicopter turn/climb limits → a flyable nap-of-the-earth trajectory.
6. **Survivability & mission score** — integrate exposure along the path, convert to a survival probability `S = exp(−λ·R_int)`, sum both legs, and fold in a fuel-budget feasibility check → a single **mission score /100**.
7. **Visualization** — a tactical dashboard, a mission scorecard, Pareto & re-planning studies, an interactive Plotly 3D view, and a **CesiumJS georeferenced globe** flythrough.

**Threats are placed by the terrain, not by hand.** A DEM carries only elevation — no radars. For every region the engine reads the terrain and the base→target corridor and *auto-places* the threat laydown: a **search radar** on the tallest summit commanding the direct path, a **sector radar** on a flanking ridge, and a **SAM** on high ground over the corridor (blind below its minimum-altitude floor). Deterministic, and different for every region.

**Why it matters — the trade-off it exposes.** Over the Western Ghats, the shortest sortie flies right past the summit radar and is only **~34% survivable**; a route that detours through the valleys is **~69% survivable** for ~25% more distance. Neither is free — and a route that maximizes survival can be rejected by the fuel budget if it can't make it home. Surfacing that tension is the entire point of the engine. The threat range is auto-calibrated to each region so this trade-off stays meaningful everywhere — from the gentle Ghats to the extreme high Himalaya around Everest (a genuinely hard region, where even the best route is only modestly survivable).

**Run it** (every mission uses a real DEM; each run writes a fresh timestamped folder):

```bash
python main.py ghats                          # Western Ghats — the trade-off flagship
python main.py everest                        # Mount Everest / Khumbu
python main.py region --lat 34.1 --lon 74.8   # any location on Earth
python serve_viewer.py                        # open the CesiumJS 3D globe
```

📄 A richer, fully-illustrated version of this brief (with every formula and its reference) is in **`AerX_Route_Engine_Brief.html`** — open it in any browser. Full command reference and options are in the [Quick Start](#️-quick-start) below.

---

## 🖼️ Results gallery

*All figures are produced by `python main.py <region>` and land in `results/<region>/<timestamp>/`. The samples below are checked into [`assets/`](assets/).*

### Western Ghats — the distance-vs-survivability trade-off
The shortest sortie flies past the summit radar and is **~34% survivable**; the valley detour is **~69% survivable** for ~25% more distance — and wins the mission score.

| Mission scorecard | Tactical dashboard |
|:---:|:---:|
| ![Ghats scorecard](assets/ghats_scorecard.png) | ![Ghats tactical dashboard](assets/ghats_dashboard.png) |

The **tactical dashboard** packs four panels: the 3D battlefield (terrain + threat ranges + ingress/egress tubes), a top-down detection-risk heatmap with terrain shadows, a nap-of-the-earth altitude profile, and the Monte-Carlo robustness swarm.

### Any location on Earth — Swiss Alps (`--lat 46.5 --lon 8.0`)
Threats auto-place on the region's own summits; the safe route detours through the valleys (**0.2% → 65% survivable**).

![Swiss Alps tactical dashboard](assets/swiss_alps_dashboard.png)

### Mount Everest — a genuinely hard region
Extreme, confined high terrain offers little masking, so even the best route is only modestly survivable — an honest result, not a failure.

![Everest tactical dashboard](assets/everest_dashboard.png)

### Pareto risk-weight sweep
"How much should I weight safety?" — each point re-plans at a different risk weight; you buy survivability with distance until diminishing returns.

![Ghats Pareto sweep](assets/ghats_pareto.png)

### 🎥 CesiumJS 3D flythrough
The `*_cesium.html` viewer flies the sortie along a timeline over a georeferenced globe (terrain, ingress/egress tubes, threat domes). To add a clip to this README:

1. `python serve_viewer.py` and press ▶ on the timeline.
2. Screen-record the flythrough (Windows **Win+G** game bar, or any recorder).
3. Either save it as a GIF into `assets/flythrough.gif` and add `![flythrough](assets/flythrough.gif)` here, **or** drag the `.mp4` straight into this README in GitHub's web editor — GitHub hosts and renders it as a video player automatically.

---

## 🚀 Key Capabilities

1. **Digital Elevation Model (DEM) & Interpolation**: Continuous 2D terrain with sub-grid bilinear interpolation. Every mission runs on a **real Copernicus GLO-30 DEM via rasterio** — built-in Everest/Khumbu and Western Ghats regions, plus **any lat/lon on Earth** (the tile is fetched from public AWS open data and cached); any user SRTM/GeoTIFF tile drops in unchanged. Every saved figure is stamped with its generation time.
2. **Line-Of-Sight (LOS) Engine**: Ray-terrain intersection checking for geometric occlusion (sampled LOS with clearance margin, per Drones 2026 Eq. 4–8).
3. **Radar-Equation Detectability & 3D Threat Volumes**: Normalized inverse-R⁴ radar-equation detectability, range-gated and LOS-gated, aggregated by probabilistic union. Threats are full **3D detection volumes** — horizontal coverage sectors, vertical cones, minimum-detection-altitude floors — and typed (`radar`/`sam`/`aaa`) with per-type lethality weighting.
4. **Hard-Feasibility / Soft-Risk Planning**: Optional detection-threshold that marks over-exposed nodes as hard keep-outs (not just costs), implementing the hard-feasibility formulation of Drones 2026, 10, 469.
5. **Layered 3D Planning State Space**: Enforces AGL clearances and kinematic climb-slope limits; fast trilinear risk lookup for inner loops.
6. **D* Lite 3D Global Planner**: Incremental graph search with multi-objective cost, supporting **incremental replanning** for pop-up threats (the `*_replanning.png` demo, produced every run).
7. **Local Trajectory Refinement**: Risk-aware LOS pruning + B-spline smoothing, or **risk-aware RRT\*** with informed-tube sampling, risk-aware shortcutting, and **helicopter kinematic limits** (max turn-rate + climb-rate for NOE flight) (`--refine rrt`).
8. **Mission Performance & Survivability Scoring**: Travel time, energy proxies, cumulative/peak risk, survivability \(S_{\text{surv}} = e^{-\kappa R}\), and composite scores.
9. **Monte Carlo Robustness Analysis**: Stochastic evaluation under tracking/hazard-estimation perturbations.
10. **Sensitivity / Pareto Study**: Sweeps the risk weight and traces the distance-vs-exposure trade-off front (the `*_pareto.png` study, produced every run).
11. **Visualization**: Multi-panel Matplotlib dashboard (elevation-shaded 3D terrain, radar-shadow heatmaps, nap-of-the-earth profiles, Monte-Carlo swarms) **and** interactive **browser 3D exports** — a self-contained Plotly HTML view and a **CesiumJS** georeferenced globe flythrough — produced every run.

> **Safety framing:** all hazards are generic simulated sensors with normalized, illustrative parameters (no real operational RCS/weapon/radar values). Results are planning-level research-simulation estimates, not operationally deployable.

---

## 📁 Repository Structure

```text
AerX_Prototype/
├── src/
│   ├── terrain.py          # Copernicus GLO-30 loader, relief-window picker, SRTM/GeoTIFF I/O
│   ├── los.py              # Bilinear interpolation & LOS ray-casting
│   ├── risk_field.py       # Hazard sites, radar-equation detectability, terrain masking
│   ├── planning_grid.py    # 3D state space, multi-objective costs, hard keep-out, risk lookup
│   ├── dstar_lite.py       # D* Lite 3D incremental path planner
│   ├── smoothing.py        # Risk-aware LOS pruning & B-spline smoothing
│   ├── rrt_star.py         # Risk-aware RRT* local refinement + shortcutting
│   ├── route_metrics.py    # Flight time, energy, distance & risk metrics
│   ├── scoring.py          # Survivability probability & composite scoring
│   ├── monte_carlo.py      # Stochastic robustness evaluation
│   ├── analysis.py         # Pareto sensitivity sweep & incremental-replanning demo
│   ├── visualization.py    # 4-panel tactical dashboard & scorecard
│   ├── web_export.py       # Interactive browser 3D (Plotly HTML) export
│   ├── cesium_export.py    # CesiumJS 3D globe viewer export
│   └── render_env.py       # Matplotlib backend / Tcl-Tk auto-configuration
│
├── tests/
│   ├── test_risk_field.py     # Unit tests for spatial risk & terrain masking
│   ├── test_planning_grid.py  # Unit tests for 3D state space & feasibility
│   ├── test_dstar_lite.py     # Unit tests for D* Lite 3D planning
│   └── test_evaluation.py     # Unit tests for metrics, scoring & Monte Carlo
│
├── data/terrain/           # Real Copernicus GLO-30 DEM tiles (downloaded once)
├── results/                # Timestamped run outputs (never overwritten)
│
├── main.py                 # Single entry point — engine + full analysis suite CLI
├── serve_viewer.py         # Local http server that opens the CesiumJS viewer
├── AerX_Route_Engine_Brief.html   # "How it works" technical brief (open in a browser)
├── requirements.txt        # Python dependency specifications
└── README.md               # System documentation
```

---

## ⚙️ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Automated Test Suite
```bash
python -m unittest discover tests
```

### 3. Run a mission — one command, one real-DEM region
Every mission runs on a **real Copernicus GLO-30 DEM** and is planned as a full sortie
(ingress + egress, scored on one fuel budget). Pick a region:
```bash
python main.py everest      # Mount Everest / Khumbu
python main.py ghats        # Western Ghats (Anamalai) — strong distance-vs-risk trade-off
```
Each run writes a fresh, timestamped folder `results/<region>/<timestamp>/` (nothing is ever
overwritten) containing the **full product set**:

| File | What it is |
|------|-----------|
| `<region>_scorecard.png`   | mission metrics + survivability scorecard |
| `<region>_dashboard.png`   | 3D tactical dashboard (terrain + threats + ingress/egress routes) |
| `<region>_pareto.png`      | risk-weight Pareto trade-off sweep |
| `<region>_replanning.png`  | D* Lite pop-up-threat incremental-repair demo |
| `<region>_3d.html`         | interactive Plotly 3D view (double-click to open) |
| `<region>_cesium.html`     | CesiumJS georeferenced globe flythrough (serve over http, see §6) |

### 4. Run any location on Earth
```bash
python main.py region --lat 34.1 --lon 74.8 --label kashmir
python main.py region --lat 46.5 --lon 8.0                     # Swiss Alps
```
It downloads the matching Copernicus tile (cached in `data/terrain/`), auto-picks a
navigable-relief sub-region, and produces the same full product set in `results/<label>/<timestamp>/`.

### 5. Options (any of the commands above)
```bash
python main.py ghats --quick          # scorecard + dashboard only (fast iteration)
python main.py everest --refine rrt   # risk-aware RRT* local refinement (else B-spline smoothing)
```

### 6. CesiumJS 3D viewer
The Cesium viewer must be served over http (its 3D workers can't load from a `file://` page):
```bash
python serve_viewer.py
```
This starts a local server and opens the most recently generated `*_cesium.html` in your browser.
Leave it running; press Ctrl+C to stop. (The Plotly `*_3d.html` viewer needs no server.)

The real DEM tiles live in `data/terrain/` (Copernicus GLO-30, downloaded once). A full "how it
works" brief is in `AerX_Route_Engine_Brief.html` (open in any browser).

> On Windows, if `python` isn't the venv interpreter, call `.venv\Scripts\python.exe` directly
> (no `Activate.ps1` needed). Headless runs need no Tcl/Tk; the backend is auto-selected.
