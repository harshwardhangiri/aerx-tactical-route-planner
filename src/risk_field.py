"""
AerX Labs — Generic Simulated Risk Field Module
src/risk_field.py

Generates research-grade synthetic spatial hazard/risk fields over terrain.
Incorporates distance attenuation models and geometric Line-of-Sight (LOS)
terrain-masking to compute 3D risk maps for path planning.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np

from src.los import check_los, terrain_height


@dataclass
class HazardSite:
    """
    Represents a generic synthetic hazard or sensor site as a 3D detection volume.

    Attributes:
        x, y (float): Horizontal position in simulation grid units.
        z (Optional[float]): Altitude ASL. If None, placed on ground + mast_height.
        mast_height (float): Mast/sensor height above ground if z is None.
        radius (float): Maximum effective slant range R_max.
        intensity (float): Peak intensity / normalized RCS ratio in [0, 1].
        requires_los (bool): If True, terrain occlusion masks/blocks detection.
        decay_type (str): 'radar' (inverse-R^4), 'quadratic', 'linear', 'gaussian'.
        name (str): Identifier for reporting and visualization.
        threat_type (str): 'radar', 'sam', or 'aaa' — informational + used by the
            detection-probability model to scale the per-type hazard rate.

    3D detection-volume gating (all optional; defaults = omnidirectional full sphere,
    preserving the simple spherical model):
        min_detect_alt (Optional[float]): minimum target altitude ASL that can be
            detected (targets below this are masked — e.g. a radar horizon / low
            clutter floor, or a SAM minimum-engagement altitude).
        azimuth_center (Optional[float]): center bearing (deg) of the horizontal
            coverage sector; None = omnidirectional.
        azimuth_width (float): angular width (deg) of the horizontal sector.
        elev_angle_min, elev_angle_max (float): vertical coverage cone, in degrees
            of elevation angle from the emitter to the target (negative = target
            below emitter). Defaults span the full -90..+90 hemisphere.
    """
    x: float
    y: float
    z: Optional[float] = None
    mast_height: float = 5.0
    radius: float = 35.0
    intensity: float = 1.0
    requires_los: bool = True
    decay_type: str = "quadratic"
    name: str = "Generic Hazard"
    threat_type: str = "radar"
    min_detect_alt: Optional[float] = None
    azimuth_center: Optional[float] = None
    azimuth_width: float = 360.0
    elev_angle_min: float = -90.0
    elev_angle_max: float = 90.0

    def get_position(self, terrain: Optional[np.ndarray] = None) -> np.ndarray:
        """Return the 3D position [x, y, z] of the hazard site."""
        if self.z is not None:
            return np.array([self.x, self.y, self.z], dtype=float)

        if terrain is not None:
            ground = terrain_height(terrain, self.x, self.y)
            return np.array([self.x, self.y, ground + self.mast_height], dtype=float)

        return np.array([self.x, self.y, self.mast_height], dtype=float)

    def covers(self, x: float, y: float, z: float, haz_pos: np.ndarray) -> bool:
        """
        True if the point (x, y, z) lies inside this hazard's 3D detection volume
        (minimum altitude, horizontal sector, and vertical cone). Range and LOS
        are checked separately by the caller.
        """
        if self.min_detect_alt is not None and z < self.min_detect_alt:
            return False

        if self.azimuth_center is not None:
            bearing = np.degrees(np.arctan2(y - haz_pos[1], x - haz_pos[0])) % 360.0
            diff = abs(((bearing - self.azimuth_center + 180.0) % 360.0) - 180.0)
            if diff > self.azimuth_width / 2.0:
                return False

        horiz = float(np.hypot(x - haz_pos[0], y - haz_pos[1]))
        elev = np.degrees(np.arctan2(z - haz_pos[2], max(horiz, 1e-6)))
        if not (self.elev_angle_min <= elev <= self.elev_angle_max):
            return False

        return True


# Normalized detectability at the maximum-range boundary R_max for the physical
# radar model. Calibrating the boundary to a small value reproduces the paper's
# detection-threshold calibration Metric(R_max, sigma_ref) = eta (Eq. 28).
RADAR_BOUNDARY_DETECTABILITY = 0.05

# Vertical weight for the detection-range geometry. Real terrain is normalized to a
# fixed planning band ([90,350] over a ~120-cell grid), which exaggerates vertical
# relief ~2x relative to the horizontal. Left uncorrected, the 3D slant range is
# vertical-dominated and a summit radar becomes all-or-nothing (covers the whole
# crop, or nothing). Counting the vertical component at this reduced weight in the
# RANGE gate restores a proper horizontal detection footprint — a summit radar
# reaches nearby valley aircraft but distant ones escape by range — so routes keep
# real dynamic range (a discernible best route) in every region. Angular gates
# (elevation cone in HazardSite.covers) still use the true geometry.
Z_RISK_SCALE = 0.35

# Per-threat-type lethality weight applied to the detection hazard rate. Generic,
# normalized values (not operational): a surface-to-air missile site is treated as
# more lethal than a search radar, which is more lethal than short-range AAA once
# a target is within its (smaller) envelope.
THREAT_TYPE_WEIGHT = {
    "radar": 1.0,
    "sam": 1.3,
    "aaa": 0.85,
}

# Generic anisotropic-RCS multipliers (relative to a reference RCS of 1.0). An
# aircraft reflects less nose-/tail-on and more broadside, so it is harder to
# detect when flying toward/away from a radar and easier when showing its side
# (Drones 2026, 10, 469, Eq. 37: sigma(a) = sigma_front + (sigma_side - sigma_front)*sin^2(a)).
# Generic values — no specific airframe.
RCS_FRONT = 0.6
RCS_SIDE = 1.5


def rcs_aspect_factor(point_xy, heading_xy, radar_xy) -> float:
    """
    Heading-dependent RCS multiplier for a target at `point_xy` flying along
    `heading_xy`, as seen by a radar at `radar_xy`.

    sin^2 of the aspect angle is 0 when the radar is dead ahead/behind (nose/tail
    aspect -> low RCS) and 1 when it is abeam (broadside -> high RCS).
    """
    h = np.asarray(heading_xy, dtype=float)[:2]
    hn = np.linalg.norm(h)
    if hn < 1e-9:
        return 1.0
    h = h / hn
    g = np.array([radar_xy[0] - point_xy[0], radar_xy[1] - point_xy[1]], dtype=float)
    gn = np.linalg.norm(g)
    if gn < 1e-9:
        return 1.0
    cos_t = float(np.clip(np.dot(h, g / gn), -1.0, 1.0))
    sin2 = 1.0 - cos_t * cos_t  # sin^2 of the aspect angle
    return RCS_FRONT + (RCS_SIDE - RCS_FRONT) * sin2


def compute_hazard_attenuation(
    distance: float,
    radius: float,
    intensity: float = 1.0,
    decay_type: str = "quadratic",
    boundary_detectability: float = RADAR_BOUNDARY_DETECTABILITY,
) -> float:
    """
    Compute spatial risk / detectability attenuation with distance.

    Parameters:
        distance: Euclidean distance from hazard source.
        radius: Maximum influence / detection radius (R_max). Range-gated: risk
            is exactly 0 for distance >= radius.
        intensity: Peak intensity (for 'radar' it acts as the normalized RCS
            ratio sigma/sigma_ref that scales the returned detectability).
        decay_type: 'radar' (normalized inverse-R^4 radar-equation model),
            'quadratic', 'linear', or 'gaussian'.
        boundary_detectability: normalized detectability at R_max for the radar
            model (the calibrated detection threshold).

    Returns:
        float in [0, 1] representing the attenuated risk / detectability.

    Notes:
        The 'radar' model follows the free-space radar equation, where received
        signal-to-noise scales as sigma / R^4 (Drones 2026, 10, 469, Eq. 27).
        Normalizing so that detectability equals `boundary_detectability` at
        R_max gives:

            M(r) = intensity * boundary_detectability * (R_max / r)^4,  clipped to [0, 1]

        This produces a physically-shaped field: saturated near the emitter and
        falling off very sharply with range, unlike the gentler (1 - d/R) forms.
    """
    if distance >= radius or radius <= 0:
        return 0.0

    if decay_type == "radar":
        r = max(distance, 1e-6)
        detect = intensity * boundary_detectability * (radius / r) ** 4
        return float(np.clip(detect, 0.0, 1.0))

    norm_d = distance / radius  # in [0, 1)

    if decay_type == "quadratic":
        # Quadratic decay: (1 - d/R)^2
        return float(intensity * ((1.0 - norm_d) ** 2))
    elif decay_type == "linear":
        # Linear decay: (1 - d/R)
        return float(intensity * (1.0 - norm_d))
    elif decay_type == "gaussian":
        # Gaussian decay: exp(-3 * (d/R)^2) ensuring smooth drop to ~5% at boundary
        return float(intensity * np.exp(-3.0 * (norm_d ** 2)))
    else:
        raise ValueError(f"Unknown decay_type '{decay_type}'. Choose 'radar', 'quadratic', 'linear', or 'gaussian'.")


def compute_point_risk(
    point: Union[Tuple[float, float, float], np.ndarray],
    hazards: List[HazardSite],
    terrain: Optional[np.ndarray] = None,
    apply_los: bool = True,
    los_samples: int = 30,
    heading: Optional[np.ndarray] = None,
) -> float:
    """
    Compute total aggregate risk at a specific 3D point (x, y, z).

    Uses the probabilistic independence union formula:
        R_total = 1 - prod_i (1 - R_i)

    If `heading` (a 2D flight-direction vector) is supplied, each radar's
    contribution is additionally scaled by the anisotropic-RCS aspect factor —
    the aircraft is more detectable broadside than nose-/tail-on. Passing None
    (the default) leaves detection isotropic, preserving the planner's behaviour.
    """
    x = float(point[0])
    y = float(point[1])
    z = float(point[2])

    # If point is below terrain surface, it is physically occluded / invalid
    if terrain is not None:
        ground = terrain_height(terrain, x, y)
        if z < ground:
            return 1.0  # Collision / invalid state

    prod_survival = 1.0

    for hazard in hazards:
        # Fast 2D bounding box check
        if abs(x - hazard.x) >= hazard.radius or abs(y - hazard.y) >= hazard.radius:
            continue

        haz_pos = hazard.get_position(terrain)
        dx = x - haz_pos[0]
        dy = y - haz_pos[1]
        dz = (z - haz_pos[2]) * Z_RISK_SCALE   # de-weight vertical (see Z_RISK_SCALE)
        dist = np.sqrt(dx * dx + dy * dy + dz * dz)

        if dist >= hazard.radius:
            continue

        # 3D detection-volume gating: minimum altitude, horizontal sector, vertical cone.
        if not hazard.covers(x, y, z, haz_pos):
            continue

        # Check LOS terrain masking if applicable
        if hazard.requires_los and apply_los and terrain is not None:
            visible, _ = check_los(terrain, haz_pos, (x, y, z), samples=los_samples)
            if not visible:
                # Occluded by terrain (in radar/sensor shadow)
                continue

        # Compute attenuated detectability, then scale by threat-type lethality.
        r_i = compute_hazard_attenuation(
            distance=dist,
            radius=hazard.radius,
            intensity=hazard.intensity,
            decay_type=hazard.decay_type
        )
        r_i = r_i * THREAT_TYPE_WEIGHT.get(hazard.threat_type, 1.0)
        if heading is not None:
            r_i *= rcs_aspect_factor((x, y), heading, (haz_pos[0], haz_pos[1]))
        r_i = min(1.0, r_i)

        prod_survival *= (1.0 - r_i)

    total_risk = 1.0 - prod_survival
    return float(np.clip(total_risk, 0.0, 1.0))


def compute_risk_slice(
    terrain: np.ndarray,
    hazards: List[HazardSite],
    altitude_asl: float,
    apply_los: bool = True,
    subsample_step: int = 1,
    los_samples: int = 60
) -> np.ndarray:
    """
    Compute a 2D horizontal slice of the risk field at a constant ASL altitude.

    Parameters:
        terrain: 2D elevation grid.
        hazards: List of HazardSite instances.
        altitude_asl: Fixed altitude ASL to evaluate.
        apply_los: Whether to check terrain masking.
        subsample_step: Grid step for resolution vs speed tradeoff.
        los_samples: Samples along LOS ray.

    Returns:
        2D numpy array of risk values in [0, 1].
    """
    ny, nx = terrain.shape
    y_indices = np.arange(0, ny, subsample_step)
    x_indices = np.arange(0, nx, subsample_step)

    risk_slice = np.zeros((len(y_indices), len(x_indices)), dtype=float)

    for j, y in enumerate(y_indices):
        for i, x in enumerate(x_indices):
            risk_slice[j, i] = compute_point_risk(
                point=(float(x), float(y), float(altitude_asl)),
                hazards=hazards,
                terrain=terrain,
                apply_los=apply_los,
                los_samples=los_samples
            )

    return risk_slice


def compute_risk_grid_3d(
    terrain: np.ndarray,
    hazards: List[HazardSite],
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    z_layers: np.ndarray,
    apply_los: bool = True,
    los_samples: int = 50
) -> np.ndarray:
    """
    Compute a full 3D discrete risk tensor (Z, Y, X) for planning grids.

    Parameters:
        terrain: 2D elevation grid.
        hazards: List of HazardSite instances.
        x_grid: 1D array of x coordinates.
        y_grid: 1D array of y coordinates.
        z_layers: 1D array of discrete ASL altitude layers.
        apply_los: Whether to evaluate terrain masking.
        los_samples: Ray sampling fidelity.

    Returns:
        3D numpy array with shape (len(z_layers), len(y_grid), len(x_grid)).
    """
    nz = len(z_layers)
    ny = len(y_grid)
    nx = len(x_grid)

    risk_tensor = np.zeros((nz, ny, nx), dtype=float)

    for k, z in enumerate(z_layers):
        for j, y in enumerate(y_grid):
            for i, x in enumerate(x_grid):
                risk_tensor[k, j, i] = compute_point_risk(
                    point=(float(x), float(y), float(z)),
                    hazards=hazards,
                    terrain=terrain,
                    apply_los=apply_los,
                    los_samples=los_samples
                )

    return risk_tensor


def create_demo_hazards(terrain: Optional[np.ndarray] = None) -> List[HazardSite]:
    """Two ridge radars on a 100x100 canvas — a self-contained fixture used by the
    unit tests' LOS-masking checks (not a runnable mission scenario)."""
    return [
        HazardSite(
            x=18.0, y=42.0, z=None, mast_height=5.0, radius=50.0,
            intensity=0.95, requires_los=True, decay_type="radar",
            name="Sensor Alpha (West Ridge)"
        ),
        HazardSite(
            x=80.0, y=65.0, z=None, mast_height=5.0, radius=45.0,
            intensity=0.90, requires_los=True, decay_type="radar",
            name="Sensor Bravo (East Basin)"
        ),
    ]


# Config for an on-demand custom real-DEM region, set via set_custom_dem()
# (main.py 'region' command) before calling load_mission_scenario("custom").
_CUSTOM_DEM = {}


def set_custom_dem(tile_path: str, row0: int, col0: int, title: str, desc: str):
    """Register a downloaded DEM tile + crop for the 'custom' scenario."""
    _CUSTOM_DEM.update(tile=tile_path, row0=row0, col0=col0, title=title, desc=desc)


def _find_summits(terrain: np.ndarray, k: int = 8, min_sep_frac: float = 0.16,
                  border_frac: float = 0.10) -> List[tuple]:
    """
    Return up to `k` dominant high points as (x=col, y=row) grid positions, tallest
    first. Each is the highest remaining interior cell at least
    `min_sep_frac * max(nx, ny)` from every already-chosen point (non-maximum
    suppression), so they fall on distinct ridgelines/summits, not one peak.
    """
    ny, nx = terrain.shape
    bx, by = max(1, int(nx * border_frac)), max(1, int(ny * border_frac))
    min_sep = min_sep_frac * max(nx, ny)
    work = terrain.astype(float).copy()
    work[:by, :] = work[ny - by:, :] = -np.inf
    work[:, :bx] = work[:, nx - bx:] = -np.inf
    chosen: List[tuple] = []
    for flat in np.argsort(work, axis=None)[::-1]:
        yy, xx = divmod(int(flat), nx)
        if not np.isfinite(work[yy, xx]):
            break
        if all((xx - cx) ** 2 + (yy - cy) ** 2 >= min_sep ** 2 for cx, cy in chosen):
            chosen.append((xx, yy))
            if len(chosen) >= k:
                break
    return chosen


def auto_place_hazards(terrain: np.ndarray, start_xy, goal_xy) -> List[HazardSite]:
    """
    Place the threat laydown from the TERRAIN and the mission corridor, instead of
    fixed template coordinates — so every real region gets a threat picture that
    actually fits its ridgelines and the direct base->target path:

      * Search Radar (summit)  — the tallest summit that sits on/near the direct
        corridor, so it commands the straight-line path and forces a detour.
      * Sector Radar (ridge)   — a second dominant summit set apart from the first,
        with its coverage arc aimed at the corridor midpoint.
      * SAM (corridor)         — high ground commanding the corridor midpoint (kept
        clear of the radars), with a minimum-engagement-altitude floor.

    All positions are in grid units; z=None keeps each emitter on the ground it
    stands on. Fully deterministic (no randomness).
    """
    ny, nx = terrain.shape
    span = float(max(nx, ny))
    sx, sy = float(start_xy[0]), float(start_xy[1])
    gx, gy = float(goal_xy[0]), float(goal_xy[1])

    dvec = np.array([gx - sx, gy - sy], dtype=float)
    dlen = float(np.hypot(dvec[0], dvec[1])) + 1e-9
    dhat = dvec / dlen
    mid = (int(round((sx + gx) / 2.0)), int(round((sy + gy) / 2.0)))

    def perp_dist(px, py):
        return abs((px - sx) * dhat[1] - (py - sy) * dhat[0])   # distance to line

    def along(px, py):
        return (px - sx) * dhat[0] + (py - sy) * dhat[1]        # projection

    def side(px, py):
        return np.sign((px - sx) * dhat[1] - (py - sy) * dhat[0])

    summits = _find_summits(terrain, k=8, min_sep_frac=0.16)
    if not summits:                                             # near-flat fallback
        summits = [mid]

    # Primary: the tallest summit near the corridor between base and target — it
    # commands the straight-line path, so the shortest route is exposed over it.
    corridor = [(x, y) for (x, y) in summits
                if perp_dist(x, y) <= 0.22 * span and 0.15 * dlen <= along(x, y) <= 0.85 * dlen]
    primary = corridor[0] if corridor else summits[0]

    # Secondary: tallest remaining summit well separated from the primary,
    # preferring the opposite side of the corridor for wider combined coverage.
    rest = [p for p in summits
            if (p[0] - primary[0]) ** 2 + (p[1] - primary[1]) ** 2 >= (0.25 * span) ** 2]
    opp = [p for p in rest if side(p[0], p[1]) != side(primary[0], primary[1])]
    secondary = (opp or rest or [primary])[0]

    # SAM: high ground commanding the corridor midpoint, kept clear of both radars
    # (so threats don't stack when the dominant summit already sits mid-corridor).
    radars = [primary, secondary]
    sam_minsep = 0.18 * span
    w = int(0.10 * span)
    sam_xy, sam_best = mid, -np.inf
    for frac in (0.4, 0.5, 0.6, 0.45, 0.55):
        cx, cy = sx + frac * dvec[0], sy + frac * dvec[1]
        x0, x1 = max(0, int(cx) - w), min(nx, int(cx) + w + 1)
        y0, y1 = max(0, int(cy) - w), min(ny, int(cy) + w + 1)
        sub = terrain[y0:y1, x0:x1]
        for flat in np.argsort(sub, axis=None)[::-1][:15]:
            sj, si = np.unravel_index(int(flat), sub.shape)
            px, py = x0 + si, y0 + sj
            if all((px - rx) ** 2 + (py - ry) ** 2 >= sam_minsep ** 2 for rx, ry in radars):
                if terrain[py, px] > sam_best:
                    sam_best, sam_xy = float(terrain[py, px]), (px, py)
                break

    def bearing_to(pt, target):
        return float(np.degrees(np.arctan2(target[1] - pt[1], target[0] - pt[0])) % 360.0)

    # Build the three emitters for a given max-range R (dx,dy in cells, dz in
    # planning units — the range gate uses the full 3D slant on the same scale).
    def make_hazards(Rval):
        return [
            HazardSite(
                x=float(primary[0]), y=float(primary[1]), z=None, mast_height=12.0,
                radius=1.05 * Rval, intensity=0.96, requires_los=True, decay_type="radar",
                threat_type="radar", name="Search Radar (summit)"),
            HazardSite(
                x=float(secondary[0]), y=float(secondary[1]), z=None, mast_height=12.0,
                radius=0.9 * Rval, intensity=0.92, requires_los=True, decay_type="radar",
                threat_type="radar", azimuth_center=bearing_to(secondary, mid),
                azimuth_width=150.0, name="Sector Radar (ridge)"),
            HazardSite(
                x=float(sam_xy[0]), y=float(sam_xy[1]), z=None, mast_height=8.0,
                radius=1.12 * Rval, intensity=0.9, requires_los=True, decay_type="radar",
                threat_type="sam", min_detect_alt=float(terrain.min()) + 90.0,
                name="SAM (corridor)"),
        ]

    # Range calibration. With the vertical de-weighting (Z_RISK_SCALE) the range is
    # a genuine horizontal footprint, but the direct corridor's exposure still
    # varies by region (some terrain funnels the base→target line through more
    # threatened ground). Scale the range down until the straight corridor at cruise
    # altitude carries a comparable target exposure — so EVERY region ends up with
    # the same "direct path is threatened, a survivable detour exists" balance, and
    # the survivability spread (the best-route signal) never collapses to all-0%.
    AGL = 55.0
    ts = np.linspace(0.0, 1.0, 26)
    line = [(sx + t * (gx - sx), sy + t * (gy - sy)) for t in ts]
    alts = [terrain_height(terrain, x, y) + AGL for (x, y) in line]
    seg = dlen / (len(ts) - 1)

    def corridor_exposure(hz):
        rs = [compute_point_risk((x, y, a), hz, terrain=terrain, apply_los=True, los_samples=12)
              for (x, y), a in zip(line, alts)]
        return float(sum(0.5 * (rs[i] + rs[i + 1]) * seg for i in range(len(rs) - 1)))

    base_R = 0.60 * span
    TARGET = 56.0
    R = base_R
    for scale in (1.0, 0.86, 0.72, 0.60, 0.50, 0.42, 0.34):
        R = base_R * scale
        if corridor_exposure(make_hazards(R)) <= TARGET:
            break
    return make_hazards(R)


# Real-world scale of the most recently loaded DEM crop (metres/cell is fixed by
# the Copernicus 30 m/px crop geometry: 600 px over 120 out-cells = 150 m/cell;
# the vertical relief is captured per region). Read via get_last_region_scale().
_LAST_REGION_SCALE = {"meters_per_cell": 150.0, "relief_m": 1000.0,
                      "elev_min_m": 0.0, "znorm_base": 90.0, "znorm_span": 260.0}


def get_last_region_scale() -> dict:
    """Real-world scale (metres/cell, real relief, base elevation) of the last DEM
    loaded via load_mission_scenario(). Used by the aircraft calibration layer."""
    return dict(_LAST_REGION_SCALE)


def _finalize_real_dem(raw: np.ndarray, title: str, desc: str):
    """
    Shared real-DEM scenario construction: normalize elevation into planning
    units (preserving morphology), auto-place terrain-driven threats, and set
    terrain-adaptive ingress/egress and altitude layers.
    """
    _LAST_REGION_SCALE.update(
        meters_per_cell=150.0, relief_m=float(raw.max() - raw.min()),
        elev_min_m=float(raw.min()), znorm_base=90.0, znorm_span=260.0)
    terrain = 90.0 + 260.0 * (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
    ny, nx = terrain.shape
    sx, sy = 0.10 * (nx - 1), 0.12 * (ny - 1)
    gx, gy = 0.90 * (nx - 1), 0.88 * (ny - 1)
    hazards = auto_place_hazards(terrain, (sx, sy), (gx, gy))
    start_pos = (sx, sy, terrain_height(terrain, sx, sy) + 55.0)
    goal_pos = (gx, gy, terrain_height(terrain, gx, gy) + 55.0)
    z_layers = np.linspace(float(terrain.min()) + 20.0, float(terrain.max()) + 120.0, 13)
    return terrain, hazards, start_pos, goal_pos, z_layers, title, desc


def load_mission_scenario(scenario_key: str = "dem"):
    """
    Load terrain, hazards, start/goal coordinates, and altitude layers for a scenario.

    Returns:
        tuple: (terrain, hazards, start_pos, goal_pos, z_layers, title, description)
    """
    from src.terrain import load_real_dem_region, download_copernicus_tile

    if scenario_key == "dem":
        # Ingest a REAL Copernicus GLO-30 DEM tile (N27 E086, Everest/Khumbu
        # region) through the rasterio pipeline — a genuine SRTM-class elevation
        # surface, not a synthetic one. Auto-select a navigable-relief window (with
        # valleys to route/hide through), then auto-place threats from the terrain.
        real_tile = download_copernicus_tile(27.25, 86.4)  # Everest / Khumbu, N27 E086
        raw = load_real_dem_region(real_tile, row0=2400, col0=1200, win_px=600, out_cells=120)
        real_lo, real_hi = float(raw.min()), float(raw.max())
        title = "Scenario 5: Real Copernicus DEM (Everest region) & Terrain-Placed Threats"
        desc = (f"Real Copernicus GLO-30 DEM · N27 E086 (Everest/Khumbu foothills), "
                f"true elevation {real_lo:.0f}-{real_hi:.0f} m ASL, normalized to planning units "
                f"— summit/ridge radars and a min-altitude SAM auto-placed on the terrain.")
        return _finalize_real_dem(raw, title, desc)

    if scenario_key == "ghats":
        # Real Copernicus GLO-30 DEM, N10 E077 (Western Ghats / Anamalai). This
        # region has an isolated dominant summit, so the auto-placed summit radar
        # forces a genuine valley detour — a sharp distance-vs-risk trade-off.
        real_tile = download_copernicus_tile(10.42, 77.13)  # Western Ghats / Anamalai, N10 E077
        raw = load_real_dem_region(real_tile, row0=1800, col0=300, win_px=600, out_cells=120)
        real_lo, real_hi = float(raw.min()), float(raw.max())
        title = "Scenario 6: Real Copernicus DEM (Western Ghats / Anamalai) & Terrain-Placed Threats"
        desc = (f"Real Copernicus GLO-30 DEM · N10 E077 (Western Ghats, Anamalai massif), "
                f"true elevation {real_lo:.0f}-{real_hi:.0f} m ASL, normalized to planning units "
                f"— the auto-placed summit radar forces a valley detour.")
        return _finalize_real_dem(raw, title, desc)

    if scenario_key == "custom":
        # User-chosen real region (main.py 'region' downloads the tile + picks the
        # crop, then registers it via set_custom_dem). Threats auto-place from the
        # region's own ridgelines and the base->target corridor.
        if not _CUSTOM_DEM:
            raise RuntimeError("No custom DEM registered — call set_custom_dem() first "
                               "(use: python main.py region --lat .. --lon ..).")
        raw = load_real_dem_region(_CUSTOM_DEM["tile"], row0=_CUSTOM_DEM["row0"],
                                   col0=_CUSTOM_DEM["col0"], win_px=600, out_cells=120)
        return _finalize_real_dem(raw, _CUSTOM_DEM["title"], _CUSTOM_DEM["desc"])

    raise ValueError(
        f"Unknown scenario '{scenario_key}'. Choose 'dem' (Everest), 'ghats' "
        f"(Western Ghats), or 'custom' (any region via set_custom_dem)."
    )
