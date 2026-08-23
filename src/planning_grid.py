"""
AerX Labs — 3D Planning Grid & Multi-Objective Cost Engine
src/planning_grid.py

Implements a layered 3D graph state space q = (x, y, z_ASL) over DEM terrain.
Enforces hard kinematic/terrain feasibility constraints (AGL clearance, climb limits)
and calculates multi-objective traversal costs (distance, generic spatial risk, energy proxy, AGL).
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np

from src.los import terrain_height
from src.risk_field import HazardSite, compute_point_risk


# Type alias for a 3D discrete grid node: (ix, iy, iz)
Node3D = Tuple[int, int, int]


@dataclass
class CostWeights:
    """
    Weights for multi-objective edge cost evaluation:
      Total Cost = w_distance * C_dist + w_risk * C_risk + w_energy * C_energy + w_altitude * C_alt

    Preset configurations support:
      - Distance-focused (w_dist=1.0, w_risk=0.0)
      - Risk-focused (w_dist=0.5, w_risk=10.0)
      - Balanced (w_dist=1.0, w_risk=4.0)
    """
    w_distance: float = 1.0
    w_risk: float = 5.0
    w_energy: float = 1.0
    w_altitude: float = 0.3
    climb_penalty_factor: float = 2.0  # Extra energy cost multiplier for climbing edges

    @classmethod
    def distance_focused(cls) -> "CostWeights":
        return cls(w_distance=1.0, w_risk=0.0, w_energy=0.2, w_altitude=0.0)

    @classmethod
    def risk_focused(cls) -> "CostWeights":
        return cls(w_distance=0.6, w_risk=6.0, w_energy=0.5, w_altitude=0.4)

    @classmethod
    def balanced(cls) -> "CostWeights":
        return cls(w_distance=1.0, w_risk=4.0, w_energy=1.0, w_altitude=0.3)


class PlanningGrid3D:
    """
    3D Layered Planning Space and Graph over Elevation Terrain.

    Discretizes the continuous 3D airspace into grid coordinates:
      ix in [0, nx-1]
      iy in [0, ny-1]
      iz in [0, nz-1] (mapping to discrete altitude layers ASL in meters)
    """

    def __init__(
        self,
        terrain: np.ndarray,
        z_layers: np.ndarray,
        min_agl: float = 15.0,
        max_agl: float = 300.0,
        max_climb_slope: float = 0.75,
        hazards: Optional[List[HazardSite]] = None,
        weights: Optional[CostWeights] = None,
        xy_scale: float = 1.0,
        detection_threshold: Optional[float] = None
    ):
        """
        Parameters:
            terrain: 2D elevation array (ny, nx)
            z_layers: 1D array of discrete ASL altitudes (in meters)
            min_agl: Minimum allowable Above Ground Level clearance (hard constraint)
            max_agl: Maximum allowable AGL ceiling
            max_climb_slope: Maximum allowable climb/descent gradient (|dz| / dxy)
            hazards: Active hazard/sensor sites
            weights: Multi-objective cost weights
            xy_scale: Real-world metric distance per grid cell (meters per unit)
            detection_threshold: Optional hard-feasibility detection cutoff in
                [0, 1]. When set, any node whose aggregate detectability meets or
                exceeds it is marked terrain-infeasible (a hard keep-out), so the
                planner is forbidden from entering it rather than merely paying a
                soft risk cost. This implements the hard-feasibility/soft-risk
                formulation of Drones 2026, 10, 469 (detection-threshold as a hard
                constraint). None disables it (pure soft-risk planning).
        """
        self.terrain = np.asarray(terrain, dtype=float)
        self.ny, self.nx = self.terrain.shape
        self.z_layers = np.asarray(z_layers, dtype=float)
        self.nz = len(self.z_layers)

        self.min_agl = float(min_agl)
        self.max_agl = float(max_agl)
        self.max_climb_slope = float(max_climb_slope)
        self.hazards = hazards or []
        self.weights = weights or CostWeights.balanced()
        self.xy_scale = float(xy_scale)
        self.detection_threshold = detection_threshold

        # Precompute feasibility tensor (nz, ny, nx) and risk cache
        self.feasibility_mask = np.zeros((self.nz, self.ny, self.nx), dtype=bool)
        self.risk_cache = np.zeros((self.nz, self.ny, self.nx), dtype=float)
        self._precompute_grid()
        self._precompute_neighbors()

    def _precompute_grid(self) -> None:
        """Precomputes feasibility and spatial risk at each discrete node."""
        # 1. Feasibility mask
        for iz, z in enumerate(self.z_layers):
            agl_layer = z - self.terrain
            self.feasibility_mask[iz] = (agl_layer >= self.min_agl) & (agl_layer <= self.max_agl)

        # 2. Risk cache (only compute for feasible nodes within hazard influence bounds)
        self.risk_cache.fill(0.0)
        if not self.hazards:
            return

        self._compute_risk_cache()

        # 3. Hard-feasibility detection keep-out: nodes whose detectability meets
        #    or exceeds the threshold are forbidden entirely (hard constraint).
        if self.detection_threshold is not None:
            self.feasibility_mask &= (self.risk_cache < self.detection_threshold)

    def _compute_risk_cache(self) -> None:
        """Populate the per-node aggregate detectability within hazard bounds."""
        for hazard in self.hazards:
            haz_pos = hazard.get_position(self.terrain)
            # Compute bounding box in grid indices
            ix_min = max(0, int(np.floor((hazard.x - hazard.radius) / self.xy_scale)))
            ix_max = min(self.nx - 1, int(np.ceil((hazard.x + hazard.radius) / self.xy_scale)))
            iy_min = max(0, int(np.floor((hazard.y - hazard.radius) / self.xy_scale)))
            iy_max = min(self.ny - 1, int(np.ceil((hazard.y + hazard.radius) / self.xy_scale)))

            for iz, z in enumerate(self.z_layers):
                for iy in range(iy_min, iy_max + 1):
                    for ix in range(ix_min, ix_max + 1):
                        if not self.feasibility_mask[iz, iy, ix]:
                            continue
                        pt = self.node_to_pos((ix, iy, iz))
                        # Only compute if within 3D sphere
                        if np.linalg.norm(pt - haz_pos) < hazard.radius:
                            self.risk_cache[iz, iy, ix] = compute_point_risk(
                                point=pt,
                                hazards=self.hazards,
                                terrain=self.terrain,
                                apply_los=True,
                                los_samples=25
                            )

    def node_to_pos(self, node: Node3D) -> np.ndarray:
        """Convert discrete node (ix, iy, iz) to continuous (x, y, z_ASL) coordinates."""
        ix, iy, iz = node
        return np.array([ix * self.xy_scale, iy * self.xy_scale, self.z_layers[iz]], dtype=float)

    def pos_to_node(self, pos: Union[Tuple[float, float, float], np.ndarray]) -> Node3D:
        """Convert continuous coordinates (x, y, z) to the nearest discrete node (ix, iy, iz)."""
        x, y, z = pos
        ix = int(np.clip(round(x / self.xy_scale), 0, self.nx - 1))
        iy = int(np.clip(round(y / self.xy_scale), 0, self.ny - 1))
        iz = int(np.argmin(np.abs(self.z_layers - z)))
        return (ix, iy, iz)

    def is_feasible(self, node: Node3D) -> bool:
        """Check if a discrete node satisfies boundary and terrain clearance constraints."""
        ix, iy, iz = node
        if 0 <= ix < self.nx and 0 <= iy < self.ny and 0 <= iz < self.nz:
            return bool(self.feasibility_mask[iz, iy, ix])
        return False

    def get_agl(self, node: Node3D) -> float:
        """Get Above Ground Level clearance for a node."""
        ix, iy, iz = node
        ground = self.terrain[iy, ix]
        return float(self.z_layers[iz] - ground)

    def get_node_risk(self, node: Node3D) -> float:
        """Retrieve precomputed risk at node."""
        ix, iy, iz = node
        return float(self.risk_cache[iz, iy, ix])

    def risk_at(self, x: float, y: float, z: float) -> float:
        """
        Fast continuous risk lookup by trilinear interpolation of the precomputed
        risk cache. Orders of magnitude cheaper than a live LOS-sampled
        `compute_point_risk`, at the cost of grid-resolution smoothing. Used by
        the RRT* refiner and other inner-loop consumers.
        """
        fx = np.clip(x / self.xy_scale, 0, self.nx - 1)
        fy = np.clip(y / self.xy_scale, 0, self.ny - 1)
        x0, y0 = int(fx), int(fy)
        x1, y1 = min(x0 + 1, self.nx - 1), min(y0 + 1, self.ny - 1)
        tx, ty = fx - x0, fy - y0

        # Locate the bracketing altitude layers.
        if z <= self.z_layers[0]:
            z0 = z1 = 0
            tz = 0.0
        elif z >= self.z_layers[-1]:
            z0 = z1 = self.nz - 1
            tz = 0.0
        else:
            z1 = int(np.searchsorted(self.z_layers, z))
            z0 = z1 - 1
            tz = (z - self.z_layers[z0]) / (self.z_layers[z1] - self.z_layers[z0])

        c = self.risk_cache
        def bil(iz):
            return (
                (1 - tx) * (1 - ty) * c[iz, y0, x0]
                + tx * (1 - ty) * c[iz, y0, x1]
                + (1 - tx) * ty * c[iz, y1, x0]
                + tx * ty * c[iz, y1, x1]
            )
        return float((1 - tz) * bil(z0) + tz * bil(z1))

    def is_feasible_pos(self, x: float, y: float, z: float) -> bool:
        """Continuous terrain-clearance + detection feasibility check for a point."""
        if not (0 <= x <= (self.nx - 1) * self.xy_scale and 0 <= y <= (self.ny - 1) * self.xy_scale):
            return False
        agl = z - terrain_height(self.terrain, x / self.xy_scale, y / self.xy_scale)
        if agl < self.min_agl or agl > self.max_agl:
            return False
        if self.detection_threshold is not None and self.risk_at(x, y, z) >= self.detection_threshold:
            return False
        return True

    def _precompute_neighbors(self) -> None:
        """Precompute the 26 neighbor offsets and geometric base distances."""
        self._neighbor_offsets = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    d_xy = np.hypot(dx * self.xy_scale, dy * self.xy_scale)
                    dz_val = (self.z_layers[1] - self.z_layers[0]) * dz if self.nz > 1 else 0.0
                    if d_xy > 0 and (abs(dz_val) / d_xy) > self.max_climb_slope:
                        continue
                    dist = float(np.sqrt(d_xy * d_xy + dz_val * dz_val))
                    dz_step = float(dz_val)
                    self._neighbor_offsets.append((dx, dy, dz, dist, dz_step))

    def get_neighbors(self, node: Node3D) -> List[Tuple[Node3D, float]]:
        """Fast lookup of all feasible 26-connected 3D neighbors and precalculated edge costs."""
        ix, iy, iz = node
        if not self.feasibility_mask[iz, iy, ix]:
            return []

        w_dist = self.weights.w_distance
        w_risk = self.weights.w_risk
        w_energy = self.weights.w_energy
        w_alt = self.weights.w_altitude
        climb_factor = self.weights.climb_penalty_factor
        max_agl = self.max_agl

        r_u = self.risk_cache[iz, iy, ix]
        agl_u = float(self.z_layers[iz] - self.terrain[iy, ix])

        neighbors = []
        for dx, dy, dz, dist, dz_val in self._neighbor_offsets:
            nx_i = ix + dx
            ny_i = iy + dy
            nz_i = iz + dz

            if not (0 <= nx_i < self.nx and 0 <= ny_i < self.ny and 0 <= nz_i < self.nz):
                continue
            if not self.feasibility_mask[nz_i, ny_i, nx_i]:
                continue

            r_v = self.risk_cache[nz_i, ny_i, nx_i]
            avg_risk = 0.5 * (r_u + r_v)
            c_risk = avg_risk * dist

            # Climb energy
            cf = (1.0 + climb_factor * (dz_val / dist)) if dz_val > 0 else 1.0
            c_energy = dist * cf

            # Altitude penalty
            agl_v = float(self.z_layers[nz_i] - self.terrain[ny_i, nx_i])
            c_alt = (0.5 * (agl_u + agl_v) / max_agl) * dist

            cost = w_dist * dist + w_risk * c_risk + w_energy * c_energy + w_alt * c_alt
            neighbors.append(((nx_i, ny_i, nz_i), float(cost)))

        return neighbors

    def compute_edge_cost(self, u: Node3D, v: Node3D) -> float:
        """Compute cost between node u and node v."""
        if not self.is_feasible(u) or not self.is_feasible(v):
            return float("inf")

        dx = (v[0] - u[0]) * self.xy_scale
        dy = (v[1] - u[1]) * self.xy_scale
        dz = self.z_layers[v[2]] - self.z_layers[u[2]]
        dist = float(np.sqrt(dx * dx + dy * dy + dz * dz))
        if dist == 0:
            return 0.0

        r_u = self.risk_cache[u[2], u[1], u[0]]
        r_v = self.risk_cache[v[2], v[1], v[0]]
        c_risk = 0.5 * (r_u + r_v) * dist

        cf = (1.0 + self.weights.climb_penalty_factor * (dz / dist)) if dz > 0 else 1.0
        c_energy = dist * cf

        agl_u = float(self.z_layers[u[2]] - self.terrain[u[1], u[0]])
        agl_v = float(self.z_layers[v[2]] - self.terrain[v[1], v[0]])
        c_alt = (0.5 * (agl_u + agl_v) / self.max_agl) * dist

        return float(
            self.weights.w_distance * dist
            + self.weights.w_risk * c_risk
            + self.weights.w_energy * c_energy
            + self.weights.w_altitude * c_alt
        )

    def add_hazard(self, hazard: HazardSite) -> List[Node3D]:
        """
        Insert a single pop-up hazard and recompute risk only inside its region
        of influence, returning the list of nodes whose cost changed.

        This is the enabling primitive for D* Lite incremental replanning: only
        the returned nodes' vertices need re-evaluation, so a mid-mission threat
        can be handled by a localized repair instead of a full re-plan.
        """
        self.hazards = list(self.hazards) + [hazard]
        ix_min = max(0, int(np.floor((hazard.x - hazard.radius) / self.xy_scale)))
        ix_max = min(self.nx - 1, int(np.ceil((hazard.x + hazard.radius) / self.xy_scale)))
        iy_min = max(0, int(np.floor((hazard.y - hazard.radius) / self.xy_scale)))
        iy_max = min(self.ny - 1, int(np.ceil((hazard.y + hazard.radius) / self.xy_scale)))

        affected: List[Node3D] = []
        for iz in range(self.nz):
            for iy in range(iy_min, iy_max + 1):
                for ix in range(ix_min, ix_max + 1):
                    if not self.feasibility_mask[iz, iy, ix]:
                        continue
                    pt = self.node_to_pos((ix, iy, iz))
                    new_r = compute_point_risk(pt, self.hazards, terrain=self.terrain,
                                               apply_los=True, los_samples=25)
                    if abs(new_r - self.risk_cache[iz, iy, ix]) > 1e-9:
                        self.risk_cache[iz, iy, ix] = new_r
                        affected.append((ix, iy, iz))

        # Re-apply the hard detection keep-out over the touched region.
        if self.detection_threshold is not None:
            for (ix, iy, iz) in affected:
                if self.risk_cache[iz, iy, ix] >= self.detection_threshold:
                    self.feasibility_mask[iz, iy, ix] = False
        return affected

    def update_hazards(self, new_hazards: List[HazardSite]) -> None:
        """
        Dynamically update the hazard configuration and recalculate the risk cache.
        This provides the mechanism for dynamic map updates and D* Lite replanning.
        """
        self.hazards = new_hazards
        for iz, z in enumerate(self.z_layers):
            for iy in range(self.ny):
                for ix in range(self.nx):
                    if self.feasibility_mask[iz, iy, ix] and self.hazards:
                        pt = self.node_to_pos((ix, iy, iz))
                        self.risk_cache[iz, iy, ix] = compute_point_risk(
                            point=pt,
                            hazards=self.hazards,
                            terrain=self.terrain,
                            apply_los=True,
                            los_samples=25
                        )
                    else:
                        self.risk_cache[iz, iy, ix] = 0.0
