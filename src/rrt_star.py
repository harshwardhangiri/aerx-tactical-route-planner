"""
AerX Labs — Risk-Aware RRT* Local Trajectory Refinement
src/rrt_star.py

Implements a 3D RRT* (Rapidly-exploring Random Tree, asymptotically optimal
variant) that refines the discrete D* Lite corridor into a smoother, shorter,
continuous trajectory while respecting:

  * terrain clearance (minimum AGL) along every edge,
  * an optional hard detection keep-out (edges may not pass through nodes whose
    detectability exceeds a threshold), and
  * a risk-aware edge cost (distance + weighted integrated threat exposure).

Refinement strategy (per the concept document: D* Lite -> RRT* -> smoothing):
    The random samples are biased into an ellipsoidal tube around the global
    D* Lite path (informed sampling), so the tree explores the already-good
    corridor instead of the whole map. The best tree path is then shortcut and
    returned. This keeps the continuous refinement close to the vetted global
    route while removing grid stair-stepping.

References:
    - Karaman & Frautwohl, "Sampling-based algorithms for optimal motion
      planning" (RRT*).
    - Gammell et al., "Informed RRT*" (ellipsoidal informed sampling).
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from src.los import terrain_height
from src.risk_field import HazardSite, compute_point_risk


@dataclass
class _Node:
    pos: np.ndarray
    parent: int = -1
    cost: float = 0.0
    indir: Optional[np.ndarray] = None  # unit horizontal heading of the incoming edge


@dataclass
class RRTStarConfig:
    """Tuning parameters for the risk-aware RRT* refiner."""
    max_iterations: int = 1500
    step_size: float = 6.0            # max steering distance per extension (m)
    goal_sample_rate: float = 0.10    # probability of sampling the goal directly
    neighbor_radius: float = 14.0     # rewiring neighborhood radius (m)
    max_neighbors: int = 16           # cap on rewire candidates (keeps RRT* near-linear)
    min_agl: float = 15.0             # hard terrain clearance floor (m)
    tube_radius: float = 18.0         # informed-sampling tube radius around the guide path (m)
    risk_weight: float = 40.0         # weight of integrated risk in the edge cost
    edge_samples: int = 8             # collision/risk samples per edge
    detection_threshold: Optional[float] = None  # hard keep-out cutoff, or None
    goal_tolerance: float = 6.0       # distance to goal counted as reached (m)
    seed: Optional[int] = 7
    # Helicopter/NOE kinematic limits enforced on every tree edge:
    max_turn_deg: float = 50.0        # max heading change between consecutive edges
    max_climb_rate: float = 0.7       # max |dz| / horizontal distance (climb gradient)


class RiskAwareRRTStar:
    """Risk-aware RRT* refiner over a terrain + hazard field."""

    def __init__(
        self,
        terrain: np.ndarray,
        hazards: List[HazardSite],
        guide_path: np.ndarray,
        config: Optional[RRTStarConfig] = None,
        risk_sampler=None,
    ):
        """
        Parameters:
            risk_sampler: optional fast callable risk_sampler(x, y, z) -> float.
                When provided (e.g. PlanningGrid3D.risk_at, a trilinear lookup of
                the precomputed cache), the refiner avoids live LOS-sampled risk
                computation and runs orders of magnitude faster. Falls back to
                compute_point_risk when None.
        """
        self.terrain = np.asarray(terrain, dtype=float)
        self.ny, self.nx = self.terrain.shape
        self.hazards = hazards or []
        self.guide = np.asarray(guide_path, dtype=float)
        self.cfg = config or RRTStarConfig()
        self._rng = np.random.default_rng(self.cfg.seed)
        self._risk_sampler = risk_sampler

        # Cumulative arc length of the guide, for informed tube sampling.
        seg = np.linalg.norm(np.diff(self.guide, axis=0), axis=1)
        self._guide_s = np.concatenate([[0.0], np.cumsum(seg)])
        self._guide_len = float(self._guide_s[-1]) if len(seg) else 0.0

        # Altitude sampling bounds taken from the guide plus clearance headroom.
        self._z_lo = float(self.guide[:, 2].min()) - 10.0
        self._z_hi = float(self.guide[:, 2].max()) + 20.0

    # ----------------------------------------------------------------- helpers
    def _ground(self, x: float, y: float) -> float:
        return terrain_height(self.terrain, x, y)

    def _risk(self, p: np.ndarray) -> float:
        if self._risk_sampler is not None:
            return float(self._risk_sampler(p[0], p[1], p[2]))
        if not self.hazards:
            return 0.0
        return compute_point_risk(p, self.hazards, terrain=self.terrain, apply_los=True)

    def _point_ok(self, p: np.ndarray) -> bool:
        """A point is valid if it clears terrain and is below the detection cutoff."""
        if not (0 <= p[0] <= self.nx - 1 and 0 <= p[1] <= self.ny - 1):
            return False
        if p[2] - self._ground(p[0], p[1]) < self.cfg.min_agl:
            return False
        if self.cfg.detection_threshold is not None and self._risk(p) >= self.cfg.detection_threshold:
            return False
        return True

    def _edge_ok(self, a: np.ndarray, b: np.ndarray) -> bool:
        """Terrain-clearance (+ optional detection) check sampled along a->b."""
        for t in np.linspace(0.0, 1.0, self.cfg.edge_samples):
            if not self._point_ok(a + t * (b - a)):
                return False
        return True

    def _heading(self, a: np.ndarray, b: np.ndarray) -> Optional[np.ndarray]:
        """Unit horizontal heading vector of edge a->b (None if degenerate)."""
        d = np.array([b[0] - a[0], b[1] - a[1]], dtype=float)
        n = np.linalg.norm(d)
        return d / n if n > 1e-9 else None

    def _kinematic_ok(self, a: np.ndarray, indir: Optional[np.ndarray], b: np.ndarray) -> bool:
        """
        Enforce fixed-wing/helicopter feasibility on edge a->b:
          * climb gradient |dz|/horizontal within max_climb_rate,
          * heading change from the incoming direction within max_turn_deg.
        """
        dz = abs(b[2] - a[2])
        horiz = float(np.hypot(b[0] - a[0], b[1] - a[1]))
        if horiz < 1e-6:
            return dz < 1e-6  # pure vertical hop is not flyable
        if dz / horiz > self.cfg.max_climb_rate:
            return False
        if indir is not None:
            newdir = self._heading(a, b)
            if newdir is not None:
                cosang = float(np.clip(np.dot(indir, newdir), -1.0, 1.0))
                if np.degrees(np.arccos(cosang)) > self.cfg.max_turn_deg:
                    return False
        return True

    def _edge_risk(self, a: np.ndarray, b: np.ndarray) -> float:
        """Integrated threat exposure along a->b (trapezoidal)."""
        if not self.hazards:
            return 0.0
        ts = np.linspace(0.0, 1.0, self.cfg.edge_samples)
        rs = np.array([self._risk(a + t * (b - a)) for t in ts])
        seg = float(np.linalg.norm(b - a))
        if len(ts) < 2:
            return 0.0
        # Trapezoidal integral over arc length (NumPy 2.x-safe, no np.trapz).
        dx = seg / (len(ts) - 1)
        return float((rs[:-1] + rs[1:]).sum() * 0.5 * dx)

    def _edge_cost(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(b - a)) + self.cfg.risk_weight * self._edge_risk(a, b)

    def _sample(self, goal: np.ndarray) -> np.ndarray:
        if self._rng.random() < self.cfg.goal_sample_rate:
            return goal.copy()
        # Informed tube sample: pick a point along the guide, offset within a ball.
        s = self._rng.uniform(0.0, self._guide_len) if self._guide_len > 0 else 0.0
        base = np.array([np.interp(s, self._guide_s, self.guide[:, d]) for d in range(3)])
        offset = self._rng.normal(0.0, self.cfg.tube_radius / 2.0, size=3)
        offset[2] *= 0.5  # tighter vertical spread
        return base + offset

    # ------------------------------------------------------------------- solve
    def refine(self, start: np.ndarray, goal: np.ndarray) -> Optional[np.ndarray]:
        """Build the tree and return the shortcutted (N,3) refined path, or None."""
        start = np.asarray(start, dtype=float)
        goal = np.asarray(goal, dtype=float)
        nodes: List[_Node] = [_Node(pos=start, parent=-1, cost=0.0)]
        # Growing contiguous position buffer for vectorized nearest/neighbor queries.
        pos_buf = np.empty((self.cfg.max_iterations + 2, 3), dtype=float)
        pos_buf[0] = start
        n_pts = 1
        best_goal_idx = -1
        best_goal_cost = float("inf")

        for _ in range(self.cfg.max_iterations):
            rnd = self._sample(goal)
            # nearest node (vectorized)
            deltas = pos_buf[:n_pts] - rnd
            d2 = np.einsum("ij,ij->i", deltas, deltas)
            nearest = int(np.argmin(d2))
            direction = rnd - nodes[nearest].pos
            length = np.linalg.norm(direction)
            if length < 1e-9:
                continue
            new_pos = nodes[nearest].pos + direction / length * min(self.cfg.step_size, length)

            if not self._point_ok(new_pos) or not self._edge_ok(nodes[nearest].pos, new_pos):
                continue

            # neighbors within rewire radius, capped to the k-nearest so per-
            # iteration work stays bounded even as the tree densifies (this
            # cap is what keeps RRT* from degrading to quadratic time here).
            nd = pos_buf[:n_pts] - new_pos
            nd2 = np.einsum("ij,ij->i", nd, nd)
            within = np.nonzero(nd2 <= self.cfg.neighbor_radius ** 2)[0]
            if len(within) > self.cfg.max_neighbors:
                order = np.argpartition(nd2[within], self.cfg.max_neighbors)[:self.cfg.max_neighbors]
                within = within[order]
            neigh = within.tolist()

            # Choose the best kinematically-valid parent (RRT* optimality).
            best_parent, best_cost = -1, float("inf")
            for i in neigh:
                if self._edge_ok(nodes[i].pos, new_pos) and \
                   self._kinematic_ok(nodes[i].pos, nodes[i].indir, new_pos):
                    c = nodes[i].cost + self._edge_cost(nodes[i].pos, new_pos)
                    if c < best_cost:
                        best_parent, best_cost = i, c
            # Fall back to the nearest node if no neighbor was kinematically valid.
            if best_parent < 0:
                if not (self._edge_ok(nodes[nearest].pos, new_pos) and
                        self._kinematic_ok(nodes[nearest].pos, nodes[nearest].indir, new_pos)):
                    continue
                best_parent = nearest
                best_cost = nodes[nearest].cost + self._edge_cost(nodes[nearest].pos, new_pos)

            new_idx = len(nodes)
            new_indir = self._heading(nodes[best_parent].pos, new_pos)
            nodes.append(_Node(pos=new_pos, parent=best_parent, cost=best_cost, indir=new_indir))
            pos_buf[n_pts] = new_pos
            n_pts += 1

            # rewire neighbors through the new node if cheaper and kinematically valid
            for i in neigh:
                if i == best_parent:
                    continue
                if self._edge_ok(new_pos, nodes[i].pos) and \
                   self._kinematic_ok(new_pos, new_indir, nodes[i].pos):
                    c = best_cost + self._edge_cost(new_pos, nodes[i].pos)
                    if c < nodes[i].cost:
                        nodes[i].parent = new_idx
                        nodes[i].cost = c
                        nodes[i].indir = self._heading(new_pos, nodes[i].pos)

            # goal connection
            if np.linalg.norm(new_pos - goal) <= self.cfg.goal_tolerance and \
               self._edge_ok(new_pos, goal) and self._kinematic_ok(new_pos, new_indir, goal):
                gc = best_cost + self._edge_cost(new_pos, goal)
                if gc < best_goal_cost:
                    best_goal_cost = gc
                    best_goal_idx = new_idx

        if best_goal_idx < 0:
            return None

        # reconstruct path start..goal
        path = [goal]
        idx = best_goal_idx
        while idx != -1:
            path.append(nodes[idx].pos)
            idx = nodes[idx].parent
        path.reverse()
        path = np.array(path, dtype=float)
        return self._shortcut(path)

    def _shortcut(self, path: np.ndarray, rounds: int = 120, risk_tol: float = 0.5) -> np.ndarray:
        """
        Randomized risk-aware shortcutting: replace a sub-path with a straight
        segment only if it stays terrain/detection-valid AND does not increase
        integrated threat exposure beyond the sub-path it replaces (+ risk_tol).
        Without the risk guard the shortcutter would straighten risk-avoiding
        detours back through the threats (the same failure mode as naive LOS
        smoothing).
        """
        pts = [p for p in path]
        for _ in range(rounds):
            if len(pts) <= 2:
                break
            i = int(self._rng.integers(0, len(pts) - 1))
            j = int(self._rng.integers(i + 1, len(pts)))
            if j - i < 2:
                continue
            if not self._edge_ok(pts[i], pts[j]):
                continue
            if not self._kinematic_ok(pts[i], None, pts[j]):  # climb-gradient guard
                continue
            # Turn-angle guard at the two new junctions the shortcut would create.
            in_i = self._heading(pts[i - 1], pts[i]) if i > 0 else None
            if not self._kinematic_ok(pts[i], in_i, pts[j]):
                continue
            if j < len(pts) - 1:
                cut_dir = self._heading(pts[i], pts[j])
                if not self._kinematic_ok(pts[j], cut_dir, pts[j + 1]):
                    continue
            # Risk of the straight shortcut vs the spanned sub-path.
            shortcut_risk = self._edge_risk(pts[i], pts[j])
            span_risk = sum(self._edge_risk(pts[k], pts[k + 1]) for k in range(i, j))
            if shortcut_risk <= span_risk + risk_tol:
                pts = pts[:i + 1] + pts[j:]
        return np.array(pts, dtype=float)


def refine_with_rrt_star(
    guide_path: np.ndarray,
    terrain: np.ndarray,
    hazards: List[HazardSite],
    config: Optional[RRTStarConfig] = None,
    risk_sampler=None,
) -> Optional[np.ndarray]:
    """
    Convenience wrapper: refine an existing global path with risk-aware RRT*.

    Pass `risk_sampler=grid.risk_at` (a PlanningGrid3D bound method) for a fast
    cached risk lookup; otherwise a slow live LOS computation is used.
    """
    guide = np.asarray(guide_path, dtype=float)
    if len(guide) < 2:
        return guide
    planner = RiskAwareRRTStar(terrain, hazards, guide, config, risk_sampler=risk_sampler)
    return planner.refine(guide[0], guide[-1])
