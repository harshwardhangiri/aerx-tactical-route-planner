"""
AerX Labs — D* Lite 3D Incremental Path Planner
src/dstar_lite.py

Implements the D* Lite (Koenig & Likhachev) incremental graph search algorithm
extended for 3D layered geospatial grids. Computes optimal, risk-sensitive paths
and supports real-time incremental replanning when risk fields or obstacles change.
"""

from heapq import heappop, heappush
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.planning_grid import Node3D, PlanningGrid3D


class PriorityQueue:
    """A min-priority queue with O(1) removal and updates using a dict of keys."""

    def __init__(self):
        self._heap = []
        self._entries = {}  # node -> (k1, k2)
        self._counter = 0

    def insert_or_update(self, node: Node3D, key: Tuple[float, float]) -> None:
        self._entries[node] = key
        self._counter += 1
        heappush(self._heap, (key[0], key[1], self._counter, node))

    def remove(self, node: Node3D) -> None:
        if node in self._entries:
            del self._entries[node]

    def pop(self) -> Tuple[Node3D, Tuple[float, float]]:
        while self._heap:
            k1, k2, _, node = heappop(self._heap)
            if node in self._entries and self._entries[node] == (k1, k2):
                del self._entries[node]
                return node, (k1, k2)
        return None, (float("inf"), float("inf"))

    def top_key(self) -> Tuple[float, float]:
        while self._heap:
            k1, k2, _, node = self._heap[0]
            if node in self._entries and self._entries[node] == (k1, k2):
                return (k1, k2)
            heappop(self._heap)
        return (float("inf"), float("inf"))

    def contains(self, node: Node3D) -> bool:
        return node in self._entries

    def is_empty(self) -> bool:
        while self._heap:
            k1, k2, _, node = self._heap[0]
            if node in self._entries and self._entries[node] == (k1, k2):
                return False
            heappop(self._heap)
        return True


class DStarLite3D:
    """
    3D D* Lite Global Path Planner for terrain and hazard-aware navigation.
    """

    def __init__(self, grid: PlanningGrid3D):
        self.grid = grid
        self.s_start: Optional[Node3D] = None
        self.s_goal: Optional[Node3D] = None
        self.s_last: Optional[Node3D] = None

        self.g: Dict[Node3D, float] = {}
        self.rhs: Dict[Node3D, float] = {}
        self.open_set = PriorityQueue()
        self.km: float = 0.0
        self.last_expansions: int = 0  # node expansions of the most recent search

    def _get_g(self, u: Node3D) -> float:
        return self.g.get(u, float("inf"))

    def _get_rhs(self, u: Node3D) -> float:
        return self.rhs.get(u, float("inf"))

    def _heuristic(self, a: Node3D, b: Node3D) -> float:
        """Fast admissible Euclidean lower-bound heuristic."""
        dx = (a[0] - b[0]) * self.grid.xy_scale
        dy = (a[1] - b[1]) * self.grid.xy_scale
        dz = self.grid.z_layers[a[2]] - self.grid.z_layers[b[2]]
        return float(np.sqrt(dx * dx + dy * dy + dz * dz) * self.grid.weights.w_distance)

    def _calculate_key(self, u: Node3D) -> Tuple[float, float]:
        min_g_rhs = min(self._get_g(u), self._get_rhs(u))
        k1 = min_g_rhs + self._heuristic(self.s_start, u) + self.km
        k2 = min_g_rhs
        return (k1, k2)

    def _update_vertex(self, u: Node3D) -> None:
        if u != self.s_goal:
            min_rhs = float("inf")
            for succ, cost in self.grid.get_neighbors(u):
                val = cost + self._get_g(succ)
                if val < min_rhs:
                    min_rhs = val
            self.rhs[u] = min_rhs

        if self.open_set.contains(u):
            self.open_set.remove(u)

        if self._get_g(u) != self._get_rhs(u):
            self.open_set.insert_or_update(u, self._calculate_key(u))

    def _compute_shortest_path(self, max_expansions: int = 400000) -> bool:
        """Main expansion loop of D* Lite."""
        expansions = 0
        while (
            self.open_set.top_key() < self._calculate_key(self.s_start)
            or self._get_rhs(self.s_start) != self._get_g(self.s_start)
        ):
            if self.open_set.is_empty() or expansions >= max_expansions:
                return False

            u, k_old = self.open_set.pop()
            k_new = self._calculate_key(u)

            if k_old < k_new:
                self.open_set.insert_or_update(u, k_new)
            elif self._get_g(u) > self._get_rhs(u):
                self.g[u] = self._get_rhs(u)
                for pred, _ in self.grid.get_neighbors(u):
                    self._update_vertex(pred)
            else:
                self.g[u] = float("inf")
                self._update_vertex(u)
                for pred, _ in self.grid.get_neighbors(u):
                    self._update_vertex(pred)

            expansions += 1

        self.last_expansions = expansions
        return self._get_g(self.s_start) < float("inf")

    def plan(
        self,
        start_pos: Tuple[float, float, float],
        goal_pos: Tuple[float, float, float]
    ) -> Optional[List[Node3D]]:
        """
        Execute initial path plan between start and goal.

        Returns:
            List of discrete Node3D waypoints from start to goal, or None if unreachable.
        """
        self.s_start = self.grid.pos_to_node(start_pos)
        self.s_goal = self.grid.pos_to_node(goal_pos)

        if not self.grid.is_feasible(self.s_start):
            raise ValueError(f"Start position {start_pos} is not feasible (ground clearance or limits violated).")
        if not self.grid.is_feasible(self.s_goal):
            raise ValueError(f"Goal position {goal_pos} is not feasible (ground clearance or limits violated).")

        self.g.clear()
        self.rhs.clear()
        self.open_set = PriorityQueue()
        self.km = 0.0
        self.s_last = self.s_start

        self.rhs[self.s_goal] = 0.0
        self.open_set.insert_or_update(self.s_goal, self._calculate_key(self.s_goal))

        success = self._compute_shortest_path()
        if not success:
            return None

        return self.extract_path()

    def extract_path(self, max_steps: int = 500) -> Optional[List[Node3D]]:
        """Extract the current best path from s_start to s_goal."""
        if self._get_g(self.s_start) == float("inf"):
            return None

        path = [self.s_start]
        curr = self.s_start
        steps = 0

        while curr != self.s_goal and steps < max_steps:
            best_succ = None
            min_cost = float("inf")

            for succ, cost in self.grid.get_neighbors(curr):
                val = cost + self._get_g(succ)
                if val < min_cost:
                    min_cost = val
                    best_succ = succ

            if best_succ is None or best_succ in path:
                # Loop or trapped
                break

            curr = best_succ
            path.append(curr)
            steps += 1

        if path[-1] != self.s_goal:
            return None

        return path

    def replan_after_hazard_change(
        self,
        current_pos: Tuple[float, float, float],
        affected_nodes: Optional[List[Node3D]] = None
    ) -> Optional[List[Node3D]]:
        """
        Perform an efficient D* Lite incremental replan when map/hazard costs change.
        """
        s_current = self.grid.pos_to_node(current_pos)
        self.km += self._heuristic(self.s_last, s_current)
        self.s_last = s_current
        self.s_start = s_current

        if affected_nodes is None:
            # Check all nodes in open set or neighbors of feasible nodes
            nodes_to_update = list(self.rhs.keys())
        else:
            nodes_to_update = affected_nodes

        for u in nodes_to_update:
            self._update_vertex(u)

        success = self._compute_shortest_path()
        if not success:
            return None

        return self.extract_path()

    def get_path_coordinates(self, path: List[Node3D]) -> np.ndarray:
        """Convert a discrete node path into an (N, 3) continuous coordinate array."""
        return np.array([self.grid.node_to_pos(node) for node in path], dtype=float)
