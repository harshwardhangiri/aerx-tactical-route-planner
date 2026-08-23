"""
AerX Labs — Line of Sight (LOS) & Terrain Interception Engine
src/los.py

Vectorized bilinear terrain interpolation and ray-casting occlusion analysis.
"""

from typing import Optional, Tuple
import numpy as np


def terrain_height(terrain: np.ndarray, x: float, y: float) -> float:
    """Fast scalar bilinear interpolation of terrain elevation."""
    ny, nx = terrain.shape

    x = float(np.clip(x, 0.0, nx - 1.0))
    y = float(np.clip(y, 0.0, ny - 1.0))

    x0 = int(x)
    x1 = min(x0 + 1, nx - 1)

    y0 = int(y)
    y1 = min(y0 + 1, ny - 1)

    dx = x - x0
    dy = y - y0

    z00 = terrain[y0, x0]
    z10 = terrain[y0, x1]
    z01 = terrain[y1, x0]
    z11 = terrain[y1, x1]

    return float(
        (1.0 - dx) * (1.0 - dy) * z00
        + dx * (1.0 - dy) * z10
        + (1.0 - dx) * dy * z01
        + dx * dy * z11
    )


def check_los(
    terrain: np.ndarray,
    start: Tuple[float, float, float],
    end: Tuple[float, float, float],
    samples: int = 50
) -> Tuple[bool, Optional[Tuple[float, float, float]]]:
    """
    Vectorized ray-casting terrain LOS check.
    
    Returns:
        (visible, blocking_point)
    """
    start_arr = np.asarray(start, dtype=float)
    end_arr = np.asarray(end, dtype=float)
    ny, nx = terrain.shape

    # Vectorized sample points along ray
    t = np.linspace(0.0, 1.0, samples)
    pts = start_arr + t[:, None] * (end_arr - start_arr)

    xs = np.clip(pts[:, 0], 0.0, nx - 1.0)
    ys = np.clip(pts[:, 1], 0.0, ny - 1.0)
    alts = pts[:, 2]

    x0 = xs.astype(int)
    x1 = np.minimum(x0 + 1, nx - 1)
    y0 = ys.astype(int)
    y1 = np.minimum(y0 + 1, ny - 1)

    dx = xs - x0
    dy = ys - y0

    # Vectorized bilinear interpolation
    grounds = (
        (1.0 - dx) * (1.0 - dy) * terrain[y0, x0]
        + dx * (1.0 - dy) * terrain[y0, x1]
        + (1.0 - dx) * dy * terrain[y1, x0]
        + dx * dy * terrain[y1, x1]
    )

    occluded_indices = np.where(grounds >= alts)[0]
    if len(occluded_indices) > 0:
        first_idx = occluded_indices[0]
        return False, (float(pts[first_idx, 0]), float(pts[first_idx, 1]), float(grounds[first_idx]))

    return True, None