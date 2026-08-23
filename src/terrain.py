"""
AerX Labs — Digital Elevation Model (DEM) & Scenario Generator
src/terrain.py

Provides procedural generation for multiple distinct tactical DEM environments:
1. Mountain Pass & Valley Corridor (Standard benchmark)
2. Deep Canyon & Gorge Network (Heavy terrain-masking trench)
3. Coastal Archipelago & Island Ridge (Maritime low-altitude navigation)
4. Rolling Hills & Multi-Threat Sensor Net (Overlapping radar coverage)
"""

import os
from typing import Dict, Tuple
import numpy as np


def create_demo_terrain(nx: int = 100, ny: int = 100) -> np.ndarray:
    """A twin-peak synthetic terrain used only as a unit-test fixture (its ridge
    gives the LOS-masking tests a peak to hide behind). Not a mission scenario —
    all real missions run on Copernicus DEMs via load_mission_scenario()."""
    return create_mountain_pass_terrain(nx, ny)


def create_mountain_pass_terrain(nx: int = 100, ny: int = 100) -> np.ndarray:
    """Twin mountain peaks and a central shielding ridge (test-fixture terrain)."""
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    X, Y = np.meshgrid(x, y)

    # Base elevation
    terrain = np.full((ny, nx), 100.0)

    # Mountain Peak 1 (West)
    terrain += 180.0 * np.exp(-((X + 0.35) ** 2 + (Y + 0.15) ** 2) / 0.08)

    # Mountain Peak 2 (East)
    terrain += 140.0 * np.exp(-((X - 0.30) ** 2 + (Y - 0.30) ** 2) / 0.06)

    # Central Shielding Ridge
    terrain += 90.0 * np.exp(-((X + 0.05) ** 2 + (Y - 0.45) ** 2) / 0.04)

    # Subtle terrain roughness
    terrain += 5.0 * np.sin(4 * np.pi * X) * np.cos(4 * np.pi * Y)

    return terrain


def create_fractal_terrain(
    nx: int = 100,
    ny: int = 100,
    base: float = 80.0,
    relief: float = 200.0,
    octaves: int = 6,
    persistence: float = 0.55,
    lacunarity: float = 2.0,
    seed: int = 12,
) -> np.ndarray:
    """
    Generate natural-looking terrain via fractal Brownian motion (value-noise
    summed over octaves).

    Real terrain is statistically self-similar (fractal), which the sum-of-
    Gaussians scenarios above do not capture — they read as smooth artificial
    bumps. fBm produces ridgelines, valleys, and roughness across scales that far
    better approximate a real DEM, while staying pure-NumPy (no data download or
    extra dependency). This is the recommended synthetic terrain when a more
    DEM-like surface is wanted; for genuine terrain use `load_srtm_dem`.

    Parameters:
        base:        minimum elevation (m).
        relief:      peak-to-trough elevation span (m).
        octaves:     number of noise layers (detail levels).
        persistence: amplitude falloff per octave (0-1).
        lacunarity:  frequency growth per octave.
        seed:        RNG seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    xs = np.linspace(0, 1, nx)
    ys = np.linspace(0, 1, ny)
    gx, gy = np.meshgrid(xs, ys)

    field = np.zeros((ny, nx), dtype=float)
    amp = 1.0
    freq = 3.0
    amp_sum = 0.0
    for _ in range(octaves):
        # Low-resolution random lattice, then smooth bilinear upsample = value noise.
        res = max(2, int(freq) + 1)
        lattice = rng.random((res, res))
        li = np.clip(gy * (res - 1), 0, res - 1)
        lj = np.clip(gx * (res - 1), 0, res - 1)
        i0 = np.floor(li).astype(int); j0 = np.floor(lj).astype(int)
        i1 = np.minimum(i0 + 1, res - 1); j1 = np.minimum(j0 + 1, res - 1)
        ti = li - i0; tj = lj - j0
        # smoothstep for smoother interpolation
        ti = ti * ti * (3 - 2 * ti); tj = tj * tj * (3 - 2 * tj)
        top = lattice[i0, j0] * (1 - tj) + lattice[i0, j1] * tj
        bot = lattice[i1, j0] * (1 - tj) + lattice[i1, j1] * tj
        octave = top * (1 - ti) + bot * ti
        field += amp * octave
        amp_sum += amp
        amp *= persistence
        freq *= lacunarity

    field /= amp_sum
    field = (field - field.min()) / (field.max() - field.min() + 1e-9)
    return base + relief * field


def load_srtm_dem(path: str, downsample: int = 1) -> np.ndarray:
    """
    Load a real SRTM/GeoTIFF DEM into the same 2D elevation array format used by
    the simulator. Requires the optional `rasterio` dependency.

    The rest of the pipeline (LOS, risk, planning, scoring) is agnostic to how
    the elevation array is produced, so a real DEM can be dropped in wherever a
    synthetic generator is currently called — no other code changes needed.

    Parameters:
        path:       path to a GeoTIFF / SRTM `.tif` elevation raster.
        downsample: integer stride to coarsen very large rasters.
    """
    try:
        import rasterio
    except ImportError as exc:
        raise ImportError(
            "load_srtm_dem requires 'rasterio'. Install it with `pip install rasterio`, "
            "or use create_fractal_terrain for a synthetic DEM-like surface."
        ) from exc

    with rasterio.open(path) as src:
        band = src.read(1).astype(float)
    if downsample > 1:
        band = band[::downsample, ::downsample]
    # Replace nodata sentinels with the local minimum so LOS/AGL stay well-defined.
    finite = np.isfinite(band)
    if not finite.all():
        band[~finite] = band[finite].min() if finite.any() else 0.0
    return band


def write_dem_geotiff(
    elevation: np.ndarray,
    path: str,
    pixel_size_m: float = 30.0,
    origin_lon: float = 77.0,
    origin_lat: float = 30.0,
) -> str:
    """
    Write an elevation array to a georeferenced GeoTIFF (EPSG:4326) with a
    ~`pixel_size_m` grid spacing. Requires `rasterio`.

    This lets the pipeline produce and round-trip a real geospatial DEM file
    through exactly the same rasterio ingestion path a downloaded SRTM tile
    would use, so the DEM-ingestion capability is exercised end-to-end even
    without a network fetch. Replace the file with a genuine SRTM/ASTER tile of
    the same format and the rest of the code is unchanged.
    """
    import rasterio
    from rasterio.transform import from_origin

    os_dir = os.path.dirname(path)
    if os_dir:
        os.makedirs(os_dir, exist_ok=True)

    # Approximate degrees-per-pixel for the requested ground spacing.
    deg = pixel_size_m / 111_320.0
    transform = from_origin(origin_lon, origin_lat, deg, deg)
    ny, nx = elevation.shape
    with rasterio.open(
        path, "w", driver="GTiff", height=ny, width=nx, count=1,
        dtype="float32", crs="EPSG:4326", transform=transform, nodata=-32768.0,
    ) as dst:
        dst.write(elevation.astype("float32"), 1)
    return path


def copernicus_tile_name(lat: float, lon: float) -> str:
    """Copernicus GLO-30 tile id for the 1-degree cell containing (lat, lon)."""
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    la = abs(int(np.floor(lat)))
    lo = abs(int(np.floor(lon)))
    return f"Copernicus_DSM_COG_10_{ns}{la:02d}_00_{ew}{lo:03d}_00_DEM"


def download_copernicus_tile(lat: float, lon: float, cache_dir: str = "data/terrain") -> str:
    """
    Return the path to the real Copernicus GLO-30 DEM tile for (lat, lon),
    downloading it from the public AWS open-data bucket on first use.

    Raises a clear error if the tile does not exist (e.g. an ocean cell).
    """
    import urllib.request
    import urllib.error

    name = copernicus_tile_name(lat, lon)
    path = os.path.join(cache_dir, name + ".tif")
    if os.path.isfile(path):
        return path
    os.makedirs(cache_dir, exist_ok=True)
    url = f"https://copernicus-dem-30m.s3.amazonaws.com/{name}/{name}.tif"
    print(f"  Downloading real DEM tile: {url}", flush=True)
    try:
        urllib.request.urlretrieve(url, path)
    except urllib.error.HTTPError as exc:
        if os.path.isfile(path):
            os.remove(path)
        raise FileNotFoundError(
            f"No Copernicus DEM tile at lat={lat}, lon={lon} ({name}). "
            "That 1-degree cell may be ocean/void — try nearby land coordinates."
        ) from exc
    return path


def find_relief_window(path: str, win_px: int = 600,
                       relief_lo: float = 700.0, relief_hi: float = 2800.0) -> Tuple[int, int]:
    """
    Scan a DEM tile for a `win_px`-square window with navigable-but-real relief
    (roughly `relief_lo`..`relief_hi` metres) and return its (row0, col0) origin.
    Falls back to the tile centre if nothing fits the band.
    """
    import rasterio
    from rasterio.enums import Resampling

    OV = 300  # coarse overview resolution for the scan
    with rasterio.open(path) as src:
        H, W = src.height, src.width
        ov = src.read(1, out_shape=(OV, OV), resampling=Resampling.average).astype(float)

    wcells = max(4, int(round(win_px / (H / OV))))  # window size in overview cells
    best = None
    for i0 in range(0, OV - wcells, max(1, wcells // 3)):
        for j0 in range(0, OV - wcells, max(1, wcells // 3)):
            w = ov[i0:i0 + wcells, j0:j0 + wcells]
            w = w[np.isfinite(w)]
            if w.size == 0:
                continue
            rel = float(w.max() - w.min())
            if relief_lo < rel < relief_hi and float(w.min()) < 4500.0:
                if best is None or rel > best[0]:
                    best = (rel, i0, j0)
    if best is None:
        return (max(0, (H - win_px) // 2), max(0, (W - win_px) // 2))
    row0 = int(best[1] * (H / OV))
    col0 = int(best[2] * (W / OV))
    row0 = min(row0, H - win_px)
    col0 = min(col0, W - win_px)
    return (row0, col0)


def load_real_dem_region(
    path: str,
    row0: int,
    col0: int,
    win_px: int = 600,
    out_cells: int = 140,
) -> np.ndarray:
    """
    Read a square window from a real GeoTIFF DEM and block-average it down to an
    `out_cells` x `out_cells` planning grid. Requires `rasterio`.

    Copernicus GLO-30 / SRTM tiles are 1 degree at ~30 m (3600x3600). A ~600 px
    window is ~18 km on a side; averaging to ~140 cells keeps real ridgelines and
    valleys while smoothing 30 m noise into a tractable planning surface.
    """
    import rasterio
    from rasterio.windows import Window
    from rasterio.enums import Resampling

    with rasterio.open(path) as src:
        arr = src.read(
            1, window=Window(col0, row0, win_px, win_px),
            out_shape=(out_cells, out_cells), resampling=Resampling.average,
        ).astype(float)
    finite = np.isfinite(arr)
    if not finite.all():
        arr[~finite] = arr[finite].min() if finite.any() else 0.0
    return arr


def ensure_sample_dem(path: str, nx: int = 120, ny: int = 120, seed: int = 21) -> np.ndarray:
    """
    Return the elevation array for a georeferenced sample DEM GeoTIFF, creating
    it (fractal terrain written via `write_dem_geotiff`) on first use.
    """
    if not os.path.isfile(path):
        dem = create_fractal_terrain(nx=nx, ny=ny, base=90.0, relief=240.0, seed=seed)
        write_dem_geotiff(dem, path)
    return load_srtm_dem(path)


def get_available_scenarios() -> Dict[str, str]:
    """Return dictionary of real-DEM scenario IDs and descriptive names."""
    return {
        "dem": "Mount Everest / Khumbu (real Copernicus DEM)",
        "ghats": "Western Ghats / Anamalai (real Copernicus DEM)",
        "custom": "Any lat/lon region (real Copernicus DEM)",
    }