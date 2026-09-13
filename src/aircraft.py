"""
AerX Labs — Aircraft Profiles & Real-World Calibration
src/aircraft.py

The planner runs in a normalized grid (terrain normalized to [90, 350] planning
units, one cell per node). This module attaches a *real-world calibration layer*
on top of it:

  * RegionScale  — how many real metres a grid cell and a normalized altitude unit
    are worth for the currently-loaded DEM crop (Copernicus GLO-30, ~150 m/cell),
    captured when the terrain is loaded.
  * Aircraft     — a real platform's flight parameters (speed, climb rate, turn/bank
    limit, fuel/endurance, radar cross-section, clearance floor, ceiling).
  * realize_sortie() — turns a planned route (in grid units) into REAL numbers:
    distance in km, time in minutes, fuel in kg, and feasibility against the
    aircraft's endurance and service ceiling.

So the planning stays in the tuned grid, but every reported quantity — and the
kinematic limits handed to the RRT* refiner — is a genuine physical value for the
selected aircraft over the selected terrain.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import math

import numpy as np

from src.los import terrain_height

# Standard gravity, for the coordinated-turn radius r = v^2 / (g * tan(bank)).
_G = 9.80665


@dataclass
class RegionScale:
    """Real-world scale of the loaded DEM crop (set when the terrain is built)."""
    meters_per_cell: float = 150.0   # horizontal: Copernicus 30 m/px * 600 px / 120 cells
    relief_m: float = 1000.0         # real elevation range (raw.max - raw.min) of the crop
    elev_min_m: float = 0.0          # real elevation of the crop's lowest point (m ASL)
    znorm_base: float = 90.0         # normalized value that maps to elev_min_m
    znorm_span: float = 260.0        # normalized span the relief was mapped onto

    @property
    def meters_per_zunit(self) -> float:
        return self.relief_m / self.znorm_span if self.znorm_span else 1.0

    def real_alt_m(self, z_norm: float) -> float:
        """Convert a normalized planning altitude to real metres ASL."""
        return self.elev_min_m + (z_norm - self.znorm_base) * self.meters_per_zunit


@dataclass
class Aircraft:
    """A real aircraft/UAV profile in physical units."""
    key: str
    name: str
    kind: str                 # "helicopter" | "drone" | "fixed-wing"
    cruise_speed: float       # m/s
    max_speed: float          # m/s
    climb_rate: float         # m/s (max vertical rate)
    max_bank_deg: float       # bank angle limit -> turn radius
    min_agl: float            # m, terrain-clearance floor the platform flies
    service_ceiling: float    # m ASL
    fuel_capacity_kg: float
    cruise_burn_kgps: float   # kg/s at cruise
    climb_burn_factor: float  # burn multiplier while climbing
    rcs_m2: float             # radar cross-section (detectability scales with this)
    note: str = ""

    def turn_radius(self, speed: Optional[float] = None) -> float:
        """Minimum coordinated-turn radius (m) at the given speed (default cruise)."""
        v = speed if speed is not None else self.cruise_speed
        return v * v / (_G * math.tan(math.radians(self.max_bank_deg)))

    def endurance_s(self) -> float:
        return self.fuel_capacity_kg / max(self.cruise_burn_kgps, 1e-9)

    def range_m(self) -> float:
        return self.cruise_speed * self.endurance_s()


# --------------------------------------------------------------------------- #
#  Registry of real platforms (public, unclassified reference figures; these are
#  representative planning values, not authoritative performance data).
# --------------------------------------------------------------------------- #
AIRCRAFT: Dict[str, Aircraft] = {
    "scout_heli": Aircraft(
        key="scout_heli", name="Scout / Attack Helicopter", kind="helicopter",
        cruise_speed=72.0, max_speed=82.0, climb_rate=12.7, max_bank_deg=45.0,
        min_agl=30.0, service_ceiling=6400.0,
        fuel_capacity_kg=1150.0, cruise_burn_kgps=0.13, climb_burn_factor=1.8,
        rcs_m2=3.0, note="Agile, nap-of-the-earth; the default profile."),
    "chinook": Aircraft(
        key="chinook", name="CH-47 Chinook", kind="helicopter",
        cruise_speed=80.0, max_speed=88.0, climb_rate=10.0, max_bank_deg=30.0,
        min_agl=45.0, service_ceiling=5600.0,
        fuel_capacity_kg=4000.0, cruise_burn_kgps=0.42, climb_burn_factor=1.7,
        rcs_m2=14.0, note="Heavy-lift tandem rotor; fast, big fuel, wide turns."),
    "quad_uav": Aircraft(
        key="quad_uav", name="Small Quadcopter UAV", kind="drone",
        cruise_speed=15.0, max_speed=22.0, climb_rate=5.0, max_bank_deg=35.0,
        min_agl=15.0, service_ceiling=1500.0,
        fuel_capacity_kg=2.0, cruise_burn_kgps=0.0011, climb_burn_factor=2.2,
        rcs_m2=0.02, note="Low-speed, tight turns, short endurance, hugs terrain."),
    "reaper": Aircraft(
        key="reaper", name="MQ-9 Reaper (fixed-wing UAV)", kind="fixed-wing",
        cruise_speed=90.0, max_speed=135.0, climb_rate=5.0, max_bank_deg=25.0,
        min_agl=120.0, service_ceiling=15000.0,
        fuel_capacity_kg=1800.0, cruise_burn_kgps=0.052, climb_burn_factor=1.5,
        rcs_m2=1.5, note="High-endurance, high-altitude, gentle turns."),
}

DEFAULT_AIRCRAFT = "scout_heli"


def get_aircraft(key: Optional[str]) -> Aircraft:
    return AIRCRAFT.get((key or DEFAULT_AIRCRAFT).lower(), AIRCRAFT[DEFAULT_AIRCRAFT])


def list_aircraft() -> List[str]:
    return list(AIRCRAFT.keys())


@dataclass
class RealStats:
    """A route/sortie expressed in real physical units for a chosen aircraft."""
    distance_km: float
    time_min: float
    fuel_kg: float
    fuel_pct: float           # % of the aircraft's fuel capacity
    within_endurance: bool
    min_agl_m: float
    max_alt_m: float
    within_ceiling: bool
    turn_radius_m: float          # aircraft's min turn radius at cruise (the limit)
    min_route_radius_m: float     # tightest turn the route actually demands (m)
    max_bank_deg: float           # bank the tightest turn needs at cruise
    max_climb_grad: float         # steepest rise/run the route demands
    within_bank: bool             # tightest turn is within the aircraft's bank limit
    within_climb: bool            # steepest climb is within the aircraft's climb rate
    flyable: bool                 # kinematically flyable (bank + climb both OK)


def realize_sortie(legs: List[np.ndarray], terrain: np.ndarray,
                   scale: RegionScale, aircraft: Aircraft) -> RealStats:
    """
    Convert one or more planned legs (each an (N,3) array of grid-unit waypoints)
    into real-world flight numbers for `aircraft` over the DEM described by `scale`.
    Horizontal distance uses metres/cell; vertical uses metres per normalized unit;
    time and fuel use the aircraft's real speed, climb rate and burn.
    """
    mpc = scale.meters_per_cell
    mpz = scale.meters_per_zunit

    total_dist_m = 0.0
    total_time_s = 0.0
    total_fuel_kg = 0.0
    min_agl_m = float("inf")
    max_alt_m = -float("inf")
    min_route_radius_m = float("inf")   # tightest turn the route demands
    max_climb_grad = 0.0                # steepest rise/run

    for wp in legs:
        if wp is None:
            continue
        wp = np.asarray(wp, dtype=float)
        for i in range(len(wp)):
            agl_m = (wp[i, 2] - terrain_height(terrain, wp[i, 0], wp[i, 1])) * mpz
            min_agl_m = min(min_agl_m, agl_m)
            max_alt_m = max(max_alt_m, scale.real_alt_m(wp[i, 2]))
        for i in range(len(wp) - 1):
            a, b = wp[i], wp[i + 1]
            horiz_m = math.hypot((b[0] - a[0]) * mpc, (b[1] - a[1]) * mpc)
            vert_m = (b[2] - a[2]) * mpz
            seg_m = math.hypot(horiz_m, vert_m)
            total_dist_m += seg_m
            if horiz_m > 1e-6:
                max_climb_grad = max(max_climb_grad, abs(vert_m) / horiz_m)

            t_cruise = seg_m / max(aircraft.cruise_speed, 1e-6)
            if vert_m > 0:
                t_climb = vert_m / max(aircraft.climb_rate, 1e-6)
                t_seg = max(t_cruise, t_climb)
                burn = aircraft.cruise_burn_kgps * aircraft.climb_burn_factor
            else:
                t_seg = t_cruise
                burn = aircraft.cruise_burn_kgps
            total_time_s += t_seg
            total_fuel_kg += burn * t_seg
        # Tightest horizontal turn on this leg (Menger curvature of each triple, in
        # metres). radius = 1/curvature; the smaller the radius, the tighter the turn.
        for i in range(1, len(wp) - 1):
            p0 = np.array([wp[i - 1, 0] * mpc, wp[i - 1, 1] * mpc])
            p1 = np.array([wp[i, 0] * mpc, wp[i, 1] * mpc])
            p2 = np.array([wp[i + 1, 0] * mpc, wp[i + 1, 1] * mpc])
            a1, b1, c1 = p1 - p0, p2 - p1, p2 - p0
            la, lb, lc = np.hypot(*a1), np.hypot(*b1), np.hypot(*c1)
            area2 = abs(a1[0] * c1[1] - a1[1] * c1[0])   # 2 * triangle area
            if la * lb * lc < 1e-6 or area2 < 1e-9:
                continue                                  # straight -> infinite radius
            radius = (la * lb * lc) / (2.0 * area2)
            min_route_radius_m = min(min_route_radius_m, radius)

    if not np.isfinite(min_agl_m):
        min_agl_m = 0.0
    if not np.isfinite(max_alt_m):
        max_alt_m = 0.0

    # Bank the tightest turn demands at cruise: tan(bank) = v^2 / (g * R).
    if np.isfinite(min_route_radius_m):
        max_bank_deg = math.degrees(math.atan(aircraft.cruise_speed ** 2 /
                                              (_G * min_route_radius_m)))
    else:
        min_route_radius_m, max_bank_deg = float("inf"), 0.0
    climb_limit_grad = aircraft.climb_rate / max(aircraft.cruise_speed, 1e-6)
    within_bank = max_bank_deg <= aircraft.max_bank_deg + 3.0     # 3 deg tolerance
    within_climb = max_climb_grad <= climb_limit_grad + 0.05      # informational

    return RealStats(
        distance_km=total_dist_m / 1000.0,
        time_min=total_time_s / 60.0,
        fuel_kg=total_fuel_kg,
        fuel_pct=100.0 * total_fuel_kg / max(aircraft.fuel_capacity_kg, 1e-9),
        within_endurance=total_fuel_kg <= aircraft.fuel_capacity_kg,
        min_agl_m=min_agl_m,
        max_alt_m=max_alt_m,
        within_ceiling=max_alt_m <= aircraft.service_ceiling,
        turn_radius_m=aircraft.turn_radius(),
        min_route_radius_m=min_route_radius_m,
        max_bank_deg=max_bank_deg,
        max_climb_grad=max_climb_grad,
        within_bank=within_bank,
        within_climb=within_climb,
        # Turn radius / bank is the HARD coordinated-turn limit; a steep climb is
        # soft (a helicopter slows to climb, which the time/fuel model captures).
        flyable=bool(within_bank),
    )


def kinematic_limits_for_grid(aircraft: Aircraft, scale: RegionScale,
                              step_cells: float):
    """
    Translate the aircraft's real turn radius and climb rate into the normalized
    limits the RRT* refiner uses (max heading change per edge, max climb gradient).

    * A turn of radius R over an edge of arc length L subtends L/R radians, so the
      per-edge heading budget is degrees(L_m / R_m), clamped to a sane range.
    * The climb-gradient cap is (max climb rate / cruise speed) — the steepest
      slope the platform can sustain — expressed as normalized dz over cells.
    """
    step_m = step_cells * scale.meters_per_cell
    r = max(aircraft.turn_radius(), 1e-6)
    max_turn_deg = float(np.clip(math.degrees(step_m / r), 8.0, 75.0))

    # Real climb gradient (dimensionless: rise/run); convert to normalized units
    # (dz in znorm per dx in cells) so the refiner can apply it on the grid.
    real_gradient = aircraft.climb_rate / max(aircraft.cruise_speed, 1e-6)
    grad_norm = real_gradient * scale.meters_per_cell / max(scale.meters_per_zunit, 1e-6)
    max_climb_rate = float(np.clip(grad_norm, 0.2, 2.5))
    return max_turn_deg, max_climb_rate
