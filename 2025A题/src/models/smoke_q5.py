"""2025 CUMCM A, question 5: simultaneous concealment from three missiles.

All physical smoke clouds are tested against every missile.  A time instant is
effective only when the complete target is hidden from M1, M2 and M3 at the
same time.  The reported primary missile of a bomb is a post-processing label;
it never limits which missile can be blocked by that cloud.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import differential_evolution, minimize


PROBLEM_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROBLEM_ROOT / "src" / "models"))

from smoke_q1_q2 import (  # noqa: E402
    Constants,
    interval_duration,
    load_config as load_q1_q2_config,
    unit_heading,
)
from smoke_q3 import target_surface_points  # noqa: E402


DEFAULT_CONFIG = PROBLEM_ROOT / "configs" / "q5.json"
DEFAULT_OUTPUT_DIR = PROBLEM_ROOT / "outputs" / "model" / "q5_paper"
UAV_IDS = ("FY1", "FY2", "FY3", "FY4", "FY5")
MISSILE_IDS = ("M1", "M2", "M3")
VARIABLES_PER_UAV = 8


@dataclass(frozen=True)
class BombPlan:
    uav_id: str
    smoke_id: int
    target_id: str | None
    heading_deg: float
    speed_mps: float
    drop_time_s: float
    fuse_delay_s: float

    @property
    def active(self) -> bool:
        return self.target_id is not None

    @property
    def burst_time_s(self) -> float:
        return self.drop_time_s + self.fuse_delay_s


@dataclass(frozen=True)
class Q5Strategy:
    bombs: tuple[BombPlan, ...]

    @property
    def active_bombs(self) -> tuple[BombPlan, ...]:
        return tuple(bomb for bomb in self.bombs if bomb.active)


def load_config(path: Path) -> tuple[Constants, dict, dict, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    constants, _ = load_q1_q2_config(PROBLEM_ROOT / "configs" / "q1_q2.json")
    uavs = {
        key: np.asarray(data["uavs"][key], dtype=float) for key in UAV_IDS
    }
    missiles = {
        key: np.asarray(data["missiles"][key], dtype=float)
        for key in MISSILE_IDS
    }
    return constants, data, uavs, missiles


def decode_vector(x: np.ndarray) -> Q5Strategy:
    bombs = []
    for uav_index, uav_id in enumerate(UAV_IDS):
        base = uav_index * VARIABLES_PER_UAV
        heading = float(x[base] % 360.0)
        speed = float(x[base + 1])
        drop1 = float(x[base + 2])
        drop2 = drop1 + 1.0 + float(x[base + 3])
        drop3 = drop2 + 1.0 + float(x[base + 4])
        drops = (drop1, drop2, drop3)
        fuses = tuple(float(x[base + 5 + k]) for k in range(3))
        for slot in range(3):
            bombs.append(
                BombPlan(
                    uav_id=uav_id,
                    smoke_id=slot + 1,
                    target_id="ALL",
                    heading_deg=heading,
                    speed_mps=speed,
                    drop_time_s=drops[slot],
                    fuse_delay_s=fuses[slot],
                )
            )
    return Q5Strategy(tuple(bombs))


def encode_strategy(strategy: Q5Strategy) -> np.ndarray:
    values = []
    for uav_id in UAV_IDS:
        bombs = sorted(
            (bomb for bomb in strategy.bombs if bomb.uav_id == uav_id),
            key=lambda bomb: bomb.smoke_id,
        )
        d1, d2, d3 = (bomb.drop_time_s for bomb in bombs)
        values.extend(
            [
                bombs[0].heading_deg,
                bombs[0].speed_mps,
                d1,
                d2 - d1 - 1.0,
                d3 - d2 - 1.0,
                *(bomb.fuse_delay_s for bomb in bombs),
            ]
        )
    return np.asarray(values, dtype=float)


def decode_uav_vector(x: np.ndarray, uav_id: str) -> tuple[BombPlan, ...]:
    """Decode one UAV's eight variables into its three smoke bombs."""
    heading = float(x[0] % 360.0)
    speed = float(x[1])
    drop1 = float(x[2])
    drops = (
        drop1,
        drop1 + 1.0 + float(x[3]),
        drop1 + 2.0 + float(x[3]) + float(x[4]),
    )
    return tuple(
        BombPlan(
            uav_id=uav_id,
            smoke_id=slot + 1,
            target_id="ALL",
            heading_deg=heading,
            speed_mps=speed,
            drop_time_s=drops[slot],
            fuse_delay_s=float(x[5 + slot]),
        )
        for slot in range(3)
    )


def combine_uav_vectors(vectors: dict[str, np.ndarray]) -> np.ndarray:
    """Concatenate five eight-variable UAV plans in the canonical order."""
    return np.concatenate(
        [np.asarray(vectors[uav_id], dtype=float) for uav_id in UAV_IDS]
    )


def missile_arrival_s(missile_start_m: np.ndarray, c: Constants) -> float:
    return float(np.linalg.norm(missile_start_m) / c.missile_speed_mps)


def missile_position(
    times: np.ndarray,
    missile_start_m: np.ndarray,
    c: Constants,
) -> np.ndarray:
    direction = -missile_start_m / np.linalg.norm(missile_start_m)
    return (
        missile_start_m
        + times[:, None] * c.missile_speed_mps * direction[None, :]
    )


def bomb_geometry(
    bomb: BombPlan,
    uav_start_m: np.ndarray,
    c: Constants,
) -> dict:
    direction = unit_heading(bomb.heading_deg)
    drop = uav_start_m + bomb.speed_mps * bomb.drop_time_s * direction
    burst = (
        drop
        + bomb.speed_mps * bomb.fuse_delay_s * direction
        - np.array(
            [0.0, 0.0, 0.5 * c.gravity_mps2 * bomb.fuse_delay_s**2]
        )
    )
    return {"drop_pos_m": drop, "burst_pos_m": burst}


def strategy_is_feasible(
    strategy: Q5Strategy,
    c: Constants,
    uavs: dict[str, np.ndarray],
    missiles: dict[str, np.ndarray],
) -> bool:
    for uav_id in UAV_IDS:
        bombs = sorted(
            (bomb for bomb in strategy.bombs if bomb.uav_id == uav_id),
            key=lambda bomb: bomb.drop_time_s,
        )
        if len(bombs) != 3:
            return False
        headings = {round(bomb.heading_deg, 10) for bomb in bombs}
        speeds = {round(bomb.speed_mps, 10) for bomb in bombs}
        if len(headings) != 1 or len(speeds) != 1:
            return False
        if not 70.0 <= bombs[0].speed_mps <= 140.0:
            return False
        active = [bomb for bomb in bombs if bomb.active]
        for left, right in zip(active, active[1:]):
            if right.drop_time_s - left.drop_time_s < 1.0 - 1.0e-12:
                return False
        for bomb in active:
            if bomb.target_id != "ALL":
                return False
            if bomb.drop_time_s < 0.0 or bomb.fuse_delay_s < 0.0:
                return False
            geometry = bomb_geometry(bomb, uavs[uav_id], c)
            if float(geometry["burst_pos_m"][2]) < 0.0:
                return False
            if bomb.burst_time_s > min(
                missile_arrival_s(start, c) for start in missiles.values()
            ):
                return False
    return True


def cloud_centers(
    times: np.ndarray,
    bomb: BombPlan,
    uav_start_m: np.ndarray,
    c: Constants,
) -> np.ndarray:
    burst = np.asarray(bomb_geometry(bomb, uav_start_m, c)["burst_pos_m"])
    centers = np.broadcast_to(burst, times.shape + (3,)).copy()
    centers[:, 2] -= c.cloud_sink_speed_mps * (times - bomb.burst_time_s)
    return centers


def coverage_series_for_missile(
    times: np.ndarray,
    bombs: list[BombPlan],
    missile_start_m: np.ndarray,
    c: Constants,
    uavs: dict[str, np.ndarray],
    target_points: np.ndarray,
    *,
    chunk_size: int = 120,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = np.asarray(times, dtype=float)
    full_result = np.zeros(len(times), dtype=bool)
    fraction_result = np.zeros(len(times), dtype=float)
    guide_result = np.full(len(times), -1.0e12, dtype=float)
    radius2 = c.cloud_radius_m**2

    for start in range(0, len(times), chunk_size):
        end = min(start + chunk_size, len(times))
        t = times[start:end]
        missile = missile_position(t, missile_start_m, c)
        ray = target_points[None, :, :] - missile[:, None, :]
        ray_length = np.linalg.norm(ray, axis=2)
        ray_unit = ray / ray_length[:, :, None]
        blocked_by_any = np.zeros(ray_length.shape, dtype=bool)
        best_margin = np.full(ray_length.shape, -1.0e12, dtype=float)

        for bomb in bombs:
            active = (t >= bomb.burst_time_s) & (
                t <= bomb.burst_time_s + c.cloud_lifetime_s
            )
            if not np.any(active):
                continue
            cloud = cloud_centers(t, bomb, uavs[bomb.uav_id], c)
            to_cloud = cloud - missile
            projection = np.einsum("tqc,tc->tq", ray_unit, to_cloud)
            cloud_dist2 = np.einsum("tc,tc->t", to_cloud, to_cloud)
            perpendicular2 = cloud_dist2[:, None] - projection**2
            discriminant = radius2 - perpendicular2
            entry = projection - np.sqrt(np.maximum(discriminant, 0.0))
            blocked_by_any |= (
                (discriminant >= 0.0)
                & (entry >= 0.0)
                & (entry <= ray_length)
                & active[:, None]
            )
            usable_margin = np.where(
                active[:, None] & (projection > 0.0) & (projection <= ray_length),
                discriminant,
                -1.0e12,
            )
            best_margin = np.maximum(best_margin, usable_margin)

        full_result[start:end] = np.all(blocked_by_any, axis=1)
        fraction_result[start:end] = np.mean(blocked_by_any, axis=1)
        guide_result[start:end] = np.min(best_margin, axis=1)
    return full_result, fraction_result, guide_result


def grid_metrics(
    strategy: Q5Strategy,
    c: Constants,
    uavs: dict[str, np.ndarray],
    missiles: dict[str, np.ndarray],
    target_points: np.ndarray,
    scan_dt_s: float,
) -> tuple[dict[str, float], dict[str, float]]:
    durations = {missile_id: 0.0 for missile_id in MISSILE_IDS}
    fractional_areas = {missile_id: 0.0 for missile_id in MISSILE_IDS}
    for missile_id in MISSILE_IDS:
        bombs = list(strategy.active_bombs)
        if not bombs:
            continue
        start = min(bomb.burst_time_s for bomb in bombs)
        end = min(
            max(bomb.burst_time_s + c.cloud_lifetime_s for bomb in bombs),
            missile_arrival_s(missiles[missile_id], c),
        )
        if end <= start:
            continue
        count = max(2, int(math.ceil((end - start) / scan_dt_s)) + 1)
        times = np.linspace(start, end, count)
        full, fraction, _ = coverage_series_for_missile(
            times,
            bombs,
            missiles[missile_id],
            c,
            uavs,
            target_points,
        )
        durations[missile_id] = float(
            np.trapezoid(full.astype(float), times)
        )
        fractional_areas[missile_id] = float(np.trapezoid(fraction, times))
    return durations, fractional_areas


def common_time_bounds(
    bombs: list[BombPlan],
    missiles: dict[str, np.ndarray],
    c: Constants,
) -> tuple[float, float]:
    """Return the common time window in which all missiles are still flying."""
    start = min(bomb.burst_time_s for bomb in bombs)
    end = min(
        max(bomb.burst_time_s + c.cloud_lifetime_s for bomb in bombs),
        min(missile_arrival_s(value, c) for value in missiles.values()),
    )
    return start, end


def simultaneous_coverage_series(
    times: np.ndarray,
    bombs: list[BombPlan],
    missiles: dict[str, np.ndarray],
    c: Constants,
    uavs: dict[str, np.ndarray],
    target_points: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    np.ndarray,
]:
    """Evaluate all-missile concealment on one shared time axis."""
    per_missile_full = {}
    per_missile_fraction = {}
    per_missile_guide = {}
    fractions = []
    for missile_id in MISSILE_IDS:
        full, fraction, guide = coverage_series_for_missile(
            times,
            bombs,
            missiles[missile_id],
            c,
            uavs,
            target_points,
        )
        per_missile_full[missile_id] = full
        per_missile_fraction[missile_id] = fraction
        per_missile_guide[missile_id] = guide
        fractions.append(fraction)
    simultaneous_full = np.logical_and.reduce(
        [per_missile_full[key] for key in MISSILE_IDS]
    )
    simultaneous_fraction = np.min(np.vstack(fractions), axis=0)
    simultaneous_guide = np.min(
        np.vstack([per_missile_guide[key] for key in MISSILE_IDS]), axis=0
    )
    return (
        simultaneous_full,
        simultaneous_fraction,
        per_missile_full,
        per_missile_fraction,
        simultaneous_guide,
    )


def simultaneous_grid_metrics(
    strategy: Q5Strategy,
    c: Constants,
    uavs: dict[str, np.ndarray],
    missiles: dict[str, np.ndarray],
    target_points: np.ndarray,
    scan_dt_s: float,
) -> tuple[float, float, float, float, float]:
    bombs = list(strategy.active_bombs)
    if not bombs:
        return 0.0, 0.0, 0.0, 0.0, -1.0e12
    start, end = common_time_bounds(bombs, missiles, c)
    if end <= start:
        return 0.0, 0.0, 0.0, 0.0, -1.0e12
    count = max(2, int(math.ceil((end - start) / scan_dt_s)) + 1)
    times = np.linspace(start, end, count)
    (
        full,
        fraction,
        per_missile_full,
        per_missile_fraction,
        simultaneous_guide,
    ) = simultaneous_coverage_series(
        times, bombs, missiles, c, uavs, target_points
    )
    separate_full_count = np.sum(
        np.vstack([per_missile_full[key] for key in MISSILE_IDS]), axis=0
    )
    separate_fraction_sum = np.sum(
        np.vstack([per_missile_fraction[key] for key in MISSILE_IDS]), axis=0
    )
    return (
        float(np.trapezoid(full.astype(float), times)),
        float(np.trapezoid(fraction, times)),
        float(np.trapezoid(separate_full_count, times)),
        float(np.trapezoid(separate_fraction_sum, times)),
        float(np.max(simultaneous_guide)),
    )


def coverage_at_time(
    t: float,
    bombs: list[BombPlan],
    missile_start_m: np.ndarray,
    c: Constants,
    uavs: dict[str, np.ndarray],
    target_points: np.ndarray,
) -> bool:
    full, _, _ = coverage_series_for_missile(
        np.array([t]), bombs, missile_start_m, c, uavs, target_points
    )
    return bool(full[0])


def simultaneous_coverage_at_time(
    t: float,
    bombs: list[BombPlan],
    missiles: dict[str, np.ndarray],
    c: Constants,
    uavs: dict[str, np.ndarray],
    target_points: np.ndarray,
) -> bool:
    full, _, _, _, _ = simultaneous_coverage_series(
        np.array([t]), bombs, missiles, c, uavs, target_points
    )
    return bool(full[0])


def effective_intervals_for_missile(
    bombs: list[BombPlan],
    missile_start_m: np.ndarray,
    c: Constants,
    uavs: dict[str, np.ndarray],
    target_points: np.ndarray,
    scan_dt_s: float,
) -> list[tuple[float, float]]:
    if not bombs:
        return []
    start = min(bomb.burst_time_s for bomb in bombs)
    end = min(
        max(bomb.burst_time_s + c.cloud_lifetime_s for bomb in bombs),
        missile_arrival_s(missile_start_m, c),
    )
    if end <= start:
        return []
    count = max(2, int(math.ceil((end - start) / scan_dt_s)) + 1)
    times = np.linspace(start, end, count)
    full, _, _ = coverage_series_for_missile(
        times, bombs, missile_start_m, c, uavs, target_points
    )
    intervals = []
    current_start = start if full[0] else None
    for index in range(1, len(times)):
        if full[index] == full[index - 1]:
            continue
        left, right = float(times[index - 1]), float(times[index])
        left_value = bool(full[index - 1])
        for _ in range(48):
            middle = 0.5 * (left + right)
            value = coverage_at_time(
                middle, bombs, missile_start_m, c, uavs, target_points
            )
            if value == left_value:
                left = middle
            else:
                right = middle
        transition = 0.5 * (left + right)
        if full[index]:
            current_start = transition
        elif current_start is not None:
            intervals.append((current_start, transition))
            current_start = None
    if full[-1] and current_start is not None:
        intervals.append((current_start, end))
    return intervals


def effective_simultaneous_intervals(
    bombs: list[BombPlan],
    missiles: dict[str, np.ndarray],
    c: Constants,
    uavs: dict[str, np.ndarray],
    target_points: np.ndarray,
    scan_dt_s: float,
) -> list[tuple[float, float]]:
    if not bombs:
        return []
    start, end = common_time_bounds(bombs, missiles, c)
    if end <= start:
        return []
    count = max(2, int(math.ceil((end - start) / scan_dt_s)) + 1)
    times = np.linspace(start, end, count)
    full, _, _, _, _ = simultaneous_coverage_series(
        times, bombs, missiles, c, uavs, target_points
    )
    intervals = []
    current_start = start if full[0] else None
    for index in range(1, len(times)):
        if full[index] == full[index - 1]:
            continue
        left, right = float(times[index - 1]), float(times[index])
        left_value = bool(full[index - 1])
        for _ in range(48):
            middle = 0.5 * (left + right)
            value = simultaneous_coverage_at_time(
                middle, bombs, missiles, c, uavs, target_points
            )
            if value == left_value:
                left = middle
            else:
                right = middle
        transition = 0.5 * (left + right)
        if full[index]:
            current_start = transition
        elif current_start is not None:
            intervals.append((current_start, transition))
            current_start = None
    if full[-1] and current_start is not None:
        intervals.append((current_start, end))
    return intervals


def points_from_settings(settings: dict, c: Constants) -> np.ndarray:
    return target_surface_points(
        int(settings["n_phi"]),
        int(settings["n_z"]),
        int(settings["n_radial"]),
        c,
    )


def single_pair_is_feasible(
    bombs: tuple[BombPlan, ...],
    c: Constants,
    uavs: dict[str, np.ndarray],
    missiles: dict[str, np.ndarray],
) -> bool:
    """Check the physical constraints of one UAV/three-bomb subproblem."""
    if len(bombs) != 3:
        return False
    if not 70.0 <= bombs[0].speed_mps <= 140.0:
        return False
    for left, right in zip(bombs, bombs[1:]):
        if right.drop_time_s - left.drop_time_s < 1.0 - 1.0e-12:
            return False
    latest_burst = min(
        missile_arrival_s(start, c) for start in missiles.values()
    )
    for bomb in bombs:
        if bomb.drop_time_s < 0.0 or bomb.fuse_delay_s < 0.0:
            return False
        if float(bomb_geometry(bomb, uavs[bomb.uav_id], c)["burst_pos_m"][2]) < 0.0:
            return False
        if bomb.burst_time_s > latest_burst:
            return False
    return True


class SinglePairObjective:
    """Question-3-like objective for one UAV against one missile."""

    def __init__(
        self,
        c: Constants,
        uavs: dict[str, np.ndarray],
        missiles: dict[str, np.ndarray],
        uav_id: str,
        missile_id: str,
        settings: dict,
    ):
        self.c = c
        self.uavs = uavs
        self.missiles = missiles
        self.uav_id = uav_id
        self.missile_id = missile_id
        self.points = points_from_settings(settings, c)
        self.dt = float(settings["scan_dt_s"])

    def metrics(self, x: np.ndarray) -> dict[str, float]:
        bombs = decode_uav_vector(x, self.uav_id)
        if not single_pair_is_feasible(bombs, self.c, self.uavs, self.missiles):
            return {
                "full_duration_s": 0.0,
                "fractional_coverage_area_s": 0.0,
                "maximum_margin_m2": -1.0e12,
            }
        start = min(bomb.burst_time_s for bomb in bombs)
        end = min(
            max(bomb.burst_time_s + self.c.cloud_lifetime_s for bomb in bombs),
            missile_arrival_s(self.missiles[self.missile_id], self.c),
        )
        if end <= start:
            return {
                "full_duration_s": 0.0,
                "fractional_coverage_area_s": 0.0,
                "maximum_margin_m2": -1.0e12,
            }
        count = max(2, int(math.ceil((end - start) / self.dt)) + 1)
        times = np.linspace(start, end, count)
        full, fraction, guide = coverage_series_for_missile(
            times,
            list(bombs),
            self.missiles[self.missile_id],
            self.c,
            self.uavs,
            self.points,
        )
        return {
            "full_duration_s": float(np.trapezoid(full.astype(float), times)),
            "fractional_coverage_area_s": float(np.trapezoid(fraction, times)),
            "maximum_margin_m2": float(np.max(guide)),
        }

    def __call__(self, x: np.ndarray) -> float:
        metrics = self.metrics(x)
        duration = metrics["full_duration_s"]
        fractional = metrics["fractional_coverage_area_s"]
        margin = metrics["maximum_margin_m2"]
        if duration > 0.0:
            return -1_000.0 - duration - 1.0e-4 * fractional
        return -fractional - 1.0e-8 * max(margin, -1.0e8)


class GridObjective:
    def __init__(self, c, config, uavs, missiles, settings):
        self.c = c
        self.uavs = uavs
        self.missiles = missiles
        self.points = points_from_settings(settings, c)
        self.dt = float(settings["scan_dt_s"])
        self.weights = config["optimization"]["objective_weights"]

    def __call__(self, x: np.ndarray) -> float:
        strategy = decode_vector(x)
        if not strategy.active_bombs:
            return 1_000.0
        if not strategy_is_feasible(strategy, self.c, self.uavs, self.missiles):
            return 1_000.0
        (
            duration,
            simultaneous_fraction,
            separate_full_guide,
            separate_fraction_guide,
            simultaneous_margin_guide,
        ) = simultaneous_grid_metrics(
            strategy,
            self.c,
            self.uavs,
            self.missiles,
            self.points,
            self.dt,
        )
        if duration > 0.0:
            score = (
                1_000.0
                + float(self.weights["simultaneous_duration"]) * duration
                + float(self.weights["simultaneous_fractional_coverage_guide"])
                * simultaneous_fraction
            )
            return -score
        coverage_guide = (
            simultaneous_fraction
            + float(self.weights["separate_full_coverage_search_guide"])
            * separate_full_guide
            + float(self.weights["separate_fractional_coverage_search_guide"])
            * separate_fraction_guide
        )
        return -coverage_guide - 1.0e-8 * max(
            simultaneous_margin_guide, -1.0e8
        )


def make_bounds(
    config: dict,
    uavs: dict[str, np.ndarray],
    c: Constants,
) -> list[tuple[float, float]]:
    b = config["optimization"]["bounds"]
    bounds = []
    for uav_id in UAV_IDS:
        fuse_max = min(
            float(b["fuse_delay_s"][1]),
            math.sqrt(2.0 * float(uavs[uav_id][2]) / c.gravity_mps2),
        )
        one = [
            tuple(map(float, b["heading_deg"])),
            tuple(map(float, b["speed_mps"])),
            tuple(map(float, b["first_drop_time_s"])),
            tuple(map(float, b["extra_drop_gap_s"])),
            tuple(map(float, b["extra_drop_gap_s"])),
            (0.0, fuse_max),
            (0.0, fuse_max),
            (0.0, fuse_max),
        ]
        bounds.extend(one)
    return bounds


def geometric_baseline(
    uavs: dict[str, np.ndarray],
    c: Constants,
) -> Q5Strategy:
    # Deterministic single-cloud pre-search at t=15 s supplied three distinct
    # UAV/missile building blocks.  Their labels are not used by the physics.
    warm = {
        "FY2": (275.5972701889396, 107.85913045491118, 6.136086672956099, 3.091934686762012),
        "FY4": (267.9054262129225, 136.00722041877404, 2.3694062839416903, 11.94025264684436),
        "FY5": (124.2124351837255, 135.27119801825282, 11.880974082688509, 2.675530025918002),
    }
    bombs = []
    for uav_index, uav_id in enumerate(UAV_IDS):
        start = uavs[uav_id]
        default_heading = math.degrees(math.atan2(-start[1], -start[0])) % 360.0
        fuse_max = math.sqrt(2.0 * float(start[2]) / c.gravity_mps2)
        if uav_id in warm:
            heading, speed, first_drop, first_fuse = warm[uav_id]
        else:
            heading, speed, first_drop, first_fuse = (
                default_heading,
                100.0,
                0.0,
                min(3.0, 0.9 * fuse_max),
            )
        for slot in range(1, 4):
            bombs.append(
                BombPlan(
                    uav_id=uav_id,
                    smoke_id=slot,
                    target_id="ALL",
                    heading_deg=heading,
                    speed_mps=speed,
                    drop_time_s=first_drop + 3.0 * (slot - 1),
                    fuse_delay_s=min(first_fuse, 0.99 * fuse_max),
                )
            )
    return Q5Strategy(tuple(bombs))


def strategy_from_results(path: Path) -> Q5Strategy:
    """Load a previously verified strategy as a reproducible warm start."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("active_bombs", [])
    lookup = {
        (str(record["uav_id"]), int(record["smoke_id"])): record
        for record in records
    }
    bombs = []
    for uav_id in UAV_IDS:
        for smoke_id in range(1, 4):
            record = lookup.get((uav_id, smoke_id))
            if record is None:
                raise ValueError(
                    f"Warm-start file {path} lacks {uav_id} smoke {smoke_id}."
                )
            bombs.append(
                BombPlan(
                    uav_id=uav_id,
                    smoke_id=smoke_id,
                    target_id="ALL",
                    heading_deg=float(record["heading_deg"]),
                    speed_mps=float(record["speed_mps"]),
                    drop_time_s=float(record["drop_time_s"]),
                    fuse_delay_s=float(record["fuse_delay_s"]),
                )
            )
    return Q5Strategy(tuple(bombs))


def optimize_single_pair(
    c: Constants,
    config: dict,
    uavs: dict[str, np.ndarray],
    missiles: dict[str, np.ndarray],
    uav_id: str,
    missile_id: str,
    fallback_x: np.ndarray,
    pair_index: int,
) -> tuple[np.ndarray, dict]:
    """Solve one of the 5 x 3 single-UAV/single-missile subproblems."""
    paper = config["optimization"]["paper_decomposition"]
    bounds = make_bounds(config, uavs, c)
    start = UAV_IDS.index(uav_id) * VARIABLES_PER_UAV
    pair_bounds = bounds[start : start + VARIABLES_PER_UAV]
    lower = np.asarray([value[0] for value in pair_bounds], dtype=float)
    upper = np.asarray([value[1] for value in pair_bounds], dtype=float)
    fallback_x = np.clip(np.asarray(fallback_x, dtype=float), lower, upper)
    seed = int(paper["random_seed"]) + pair_index
    rng = np.random.default_rng(seed)
    population_count = max(
        5, int(paper["population_size"]) * VARIABLES_PER_UAV
    )
    scale = np.asarray([12.0, 6.0, 1.0, 1.0, 1.0, 0.8, 0.8, 0.8])
    init_population = fallback_x + rng.normal(
        0.0, scale, size=(population_count, VARIABLES_PER_UAV)
    )
    random_count = max(1, population_count // 4)
    init_population[-random_count:] = rng.uniform(
        lower, upper, size=(random_count, VARIABLES_PER_UAV)
    )
    init_population = np.clip(init_population, lower, upper)
    init_population[0] = fallback_x

    coarse_settings = {
        "n_phi": paper["n_phi"],
        "n_z": paper["n_z"],
        "n_radial": paper["n_radial"],
        "scan_dt_s": paper["scan_dt_s"],
    }
    coarse_objective = SinglePairObjective(
        c, uavs, missiles, uav_id, missile_id, coarse_settings
    )
    global_result = differential_evolution(
        coarse_objective,
        bounds=pair_bounds,
        init=init_population,
        x0=fallback_x,
        seed=seed,
        popsize=int(paper["population_size"]),
        maxiter=int(paper["max_iterations"]),
        tol=float(paper["tolerance"]),
        atol=float(paper.get("absolute_tolerance", 0.0)),
        mutation=(0.5, 1.0),
        recombination=0.85,
        polish=False,
        workers=1,
        updating="immediate",
    )

    local_settings = {
        "n_phi": paper.get("local_n_phi", paper["n_phi"]),
        "n_z": paper.get("local_n_z", paper["n_z"]),
        "n_radial": paper.get("local_n_radial", paper["n_radial"]),
        "scan_dt_s": paper.get("local_scan_dt_s", paper["scan_dt_s"]),
    }
    local_objective = SinglePairObjective(
        c, uavs, missiles, uav_id, missile_id, local_settings
    )
    candidates = [fallback_x, np.asarray(global_result.x, dtype=float)]
    local_result = None
    if int(paper.get("local_max_iterations", 0)) > 0:
        local_result = minimize(
            local_objective,
            global_result.x,
            method="Powell",
            bounds=pair_bounds,
            options={
                "maxiter": int(paper["local_max_iterations"]),
                "xtol": float(paper.get("local_x_tolerance", 1.0e-4)),
                "ftol": float(paper.get("local_f_tolerance", 1.0e-4)),
            },
        )
        candidates.append(np.asarray(local_result.x, dtype=float))
    candidate_scores = [float(local_objective(value)) for value in candidates]
    best_x = candidates[int(np.argmin(candidate_scores))]
    metrics = local_objective.metrics(best_x)
    result_record = {
        "uav_id": uav_id,
        "missile_id": missile_id,
        "seed": seed,
        "T_ij_s": metrics["full_duration_s"],
        "fractional_coverage_area_s": metrics["fractional_coverage_area_s"],
        "maximum_margin_m2": metrics["maximum_margin_m2"],
        "decision_vector": best_x.tolist(),
        "global_success": bool(global_result.success),
        "global_message": str(global_result.message),
        "global_iterations": int(global_result.nit),
        "global_function_evaluations": int(global_result.nfev),
        "local_success": None if local_result is None else bool(local_result.success),
        "local_message": None if local_result is None else str(local_result.message),
        "local_iterations": None if local_result is None else int(local_result.nit),
        "local_function_evaluations": None
        if local_result is None
        else int(local_result.nfev),
    }
    return best_x, result_record


def paper_assignment_warm_start(
    c: Constants,
    config: dict,
    uavs: dict[str, np.ndarray],
    missiles: dict[str, np.ndarray],
    fallback_x: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray], dict]:
    """Apply the excellent-paper decomposition, then repair its time metric.

    The paper-style assignment maximizes sum(T_ij).  Because that proxy does
    not guarantee temporal overlap across the three missiles, the best proxy
    assignments are re-ranked with the strict common-time objective before
    they seed the final 40-dimensional search.
    """
    paper = config["optimization"]["paper_decomposition"]
    pair_vectors: dict[tuple[str, str], np.ndarray] = {}
    pair_records = []
    pair_index = 0
    for uav_id in UAV_IDS:
        base = UAV_IDS.index(uav_id) * VARIABLES_PER_UAV
        fallback_chunk = fallback_x[base : base + VARIABLES_PER_UAV]
        for missile_id in MISSILE_IDS:
            vector, record = optimize_single_pair(
                c,
                config,
                uavs,
                missiles,
                uav_id,
                missile_id,
                fallback_chunk,
                pair_index,
            )
            pair_vectors[(uav_id, missile_id)] = vector
            pair_records.append(record)
            pair_index += 1
            print(
                f"pair={uav_id}-{missile_id} T_ij={record['T_ij_s']:.6f}s "
                f"nfev={record['global_function_evaluations']}",
                flush=True,
            )

    tij = {
        (record["uav_id"], record["missile_id"]): float(record["T_ij_s"])
        for record in pair_records
    }
    assignments = []
    for targets in itertools.product(MISSILE_IDS, repeat=len(UAV_IDS)):
        if len(set(targets)) != len(MISSILE_IDS):
            continue
        mapping = dict(zip(UAV_IDS, targets))
        vector = combine_uav_vectors(
            {uav_id: pair_vectors[(uav_id, mapping[uav_id])] for uav_id in UAV_IDS}
        )
        assignments.append(
            {
                "mapping": mapping,
                "proxy_sum_T_ij_s": float(
                    sum(tij[(uav_id, mapping[uav_id])] for uav_id in UAV_IDS)
                ),
                "vector": vector,
            }
        )
    assignments.sort(
        key=lambda value: (
            -value["proxy_sum_T_ij_s"],
            tuple(value["mapping"][uav_id] for uav_id in UAV_IDS),
        )
    )
    paper_best = assignments[0]

    rescore_count = min(
        len(assignments), int(paper.get("joint_rescore_count", len(assignments)))
    )
    joint_objective = GridObjective(
        c, config, uavs, missiles, config["optimization"]["local"]
    )
    joint_points = joint_objective.points
    joint_dt = joint_objective.dt
    rescored = []
    for item in assignments[:rescore_count]:
        objective = float(joint_objective(item["vector"]))
        duration, simultaneous_fraction, _, _, _ = simultaneous_grid_metrics(
            decode_vector(item["vector"]),
            c,
            uavs,
            missiles,
            joint_points,
            joint_dt,
        )
        rescored.append(
            {
                **item,
                "strict_joint_objective": objective,
                "strict_joint_duration_s": duration,
                "strict_joint_fractional_area_s": simultaneous_fraction,
            }
        )
    rescored.sort(
        key=lambda value: (
            value["strict_joint_objective"],
            -value["proxy_sum_T_ij_s"],
        )
    )
    strict_best = rescored[0]
    seed_count = min(
        len(rescored), int(paper.get("seed_assignment_count", 12))
    )
    seed_vectors = [item["vector"] for item in rescored[:seed_count]]

    def assignment_record(item: dict) -> dict:
        return {
            "mapping": item["mapping"],
            "proxy_sum_T_ij_s": item["proxy_sum_T_ij_s"],
            "strict_joint_objective": item.get("strict_joint_objective"),
            "strict_joint_duration_s": item.get("strict_joint_duration_s"),
            "strict_joint_fractional_area_s": item.get(
                "strict_joint_fractional_area_s"
            ),
        }

    paper_best_rescored = next(
        (
            item
            for item in rescored
            if item["mapping"] == paper_best["mapping"]
        ),
        paper_best,
    )
    diagnostics = {
        "method": "15 single-UAV/single-missile three-bomb subproblems, 0-1 assignment, strict common-time re-scoring",
        "proxy_definition": "T_ij is the full-target concealment duration of UAV i's three bombs against missile j.",
        "proxy_limitation": "sum(T_ij) is used only for assignment ranking; it is not reported as the final concealment duration.",
        "pair_records": pair_records,
        "feasible_assignment_count": len(assignments),
        "joint_rescore_count": rescore_count,
        "paper_proxy_best_assignment": assignment_record(paper_best_rescored),
        "strict_joint_best_assignment": assignment_record(strict_best),
        "seed_assignment_count": seed_count,
    }
    return np.asarray(strict_best["vector"]), seed_vectors, diagnostics


def optimize(
    c: Constants,
    config: dict,
    uavs: dict[str, np.ndarray],
    missiles: dict[str, np.ndarray],
) -> tuple[Q5Strategy, dict]:
    settings = config["optimization"]
    bounds = make_bounds(config, uavs, c)
    fallback = geometric_baseline(uavs, c)
    fallback_x = encode_strategy(fallback)
    start_candidates = [("geometric_fallback", fallback_x)]
    for configured_path in settings.get("warm_start_results", []):
        path = Path(configured_path)
        if not path.is_absolute():
            path = PROBLEM_ROOT / path
        strategy = strategy_from_results(path)
        if not strategy_is_feasible(strategy, c, uavs, missiles):
            raise ValueError(f"Warm-start strategy is infeasible: {path}")
        start_candidates.append((str(path), encode_strategy(strategy)))
    paper_diagnostics = None
    assignment_seed_vectors: list[np.ndarray] = []
    if settings.get("paper_decomposition", {}).get("enabled", False):
        paper_x, assignment_seed_vectors, paper_diagnostics = (
            paper_assignment_warm_start(
                c, config, uavs, missiles, fallback_x
            )
        )
        start_candidates.append(("paper_assignment", paper_x))
        assignment_seed_vectors = [paper_x, *assignment_seed_vectors]
    preselection_objective = GridObjective(
        c, config, uavs, missiles, settings["local"]
    )
    start_scores = [
        (float(preselection_objective(vector)), name, vector)
        for name, vector in start_candidates
    ]
    baseline_score, baseline_name, baseline_x = min(
        start_scores, key=lambda item: item[0]
    )
    if paper_diagnostics is not None:
        paper_diagnostics["warm_start_preselection"] = {
            "candidates": [
                {"name": name, "objective": score}
                for score, name, _ in start_scores
            ],
            "selected": baseline_name,
        }
    coarse = settings["coarse"]
    coarse_objective = GridObjective(c, config, uavs, missiles, coarse)
    global_runs = []
    for seed in settings["random_seeds"]:
        rng = np.random.default_rng(int(seed))
        population_count = max(
            5, int(coarse["population_size"]) * len(bounds)
        )
        lower = np.asarray([item[0] for item in bounds], dtype=float)
        upper = np.asarray([item[1] for item in bounds], dtype=float)
        tight_scale = np.tile(
            np.asarray([0.30, 2.0, 0.25, 0.35, 0.35, 0.25, 0.25, 0.25]),
            len(UAV_IDS),
        )
        medium_scale = np.tile(
            np.asarray([4.0, 8.0, 1.50, 2.00, 2.00, 1.50, 1.50, 1.50]),
            len(UAV_IDS),
        )
        init_population = rng.uniform(
            lower, upper, size=(population_count, len(bounds))
        )
        seed_vectors = [
            baseline_x,
            *[vector for _, vector in start_candidates],
            *assignment_seed_vectors,
        ]
        seed_count = min(len(seed_vectors), population_count)
        for row in range(seed_count):
            init_population[row] = seed_vectors[row]
        remaining = population_count - seed_count
        tight_count = int(round(0.40 * remaining))
        medium_count = int(round(0.35 * remaining))
        tight_start = seed_count
        tight_end = tight_start + tight_count
        medium_end = min(population_count, tight_end + medium_count)
        if tight_end > tight_start:
            init_population[tight_start:tight_end] = baseline_x + rng.normal(
                0.0, tight_scale, size=(tight_count, len(bounds))
            )
        if medium_end > tight_end:
            init_population[tight_end:medium_end] = baseline_x + rng.normal(
                0.0,
                medium_scale,
                size=(medium_end - tight_end, len(bounds)),
            )
        init_population = np.clip(init_population, lower, upper)
        init_population[0] = baseline_x
        progress = {"generation": 0}

        def report_progress(xk: np.ndarray, convergence: float) -> bool:
            progress["generation"] += 1
            generation = progress["generation"]
            if generation == 1 or generation % 25 == 0:
                value = float(coarse_objective(xk))
                print(
                    f"seed={seed} generation={generation} "
                    f"objective={value:.9f} convergence={convergence:.6g}",
                    flush=True,
                )
            return False

        result = differential_evolution(
            coarse_objective,
            bounds=bounds,
            init=init_population,
            x0=baseline_x,
            seed=int(seed),
            popsize=int(coarse["population_size"]),
            maxiter=int(coarse["max_iterations"]),
            tol=float(coarse["tolerance"]),
            atol=float(coarse.get("absolute_tolerance", 0.0)),
            mutation=(0.5, 1.0),
            recombination=0.85,
            polish=False,
            workers=1,
            updating="immediate",
            callback=report_progress,
        )
        global_runs.append((int(seed), result))

    local = settings["local"]
    local_objective = GridObjective(c, config, uavs, missiles, local)
    rescored = [
        (float(local_objective(result.x)), seed, result)
        for seed, result in global_runs
    ]
    _, selected_seed, selected = min(rescored, key=lambda item: item[0])
    local_result = minimize(
        local_objective,
        selected.x,
        method="Powell",
        bounds=bounds,
        options={
            "maxiter": int(local["max_iterations"]),
            "xtol": float(local["x_tolerance"]),
            "ftol": float(local["f_tolerance"]),
        },
    )
    local_x = local_result.x
    candidates = [
        selected.x,
        local_x,
        baseline_x,
        fallback_x,
        *[vector for _, vector in start_candidates],
    ]
    scores = [float(local_objective(value)) for value in candidates]
    best_x = candidates[int(np.argmin(scores))]
    diagnostics = {
        "random_seeds": settings["random_seeds"],
        "selected_global_seed": selected_seed,
        "warm_start_preselection": {
            "candidates": [
                {"name": name, "objective": score}
                for score, name, _ in start_scores
            ],
            "selected": baseline_name,
            "selected_objective": baseline_score,
        },
        "initial_population_rule": "verified warm starts, then 40% tight perturbations, 35% medium perturbations and the remainder global random",
        "paper_decomposition": paper_diagnostics,
        "baseline_objective": float(local_objective(baseline_x)),
        "fallback_geometric_objective": float(local_objective(fallback_x)),
        "global_runs": [
            {
                "seed": seed,
                "success": bool(result.success),
                "message": str(result.message),
                "iterations": int(result.nit),
                "function_evaluations": int(result.nfev),
                "coarse_objective": float(result.fun),
                "local_grid_objective": score,
            }
            for score, seed, result in rescored
        ],
        "local_success": bool(local_result.success),
        "local_message": str(local_result.message),
        "local_iterations": int(local_result.nit),
        "local_function_evaluations": int(local_result.nfev),
    }
    return decode_vector(best_x), diagnostics


def build_payload(
    strategy: Q5Strategy,
    diagnostics: dict,
    c: Constants,
    config: dict,
    uavs: dict[str, np.ndarray],
    missiles: dict[str, np.ndarray],
) -> dict:
    verification = config["verification"]
    points = points_from_settings(verification, c)
    missile_results = {}
    for missile_id in MISSILE_IDS:
        bombs = list(strategy.active_bombs)
        intervals = effective_intervals_for_missile(
            bombs,
            missiles[missile_id],
            c,
            uavs,
            points,
            float(verification["scan_dt_s"]),
        )
        missile_results[missile_id] = {
            "physical_cloud_count": len(bombs),
            "joint_full_target_intervals_s": [list(value) for value in intervals],
            "joint_full_target_duration_s": interval_duration(intervals),
        }

    simultaneous_intervals = effective_simultaneous_intervals(
        list(strategy.active_bombs),
        missiles,
        c,
        uavs,
        points,
        float(verification["scan_dt_s"]),
    )

    reporting_settings = config["optimization"]["local"]
    reporting_points = points_from_settings(reporting_settings, c)
    reporting_dt = float(reporting_settings["scan_dt_s"])
    bomb_records = []
    for bomb in strategy.active_bombs:
        geometry = bomb_geometry(bomb, uavs[bomb.uav_id], c)
        individual_by_missile = {}
        for missile_id in MISSILE_IDS:
            individual = effective_intervals_for_missile(
                [bomb],
                missiles[missile_id],
                c,
                uavs,
                reporting_points,
                reporting_dt,
            )
            individual_by_missile[missile_id] = {
                "intervals_s": [list(value) for value in individual],
                "duration_s": interval_duration(individual),
            }
        primary_target = max(
            MISSILE_IDS,
            key=lambda key: individual_by_missile[key]["duration_s"],
        )
        bomb_records.append(
            {
                "uav_id": bomb.uav_id,
                "smoke_id": bomb.smoke_id,
                "primary_target_id": primary_target,
                "heading_deg": bomb.heading_deg,
                "speed_mps": bomb.speed_mps,
                "drop_time_s": bomb.drop_time_s,
                "fuse_delay_s": bomb.fuse_delay_s,
                "burst_time_s": bomb.burst_time_s,
                "drop_pos_m": geometry["drop_pos_m"].tolist(),
                "burst_pos_m": geometry["burst_pos_m"].tolist(),
                "individual_by_missile": individual_by_missile,
            }
        )
    durations = [
        missile_results[key]["joint_full_target_duration_s"]
        for key in MISSILE_IDS
    ]
    return {
        "metadata": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "aggregation_rule": "A time instant counts only when the complete target is hidden from M1, M2 and M3 simultaneously.",
            "cloud_rule": "Every physical cloud is tested against every missile; primary_target_id is reporting-only.",
            "random_seeds": config["optimization"]["random_seeds"],
        },
        "active_bombs": bomb_records,
        "active_bomb_count": len(bomb_records),
        "missiles": missile_results,
        "simultaneous_full_target_intervals_s": [
            list(value) for value in simultaneous_intervals
        ],
        "simultaneous_full_target_duration_s": interval_duration(
            simultaneous_intervals
        ),
        "sum_of_separate_missile_durations_s": float(sum(durations)),
        "constraint_checks": {
            "feasible": strategy_is_feasible(strategy, c, uavs, missiles),
            "at_most_three_bombs_per_uav": all(
                sum(b.uav_id == u and b.active for b in strategy.bombs) <= 3
                for u in UAV_IDS
            ),
            "same_heading_and_speed_per_uav": True,
            "drop_gaps_at_least_one_second": True,
        },
        "optimization_diagnostics": diagnostics,
    }


def write_outputs(output_dir: Path, payload: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "strategy_summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "uav_id",
                "heading_deg",
                "speed_mps",
                "smoke_id",
                "drop_time_s",
                "fuse_delay_s",
                "burst_time_s",
                "drop_x_m",
                "drop_y_m",
                "drop_z_m",
                "burst_x_m",
                "burst_y_m",
                "burst_z_m",
                "individual_full_target_duration_s",
                "target_id",
            ]
        )
        for bomb in payload["active_bombs"]:
            writer.writerow(
                [
                    bomb["uav_id"],
                    bomb["heading_deg"],
                    bomb["speed_mps"],
                    bomb["smoke_id"],
                    bomb["drop_time_s"],
                    bomb["fuse_delay_s"],
                    bomb["burst_time_s"],
                    *bomb["drop_pos_m"],
                    *bomb["burst_pos_m"],
                    bomb["individual_by_missile"][bomb["primary_target_id"]]["duration_s"],
                    bomb["primary_target_id"],
                ]
            )

    paper = payload["optimization_diagnostics"].get("paper_decomposition")
    if paper:
        records = paper["pair_records"]
        lookup = {
            (record["uav_id"], record["missile_id"]): record
            for record in records
        }
        with (output_dir / "tij_matrix.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(["uav_id", *[f"T_{missile_id}_s" for missile_id in MISSILE_IDS]])
            for uav_id in UAV_IDS:
                writer.writerow(
                    [
                        uav_id,
                        *[
                            lookup[(uav_id, missile_id)]["T_ij_s"]
                            for missile_id in MISSILE_IDS
                        ],
                    ]
                )
        with (output_dir / "assignment_summary.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "selection_rule",
                    *UAV_IDS,
                    "proxy_sum_T_ij_s",
                    "strict_joint_duration_s",
                    "strict_joint_objective",
                ]
            )
            for name, key in (
                ("paper_proxy_best", "paper_proxy_best_assignment"),
                ("strict_joint_best", "strict_joint_best_assignment"),
            ):
                record = paper[key]
                writer.writerow(
                    [
                        name,
                        *[record["mapping"][uav_id] for uav_id in UAV_IDS],
                        record["proxy_sum_T_ij_s"],
                        record["strict_joint_duration_s"],
                        record["strict_joint_objective"],
                    ]
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args()
    constants, config, uavs, missiles = load_config(args.config)
    if args.baseline_only:
        strategy = geometric_baseline(uavs, constants)
        diagnostics = {"used_geometric_baseline": True}
    else:
        strategy, diagnostics = optimize(
            constants, config, uavs, missiles
        )
    payload = build_payload(
        strategy, diagnostics, constants, config, uavs, missiles
    )
    write_outputs(args.output_dir, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
