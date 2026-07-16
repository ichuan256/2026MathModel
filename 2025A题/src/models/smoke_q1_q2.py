"""2025 CUMCM A, questions 1 and 2.

The main criterion treats the missile as a point observer.  A cloud is fully
effective only when every line of sight to the cylindrical true target meets
the radius-10 m cloud sphere before reaching the target.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import scipy
from scipy.optimize import differential_evolution, minimize, minimize_scalar


PROBLEM_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROBLEM_ROOT / "configs" / "q1_q2.json"
DEFAULT_OUTPUT_DIR = PROBLEM_ROOT / "outputs" / "model" / "q1_q2"


@dataclass(frozen=True)
class Constants:
    gravity_mps2: float
    missile_speed_mps: float
    cloud_radius_m: float
    cloud_lifetime_s: float
    cloud_sink_speed_mps: float
    target_radius_m: float
    target_height_m: float
    target_base_center_m: tuple[float, float, float]
    missile_pos0_m: tuple[float, float, float]
    uav_pos0_m: tuple[float, float, float]


@dataclass(frozen=True)
class Strategy:
    heading_deg: float
    uav_speed_mps: float
    burst_time_s: float
    fuse_delay_s: float

    @property
    def drop_time_s(self) -> float:
        return self.burst_time_s - self.fuse_delay_s


def load_config(path: Path) -> tuple[Constants, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    c = data["constants"]
    constants = Constants(
        gravity_mps2=float(c["gravity_mps2"]),
        missile_speed_mps=float(c["missile_speed_mps"]),
        cloud_radius_m=float(c["cloud_radius_m"]),
        cloud_lifetime_s=float(c["cloud_lifetime_s"]),
        cloud_sink_speed_mps=float(c["cloud_sink_speed_mps"]),
        target_radius_m=float(c["target_radius_m"]),
        target_height_m=float(c["target_height_m"]),
        target_base_center_m=tuple(map(float, c["target_base_center_m"])),
        missile_pos0_m=tuple(map(float, c["missile_pos0_m"])),
        uav_pos0_m=tuple(map(float, c["uav_pos0_m"])),
    )
    return constants, data


def unit_heading(heading_deg: float) -> np.ndarray:
    angle = math.radians(heading_deg)
    return np.array([math.cos(angle), math.sin(angle), 0.0], dtype=float)


def missile_arrival_s(c: Constants) -> float:
    return float(np.linalg.norm(c.missile_pos0_m) / c.missile_speed_mps)


def missile_position(t: np.ndarray | float, c: Constants) -> np.ndarray:
    t_array = np.asarray(t, dtype=float)
    p0 = np.asarray(c.missile_pos0_m, dtype=float)
    direction = -p0 / np.linalg.norm(p0)
    return p0 + np.expand_dims(t_array, axis=-1) * c.missile_speed_mps * direction


def strategy_geometry(strategy: Strategy, c: Constants) -> dict[str, np.ndarray | float]:
    heading = unit_heading(strategy.heading_deg)
    u0 = np.asarray(c.uav_pos0_m, dtype=float)
    drop_pos = u0 + strategy.uav_speed_mps * strategy.drop_time_s * heading
    burst_pos = (
        drop_pos
        + strategy.uav_speed_mps * strategy.fuse_delay_s * heading
        - np.array(
            [0.0, 0.0, 0.5 * c.gravity_mps2 * strategy.fuse_delay_s**2],
            dtype=float,
        )
    )
    return {
        "drop_time_s": strategy.drop_time_s,
        "burst_time_s": strategy.burst_time_s,
        "drop_pos_m": drop_pos,
        "burst_pos_m": burst_pos,
    }


def cloud_center(t: np.ndarray | float, strategy: Strategy, c: Constants) -> np.ndarray:
    t_array = np.asarray(t, dtype=float)
    burst_pos = np.asarray(strategy_geometry(strategy, c)["burst_pos_m"], dtype=float)
    result = np.broadcast_to(burst_pos, t_array.shape + (3,)).copy()
    result[..., 2] -= c.cloud_sink_speed_mps * (t_array - strategy.burst_time_s)
    return result


def target_boundary_points(n_angle: int, c: Constants) -> np.ndarray:
    """Extreme circles whose convex hull is the complete target cylinder."""
    phi = np.linspace(0.0, 2.0 * math.pi, n_angle, endpoint=False)
    x0, y0, z0 = c.target_base_center_m
    ring_xy = np.column_stack(
        (
            x0 + c.target_radius_m * np.cos(phi),
            y0 + c.target_radius_m * np.sin(phi),
        )
    )
    bottom = np.column_stack((ring_xy, np.full(n_angle, z0)))
    top = np.column_stack((ring_xy, np.full(n_angle, z0 + c.target_height_m)))
    return np.vstack((bottom, top))


def coverage_series(
    times: np.ndarray,
    strategy: Strategy,
    c: Constants,
    target_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return full-coverage booleans and worst tangent margins for all times."""
    p = missile_position(times, c)
    cloud = cloud_center(times, strategy, c)
    ray = target_points[None, :, :] - p[:, None, :]
    ray_length = np.linalg.norm(ray, axis=2)
    ray_unit = ray / ray_length[:, :, None]
    to_cloud = cloud - p
    projection = np.einsum("tqc,tc->tq", ray_unit, to_cloud)
    cloud_dist2 = np.einsum("tc,tc->t", to_cloud, to_cloud)
    perpendicular2 = cloud_dist2[:, None] - projection**2
    discriminant = c.cloud_radius_m**2 - perpendicular2
    root = np.sqrt(np.maximum(discriminant, 0.0))
    entry_distance = projection - root
    ray_is_blocked = (
        (discriminant >= 0.0)
        & (entry_distance >= 0.0)
        & (entry_distance <= ray_length)
    )
    full = np.all(ray_is_blocked, axis=1)
    worst_margin = np.min(discriminant, axis=1)
    return full, worst_margin


def strategy_is_feasible(strategy: Strategy, c: Constants) -> bool:
    if not (0.0 <= strategy.heading_deg < 360.0):
        return False
    if not (70.0 <= strategy.uav_speed_mps <= 140.0):
        return False
    if strategy.drop_time_s < 0.0 or strategy.fuse_delay_s < 0.0:
        return False
    geometry = strategy_geometry(strategy, c)
    if np.asarray(geometry["drop_pos_m"])[2] < 0.0:
        return False
    if np.asarray(geometry["burst_pos_m"])[2] < 0.0:
        return False
    if strategy.burst_time_s > missile_arrival_s(c):
        return False
    return True


def coverage_at_time(
    t: float,
    strategy: Strategy,
    c: Constants,
    target_points: np.ndarray,
) -> bool:
    full, _ = coverage_series(np.array([t]), strategy, c, target_points)
    return bool(full[0])


def _transition_time(
    left: float,
    right: float,
    left_value: bool,
    strategy: Strategy,
    c: Constants,
    target_points: np.ndarray,
) -> float:
    for _ in range(50):
        mid = 0.5 * (left + right)
        mid_value = coverage_at_time(mid, strategy, c, target_points)
        if mid_value == left_value:
            left = mid
        else:
            right = mid
    return 0.5 * (left + right)


def effective_intervals(
    strategy: Strategy,
    c: Constants,
    *,
    n_angle: int,
    scan_dt_s: float,
) -> list[tuple[float, float]]:
    if not strategy_is_feasible(strategy, c):
        return []
    start = strategy.burst_time_s
    end = min(start + c.cloud_lifetime_s, missile_arrival_s(c))
    if end <= start:
        return []
    count = max(2, int(math.ceil((end - start) / scan_dt_s)) + 1)
    times = np.linspace(start, end, count)
    points = target_boundary_points(n_angle, c)
    full, _ = coverage_series(times, strategy, c, points)

    intervals: list[tuple[float, float]] = []
    current_start: float | None = start if full[0] else None
    for idx in range(1, len(times)):
        if full[idx] == full[idx - 1]:
            continue
        transition = _transition_time(
            float(times[idx - 1]),
            float(times[idx]),
            bool(full[idx - 1]),
            strategy,
            c,
            points,
        )
        if full[idx]:
            current_start = transition
        else:
            if current_start is not None:
                intervals.append((current_start, transition))
            current_start = None
    if full[-1] and current_start is not None:
        intervals.append((current_start, end))
    return intervals


def interval_duration(intervals: Iterable[tuple[float, float]]) -> float:
    return float(sum(right - left for left, right in intervals))


def center_line_intervals(
    strategy: Strategy,
    c: Constants,
    *,
    scan_dt_s: float,
) -> list[tuple[float, float]]:
    """Baseline: only the line of sight to the target center must be blocked."""
    x0, y0, z0 = c.target_base_center_m
    center = np.array([x0, y0, z0 + 0.5 * c.target_height_m])
    start = strategy.burst_time_s
    end = min(start + c.cloud_lifetime_s, missile_arrival_s(c))
    count = max(2, int(math.ceil((end - start) / scan_dt_s)) + 1)
    times = np.linspace(start, end, count)
    p = missile_position(times, c)
    cloud = cloud_center(times, strategy, c)
    ray = center - p
    ray_length = np.linalg.norm(ray, axis=1)
    ray_unit = ray / ray_length[:, None]
    to_cloud = cloud - p
    projection = np.einsum("tc,tc->t", ray_unit, to_cloud)
    perpendicular2 = np.einsum("tc,tc->t", to_cloud, to_cloud) - projection**2
    disc = c.cloud_radius_m**2 - perpendicular2
    entry = projection - np.sqrt(np.maximum(disc, 0.0))
    full = (disc >= 0.0) & (entry >= 0.0) & (entry <= ray_length)

    def predicate(t: float) -> bool:
        pp = missile_position(t, c)
        cc = cloud_center(t, strategy, c)
        rr = center - pp
        length = np.linalg.norm(rr)
        uu = rr / length
        vv = cc - pp
        proj = float(np.dot(uu, vv))
        d2 = float(np.dot(vv, vv) - proj**2)
        discriminant = c.cloud_radius_m**2 - d2
        if discriminant < 0:
            return False
        entry_distance = proj - math.sqrt(max(discriminant, 0.0))
        return 0.0 <= entry_distance <= length

    intervals: list[tuple[float, float]] = []
    current_start: float | None = start if full[0] else None
    for idx in range(1, len(times)):
        if full[idx] == full[idx - 1]:
            continue
        left, right = float(times[idx - 1]), float(times[idx])
        left_value = bool(full[idx - 1])
        for _ in range(50):
            mid = 0.5 * (left + right)
            if predicate(mid) == left_value:
                left = mid
            else:
                right = mid
        transition = 0.5 * (left + right)
        if full[idx]:
            current_start = transition
        else:
            if current_start is not None:
                intervals.append((current_start, transition))
            current_start = None
    if full[-1] and current_start is not None:
        intervals.append((current_start, end))
    return intervals


class Q2Objective:
    def __init__(self, c: Constants, n_angle: int, scan_dt_s: float):
        self.c = c
        self.n_angle = n_angle
        self.scan_dt_s = scan_dt_s
        self.target_points = target_boundary_points(n_angle, c)

    def __call__(self, x: np.ndarray) -> float:
        strategy = Strategy(
            heading_deg=float(x[0] % 360.0),
            uav_speed_mps=float(x[1]),
            burst_time_s=float(x[2]),
            fuse_delay_s=float(x[3]),
        )
        if not strategy_is_feasible(strategy, self.c):
            violation = max(0.0, -strategy.drop_time_s)
            violation += max(
                0.0,
                -float(np.asarray(strategy_geometry(strategy, self.c)["burst_pos_m"])[2]),
            )
            return 1_000.0 + violation

        start = strategy.burst_time_s
        end = min(start + self.c.cloud_lifetime_s, missile_arrival_s(self.c))
        count = max(2, int(math.ceil((end - start) / self.scan_dt_s)) + 1)
        times = np.linspace(start, end, count)
        full, margin = coverage_series(times, strategy, self.c, self.target_points)
        duration_grid = float(np.trapezoid(full.astype(float), times))
        guide = float(np.max(margin))
        if np.any(full):
            return -100.0 - duration_grid - 1.0e-5 * min(guide, 100.0)
        return -1.0e-5 * max(guide, -1.0e7)


class Q2IntervalDurationObjective:
    """Accurate local objective based on refined effective-interval endpoints."""

    def __init__(self, c: Constants, n_angle: int, scan_dt_s: float):
        self.c = c
        self.n_angle = n_angle
        self.scan_dt_s = scan_dt_s

    def __call__(self, x: np.ndarray) -> float:
        strategy = Strategy(
            heading_deg=float(x[0] % 360.0),
            uav_speed_mps=float(x[1]),
            burst_time_s=float(x[2]),
            fuse_delay_s=float(x[3]),
        )
        if not strategy_is_feasible(strategy, self.c):
            return 1_000.0
        intervals = effective_intervals(
            strategy,
            self.c,
            n_angle=self.n_angle,
            scan_dt_s=self.scan_dt_s,
        )
        return -interval_duration(intervals)


def optimize_q2(c: Constants, settings: dict) -> tuple[Strategy, dict]:
    seeds = [int(value) for value in settings["random_seeds"]]
    arrival = missile_arrival_s(c)
    max_fuse = math.sqrt(2.0 * c.uav_pos0_m[2] / c.gravity_mps2)
    bounds = [
        (0.0, 360.0),
        (70.0, 140.0),
        (0.0, arrival),
        (0.0, max_fuse),
    ]
    coarse = settings["coarse"]
    objective = Q2Objective(
        c,
        n_angle=int(coarse["target_angles"]),
        scan_dt_s=float(coarse["scan_dt_s"]),
    )
    global_results = []
    for seed in seeds:
        result = differential_evolution(
            objective,
            bounds=bounds,
            seed=seed,
            popsize=int(coarse["population_size"]),
            maxiter=int(coarse["max_iterations"]),
            tol=float(coarse["tolerance"]),
            mutation=(0.5, 1.0),
            recombination=0.8,
            polish=False,
            workers=1,
            updating="immediate",
        )
        global_results.append((seed, result))

    local_settings = settings["local"]
    local_objective = Q2Objective(
        c,
        n_angle=int(local_settings["target_angles"]),
        scan_dt_s=float(local_settings["scan_dt_s"]),
    )
    scored_global_results = [
        (float(local_objective(result.x)), seed, result)
        for seed, result in global_results
    ]
    _, selected_seed, global_result = min(scored_global_results, key=lambda x: x[0])
    local_result = minimize(
        local_objective,
        global_result.x,
        method="Nelder-Mead",
        options={
            "maxiter": int(local_settings["max_iterations"]),
            "xatol": float(local_settings["x_tolerance"]),
            "fatol": float(local_settings["f_tolerance"]),
        },
    )
    global_at_local_resolution = float(local_objective(global_result.x))
    grid_best_x = (
        local_result.x
        if local_result.fun < global_at_local_resolution
        else global_result.x
    )

    interval_settings = settings["interval_refinement"]
    duration_objective = Q2IntervalDurationObjective(
        c,
        n_angle=int(interval_settings["target_angles"]),
        scan_dt_s=float(interval_settings["scan_dt_s"]),
    )
    interval_result = minimize(
        duration_objective,
        grid_best_x,
        method="Nelder-Mead",
        options={
            "maxiter": int(interval_settings["max_iterations"]),
            "xatol": float(interval_settings["x_tolerance"]),
            "fatol": float(interval_settings["f_tolerance"]),
        },
    )

    candidate_x = interval_result.x
    candidate_strategy = Strategy(
        heading_deg=float(candidate_x[0] % 360.0),
        uav_speed_mps=float(candidate_x[1]),
        burst_time_s=float(candidate_x[2]),
        fuse_delay_s=float(candidate_x[3]),
    )

    boundary_result = None
    if (
        candidate_strategy.uav_speed_mps <= 70.05
        and candidate_strategy.drop_time_s <= 0.05
    ):
        boundary_settings = settings["boundary_refinement"]

        def boundary_objective(y: np.ndarray) -> float:
            strategy = Strategy(
                heading_deg=float(y[0] % 360.0),
                uav_speed_mps=70.0,
                burst_time_s=float(y[1]),
                fuse_delay_s=float(y[1]),
            )
            if not strategy_is_feasible(strategy, c):
                return 1_000.0
            return -interval_duration(
                effective_intervals(
                    strategy,
                    c,
                    n_angle=int(boundary_settings["target_angles"]),
                    scan_dt_s=float(boundary_settings["scan_dt_s"]),
                )
            )

        boundary_result = minimize(
            boundary_objective,
            np.array(
                [candidate_strategy.heading_deg, candidate_strategy.fuse_delay_s]
            ),
            method="Nelder-Mead",
            options={
                "maxiter": int(boundary_settings["max_iterations"]),
                "xatol": float(boundary_settings["x_tolerance"]),
                "fatol": float(boundary_settings["f_tolerance"]),
            },
        )
        if boundary_result.fun < interval_result.fun:
            candidate_x = np.array(
                [
                    boundary_result.x[0] % 360.0,
                    70.0,
                    boundary_result.x[1],
                    boundary_result.x[1],
                ]
            )

    best_x = candidate_x
    best = Strategy(
        heading_deg=float(best_x[0] % 360.0),
        uav_speed_mps=float(np.clip(best_x[1], 70.0, 140.0)),
        burst_time_s=float(best_x[2]),
        fuse_delay_s=float(best_x[3]),
    )
    diagnostics = {
        "random_seeds": seeds,
        "selected_global_seed": selected_seed,
        "global_runs": [
            {
                "seed": seed,
                "success": bool(result.success),
                "message": str(result.message),
                "iterations": int(result.nit),
                "function_evaluations": int(result.nfev),
                "coarse_objective": float(result.fun),
                "local_resolution_objective": score,
            }
            for score, seed, result in scored_global_results
        ],
        "global_success": bool(global_result.success),
        "global_message": str(global_result.message),
        "global_iterations": int(global_result.nit),
        "global_function_evaluations": int(global_result.nfev),
        "global_objective": float(global_result.fun),
        "global_objective_at_local_resolution": global_at_local_resolution,
        "local_success": bool(local_result.success),
        "local_message": str(local_result.message),
        "local_iterations": int(local_result.nit),
        "local_function_evaluations": int(local_result.nfev),
        "local_objective": float(local_result.fun),
        "interval_success": bool(interval_result.success),
        "interval_message": str(interval_result.message),
        "interval_iterations": int(interval_result.nit),
        "interval_function_evaluations": int(interval_result.nfev),
        "interval_objective": float(interval_result.fun),
        "boundary_refinement_used": boundary_result is not None,
        "boundary_success": (
            bool(boundary_result.success) if boundary_result is not None else None
        ),
        "boundary_iterations": (
            int(boundary_result.nit) if boundary_result is not None else None
        ),
        "boundary_function_evaluations": (
            int(boundary_result.nfev) if boundary_result is not None else None
        ),
        "boundary_objective": (
            float(boundary_result.fun) if boundary_result is not None else None
        ),
    }
    return best, diagnostics


def exact_ring_max_perpendicular2(
    t: float,
    strategy: Strategy,
    c: Constants,
    *,
    seed_count: int = 180,
) -> float:
    """Continuously refine the worst point on both cylinder rim circles."""
    p = np.asarray(missile_position(t, c), dtype=float)
    center = np.asarray(cloud_center(t, strategy, c), dtype=float)
    to_cloud = center - p
    x0, y0, z0 = c.target_base_center_m
    step = 2.0 * math.pi / seed_count
    phi_seed = np.arange(seed_count) * step
    worst = -math.inf

    def perpendicular2(phi: float, z: float) -> float:
        q = np.array(
            [
                x0 + c.target_radius_m * math.cos(phi),
                y0 + c.target_radius_m * math.sin(phi),
                z,
            ]
        )
        ray = q - p
        ray_unit = ray / np.linalg.norm(ray)
        projection = float(np.dot(ray_unit, to_cloud))
        return float(np.dot(to_cloud, to_cloud) - projection**2)

    for z in (z0, z0 + c.target_height_m):
        values = np.array([perpendicular2(phi, z) for phi in phi_seed])
        candidates = np.argsort(values)[-4:]
        for idx in candidates:
            center_phi = float(phi_seed[idx])
            result = minimize_scalar(
                lambda phi: -perpendicular2(phi % (2.0 * math.pi), z),
                bounds=(center_phi - step, center_phi + step),
                method="bounded",
                options={"xatol": 1.0e-13},
            )
            worst = max(worst, -float(result.fun))
    return worst


def refined_interval_boundaries(
    intervals: list[tuple[float, float]],
    strategy: Strategy,
    c: Constants,
) -> list[tuple[float, float]]:
    """Refine sampled target boundaries using continuous circle maximization."""
    if not intervals:
        return []
    radius2 = c.cloud_radius_m**2

    def margin(t: float) -> float:
        return radius2 - exact_ring_max_perpendicular2(t, strategy, c)

    refined: list[tuple[float, float]] = []
    for left, right in intervals:
        width = max(right - left, 1.0e-4)
        search = min(0.05, 0.2 * width)
        lo = max(strategy.burst_time_s, left - search)
        hi = min(
            strategy.burst_time_s + c.cloud_lifetime_s,
            missile_arrival_s(c),
            right + search,
        )

        left_a, left_b = lo, min(left + search, right)
        if margin(left_a) * margin(left_b) <= 0.0:
            for _ in range(55):
                mid = 0.5 * (left_a + left_b)
                if margin(left_a) * margin(mid) <= 0.0:
                    left_b = mid
                else:
                    left_a = mid
            left = 0.5 * (left_a + left_b)

        right_a, right_b = max(left, right - search), hi
        if margin(right_a) * margin(right_b) <= 0.0:
            for _ in range(55):
                mid = 0.5 * (right_a + right_b)
                if margin(right_a) * margin(mid) <= 0.0:
                    right_b = mid
                else:
                    right_a = mid
            right = 0.5 * (right_a + right_b)
        refined.append((left, right))
    return refined


def strategy_record(
    name: str,
    strategy: Strategy,
    c: Constants,
    settings: dict,
) -> dict:
    geometry = strategy_geometry(strategy, c)
    verification = settings["verification"]
    sampled_intervals = effective_intervals(
        strategy,
        c,
        n_angle=int(verification["target_angles"]),
        scan_dt_s=float(verification["scan_dt_s"]),
    )
    strict_intervals = refined_interval_boundaries(sampled_intervals, strategy, c)
    center_intervals = center_line_intervals(
        strategy,
        c,
        scan_dt_s=float(verification["scan_dt_s"]),
    )
    burst_pos = np.asarray(geometry["burst_pos_m"])
    record = {
        "name": name,
        "strategy": asdict(strategy),
        "drop_time_s": float(geometry["drop_time_s"]),
        "drop_pos_m": np.asarray(geometry["drop_pos_m"]).tolist(),
        "burst_pos_m": burst_pos.tolist(),
        "strict_full_target_intervals_s": [list(x) for x in strict_intervals],
        "strict_full_target_duration_s": interval_duration(strict_intervals),
        "center_line_intervals_s": [list(x) for x in center_intervals],
        "center_line_duration_s": interval_duration(center_intervals),
        "constraint_checks": {
            "feasible": strategy_is_feasible(strategy, c),
            "speed_in_range": 70.0 <= strategy.uav_speed_mps <= 140.0,
            "drop_time_nonnegative": strategy.drop_time_s >= 0.0,
            "burst_height_nonnegative": float(burst_pos[2]) >= 0.0,
            "burst_before_missile_arrival": strategy.burst_time_s
            <= missile_arrival_s(c),
        },
    }
    return record


def write_outputs(output_dir: Path, payload: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "strategy_summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "question",
                "heading_deg",
                "uav_speed_mps",
                "drop_time_s",
                "fuse_delay_s",
                "burst_time_s",
                "drop_x_m",
                "drop_y_m",
                "drop_z_m",
                "burst_x_m",
                "burst_y_m",
                "burst_z_m",
                "strict_duration_s",
                "center_line_duration_s",
            ]
        )
        for key in ("question_1", "question_2"):
            record = payload[key]
            strategy = record["strategy"]
            writer.writerow(
                [
                    key,
                    strategy["heading_deg"],
                    strategy["uav_speed_mps"],
                    record["drop_time_s"],
                    strategy["fuse_delay_s"],
                    strategy["burst_time_s"],
                    *record["drop_pos_m"],
                    *record["burst_pos_m"],
                    record["strict_full_target_duration_s"],
                    record["center_line_duration_s"],
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--skip-optimize", action="store_true")
    args = parser.parse_args()

    constants, config = load_config(args.config)
    q1 = Strategy(
        heading_deg=180.0,
        uav_speed_mps=120.0,
        burst_time_s=1.5 + 3.6,
        fuse_delay_s=3.6,
    )

    if args.skip_optimize:
        stored = config["q2_stored_strategy"]
        q2 = Strategy(**stored)
        diagnostics = {"used_stored_strategy": True}
    else:
        q2, diagnostics = optimize_q2(constants, config["optimization"])

    payload = {
        "metadata": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "config": str(args.config),
            "criterion": (
                "Every line of sight from the missile to the cylindrical target "
                "must intersect the cloud sphere before reaching the target."
            ),
        },
        "constants": asdict(constants),
        "missile_arrival_s": missile_arrival_s(constants),
        "question_1": strategy_record("question_1", q1, constants, config),
        "question_2": strategy_record("question_2", q2, constants, config),
        "optimization_diagnostics": diagnostics,
    }
    write_outputs(args.output_dir, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
