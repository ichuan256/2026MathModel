"""2025 CUMCM A, question 3: one UAV, three jointly covering clouds."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import differential_evolution, minimize


PROBLEM_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROBLEM_ROOT / "src" / "models"))

from smoke_q1_q2 import (  # noqa: E402
    Constants,
    Strategy,
    effective_intervals,
    interval_duration,
    load_config as load_q1_q2_config,
    missile_arrival_s,
    missile_position,
    refined_interval_boundaries,
    strategy_geometry,
    unit_heading,
)


DEFAULT_CONFIG = PROBLEM_ROOT / "configs" / "q3.json"
DEFAULT_OUTPUT_DIR = PROBLEM_ROOT / "outputs" / "model" / "q3"


@dataclass(frozen=True)
class Q3Strategy:
    heading_deg: float
    uav_speed_mps: float
    drop_times_s: tuple[float, float, float]
    fuse_delays_s: tuple[float, float, float]

    @property
    def burst_times_s(self) -> tuple[float, float, float]:
        return tuple(
            drop + fuse
            for drop, fuse in zip(self.drop_times_s, self.fuse_delays_s)
        )


def load_config(path: Path) -> tuple[Constants, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    constants, _ = load_q1_q2_config(PROBLEM_ROOT / "configs" / "q1_q2.json")
    return constants, data


def decode_vector(x: np.ndarray) -> Q3Strategy:
    first_drop = float(x[2])
    second_drop = first_drop + 1.0 + float(x[3])
    third_drop = second_drop + 1.0 + float(x[4])
    return Q3Strategy(
        heading_deg=float(x[0] % 360.0),
        uav_speed_mps=float(x[1]),
        drop_times_s=(first_drop, second_drop, third_drop),
        fuse_delays_s=(float(x[5]), float(x[6]), float(x[7])),
    )


def encode_strategy(strategy: Q3Strategy) -> np.ndarray:
    d1, d2, d3 = strategy.drop_times_s
    return np.array(
        [
            strategy.heading_deg,
            strategy.uav_speed_mps,
            d1,
            d2 - d1 - 1.0,
            d3 - d2 - 1.0,
            *strategy.fuse_delays_s,
        ],
        dtype=float,
    )


def bomb_strategies(strategy: Q3Strategy) -> list[Strategy]:
    return [
        Strategy(
            heading_deg=strategy.heading_deg,
            uav_speed_mps=strategy.uav_speed_mps,
            burst_time_s=drop + fuse,
            fuse_delay_s=fuse,
        )
        for drop, fuse in zip(strategy.drop_times_s, strategy.fuse_delays_s)
    ]


def q3_is_feasible(strategy: Q3Strategy, c: Constants) -> bool:
    if not (0.0 <= strategy.heading_deg < 360.0):
        return False
    if not (70.0 <= strategy.uav_speed_mps <= 140.0):
        return False
    drops = strategy.drop_times_s
    if drops[0] < 0.0:
        return False
    if drops[1] - drops[0] < 1.0 - 1.0e-12:
        return False
    if drops[2] - drops[1] < 1.0 - 1.0e-12:
        return False
    arrival = missile_arrival_s(c)
    for bomb in bomb_strategies(strategy):
        geometry = strategy_geometry(bomb, c)
        if bomb.fuse_delay_s < 0.0:
            return False
        if bomb.burst_time_s > arrival:
            return False
        if float(np.asarray(geometry["burst_pos_m"])[2]) < 0.0:
            return False
    return True


def target_surface_points(
    n_phi: int,
    n_z: int,
    n_radial: int,
    c: Constants,
) -> np.ndarray:
    """Sample the entire cylinder surface, not just its silhouette boundary."""
    phi = np.linspace(0.0, 2.0 * math.pi, n_phi, endpoint=False)
    x0, y0, z0 = c.target_base_center_m
    radius = c.target_radius_m

    z_values = np.linspace(z0, z0 + c.target_height_m, n_z)
    side = np.array(
        [
            [
                x0 + radius * math.cos(angle),
                y0 + radius * math.sin(angle),
                z,
            ]
            for z in z_values
            for angle in phi
        ],
        dtype=float,
    )

    disk_points = []
    radial_values = np.linspace(0.0, radius, n_radial)
    for z in (z0, z0 + c.target_height_m):
        disk_points.append([x0, y0, z])
        for radial in radial_values[1:]:
            disk_points.extend(
                [
                    [
                        x0 + radial * math.cos(angle),
                        y0 + radial * math.sin(angle),
                        z,
                    ]
                    for angle in phi
                ]
            )
    return np.vstack((side, np.asarray(disk_points, dtype=float)))


def cloud_centers_for_bomb(
    times: np.ndarray,
    bomb: Strategy,
    c: Constants,
) -> np.ndarray:
    geometry = strategy_geometry(bomb, c)
    burst = np.asarray(geometry["burst_pos_m"], dtype=float)
    centers = np.broadcast_to(burst, times.shape + (3,)).copy()
    centers[:, 2] -= c.cloud_sink_speed_mps * (times - bomb.burst_time_s)
    return centers


def joint_coverage_series(
    times: np.ndarray,
    strategy: Q3Strategy,
    c: Constants,
    target_points: np.ndarray,
    *,
    chunk_size: int = 160,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return joint flags/fractions, guide, and each cloud's full flags."""
    times = np.asarray(times, dtype=float)
    full_result = np.zeros(times.shape, dtype=bool)
    fraction_result = np.zeros(times.shape, dtype=float)
    guide_result = np.full(times.shape, -1.0e12, dtype=float)
    individual_full_result = np.zeros((3, len(times)), dtype=bool)
    bombs = bomb_strategies(strategy)
    radius2 = c.cloud_radius_m**2

    for start in range(0, len(times), chunk_size):
        end = min(start + chunk_size, len(times))
        t = times[start:end]
        missile = missile_position(t, c)
        ray = target_points[None, :, :] - missile[:, None, :]
        ray_length = np.linalg.norm(ray, axis=2)
        ray_unit = ray / ray_length[:, :, None]

        blocked_by_any = np.zeros(ray_length.shape, dtype=bool)
        best_ray_margin = np.full(ray_length.shape, -1.0e12, dtype=float)

        for bomb_index, bomb in enumerate(bombs):
            active = (
                (t >= bomb.burst_time_s)
                & (t <= bomb.burst_time_s + c.cloud_lifetime_s)
            )
            if not np.any(active):
                continue
            cloud = cloud_centers_for_bomb(t, bomb, c)
            to_cloud = cloud - missile
            projection = np.einsum("tqc,tc->tq", ray_unit, to_cloud)
            cloud_dist2 = np.einsum("tc,tc->t", to_cloud, to_cloud)
            perpendicular2 = cloud_dist2[:, None] - projection**2
            discriminant = radius2 - perpendicular2
            root = np.sqrt(np.maximum(discriminant, 0.0))
            entry_distance = projection - root
            valid_order = (entry_distance >= 0.0) & (entry_distance <= ray_length)
            blocked = (
                (discriminant >= 0.0)
                & valid_order
                & active[:, None]
            )
            blocked_by_any |= blocked
            individual_full_result[bomb_index, start:end] = np.all(
                blocked, axis=1
            )
            ordered_margin = np.where(
                active[:, None] & (projection > 0.0) & (projection <= ray_length),
                discriminant,
                -1.0e12,
            )
            best_ray_margin = np.maximum(best_ray_margin, ordered_margin)

        full_result[start:end] = np.all(blocked_by_any, axis=1)
        fraction_result[start:end] = np.mean(blocked_by_any, axis=1)
        guide_result[start:end] = np.min(best_ray_margin, axis=1)
    return (
        full_result,
        fraction_result,
        guide_result,
        individual_full_result,
    )


def joint_coverage_at_time(
    t: float,
    strategy: Q3Strategy,
    c: Constants,
    target_points: np.ndarray,
) -> bool:
    flags, _, _, _ = joint_coverage_series(
        np.array([t], dtype=float),
        strategy,
        c,
        target_points,
        chunk_size=1,
    )
    return bool(flags[0])


def transition_time(
    left: float,
    right: float,
    left_value: bool,
    strategy: Q3Strategy,
    c: Constants,
    target_points: np.ndarray,
) -> float:
    for _ in range(48):
        mid = 0.5 * (left + right)
        value = joint_coverage_at_time(mid, strategy, c, target_points)
        if value == left_value:
            left = mid
        else:
            right = mid
    return 0.5 * (left + right)


def joint_effective_intervals(
    strategy: Q3Strategy,
    c: Constants,
    target_points: np.ndarray,
    *,
    scan_dt_s: float,
) -> list[tuple[float, float]]:
    if not q3_is_feasible(strategy, c):
        return []
    bursts = strategy.burst_times_s
    start = min(bursts)
    end = min(
        max(value + c.cloud_lifetime_s for value in bursts),
        missile_arrival_s(c),
    )
    count = max(2, int(math.ceil((end - start) / scan_dt_s)) + 1)
    times = np.linspace(start, end, count)
    full, _, _, _ = joint_coverage_series(times, strategy, c, target_points)

    intervals: list[tuple[float, float]] = []
    current_start = start if full[0] else None
    for idx in range(1, len(times)):
        if full[idx] == full[idx - 1]:
            continue
        transition = transition_time(
            float(times[idx - 1]),
            float(times[idx]),
            bool(full[idx - 1]),
            strategy,
            c,
            target_points,
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


class Q3GridObjective:
    def __init__(
        self,
        c: Constants,
        target_points: np.ndarray,
        scan_dt_s: float,
    ):
        self.c = c
        self.target_points = target_points
        self.scan_dt_s = scan_dt_s

    def __call__(self, x: np.ndarray) -> float:
        strategy = decode_vector(x)
        if not q3_is_feasible(strategy, self.c):
            return 1_000.0
        bursts = strategy.burst_times_s
        start = min(bursts)
        end = min(
            max(value + self.c.cloud_lifetime_s for value in bursts),
            missile_arrival_s(self.c),
        )
        count = max(2, int(math.ceil((end - start) / self.scan_dt_s)) + 1)
        times = np.linspace(start, end, count)
        full, covered_fraction, guide, individual_full = joint_coverage_series(
            times, strategy, self.c, self.target_points
        )
        duration = float(np.trapezoid(full.astype(float), times))
        individual_durations = np.trapezoid(
            individual_full.astype(float), times, axis=1
        )
        individual_duration_sum = float(np.sum(individual_durations))
        weakest_cloud_duration = float(np.min(individual_durations))
        coverage_area = float(np.trapezoid(covered_fraction, times))
        max_guide = float(np.max(guide))
        if np.any(full):
            # Strict duration is primary.  The small fractional-coverage term
            # gives an otherwise unused cloud a direction in which to improve.
            return (
                -100.0
                - duration
                - 0.01 * individual_duration_sum
                - 0.05 * weakest_cloud_duration
                - 1.0e-4 * coverage_area
                - 1.0e-8 * min(max_guide, 100.0)
            )
        # Before any full-coverage instant exists, use the covered fraction to
        # guide the population toward the feasible full-coverage region.
        return -coverage_area - 1.0e-8 * max(max_guide, -1.0e8)


class Q3IntervalObjective:
    def __init__(
        self,
        c: Constants,
        target_points: np.ndarray,
        scan_dt_s: float,
    ):
        self.c = c
        self.target_points = target_points
        self.scan_dt_s = scan_dt_s

    def __call__(self, x: np.ndarray) -> float:
        strategy = decode_vector(x)
        if not q3_is_feasible(strategy, self.c):
            return 1_000.0
        intervals = joint_effective_intervals(
            strategy,
            self.c,
            self.target_points,
            scan_dt_s=self.scan_dt_s,
        )
        return -interval_duration(intervals)


def surface_from_settings(settings: dict, c: Constants) -> np.ndarray:
    return target_surface_points(
        int(settings["n_phi"]),
        int(settings["n_z"]),
        int(settings["n_radial"]),
        c,
    )


def optimize_q3(c: Constants, config: dict) -> tuple[Q3Strategy, dict]:
    settings = config["optimization"]
    bounds = [tuple(map(float, pair)) for pair in settings["bounds"]]
    coarse = settings["coarse"]
    coarse_points = surface_from_settings(coarse, c)
    coarse_objective = Q3GridObjective(
        c, coarse_points, float(coarse["scan_dt_s"])
    )
    x0 = np.asarray(settings["warm_start"], dtype=float)

    global_runs = []
    for seed in settings["random_seeds"]:
        result = differential_evolution(
            coarse_objective,
            bounds=bounds,
            seed=int(seed),
            popsize=int(coarse["population_size"]),
            maxiter=int(coarse["max_iterations"]),
            tol=float(coarse["tolerance"]),
            mutation=(0.5, 1.0),
            recombination=0.85,
            polish=False,
            workers=1,
            updating="immediate",
            x0=x0,
        )
        global_runs.append((int(seed), result))

    local = settings["local"]
    local_points = surface_from_settings(local, c)
    local_grid_objective = Q3GridObjective(
        c, local_points, float(local["scan_dt_s"])
    )
    scored = [
        (float(local_grid_objective(result.x)), seed, result)
        for seed, result in global_runs
    ]
    _, selected_seed, selected_global = min(scored, key=lambda item: item[0])

    local_result = minimize(
        local_grid_objective,
        selected_global.x,
        method="Nelder-Mead",
        options={
            "maxiter": int(local["max_iterations"]),
            "xatol": float(local["x_tolerance"]),
            "fatol": float(local["f_tolerance"]),
        },
    )

    interval_settings = settings["interval_refinement"]
    interval_points = surface_from_settings(interval_settings, c)
    interval_objective = Q3IntervalObjective(
        c,
        interval_points,
        float(interval_settings["scan_dt_s"]),
    )
    candidates = [selected_global.x, local_result.x, x0]
    candidate_scores = [float(interval_objective(value)) for value in candidates]
    interval_start = candidates[int(np.argmin(candidate_scores))]
    interval_result = minimize(
        interval_objective,
        interval_start,
        method="Nelder-Mead",
        options={
            "maxiter": int(interval_settings["max_iterations"]),
            "xatol": float(interval_settings["x_tolerance"]),
            "fatol": float(interval_settings["f_tolerance"]),
        },
    )
    final_candidates = candidates + [interval_result.x]
    final_scores = [float(interval_objective(value)) for value in final_candidates]
    best_x = final_candidates[int(np.argmin(final_scores))]

    diagnostics = {
        "random_seeds": settings["random_seeds"],
        "selected_global_seed": selected_seed,
        "global_runs": [
            {
                "seed": seed,
                "success": bool(result.success),
                "message": str(result.message),
                "iterations": int(result.nit),
                "function_evaluations": int(result.nfev),
                "coarse_objective": float(result.fun),
                "local_grid_objective": score,
                "strategy": asdict(decode_vector(result.x)),
            }
            for score, seed, result in scored
        ],
        "local_success": bool(local_result.success),
        "local_message": str(local_result.message),
        "local_iterations": int(local_result.nit),
        "local_function_evaluations": int(local_result.nfev),
        "interval_success": bool(interval_result.success),
        "interval_message": str(interval_result.message),
        "interval_iterations": int(interval_result.nit),
        "interval_function_evaluations": int(interval_result.nfev),
        "interval_objective": float(interval_result.fun),
    }
    return decode_vector(best_x), diagnostics


def individual_bomb_records(
    strategy: Q3Strategy,
    c: Constants,
    verification: dict,
) -> list[dict]:
    records = []
    for index, bomb in enumerate(bomb_strategies(strategy), start=1):
        geometry = strategy_geometry(bomb, c)
        sampled = effective_intervals(
            bomb,
            c,
            n_angle=int(verification["single_cloud_target_angles"]),
            scan_dt_s=float(verification["scan_dt_s"]),
        )
        intervals = refined_interval_boundaries(sampled, bomb, c)
        records.append(
            {
                "smoke_id": index,
                "drop_time_s": bomb.drop_time_s,
                "fuse_delay_s": bomb.fuse_delay_s,
                "burst_time_s": bomb.burst_time_s,
                "drop_pos_m": np.asarray(geometry["drop_pos_m"]).tolist(),
                "burst_pos_m": np.asarray(geometry["burst_pos_m"]).tolist(),
                "individual_full_target_intervals_s": [
                    list(value) for value in intervals
                ],
                "individual_full_target_duration_s": interval_duration(intervals),
            }
        )
    return records


def build_payload(
    strategy: Q3Strategy,
    diagnostics: dict,
    c: Constants,
    config: dict,
) -> dict:
    verification = config["verification"]
    verification_points = surface_from_settings(verification, c)
    joint_intervals = joint_effective_intervals(
        strategy,
        c,
        verification_points,
        scan_dt_s=float(verification["scan_dt_s"]),
    )
    bomb_records = individual_bomb_records(strategy, c, verification)
    individual_intervals = [
        tuple(interval)
        for record in bomb_records
        for interval in record["individual_full_target_intervals_s"]
    ]
    individual_union = merge_intervals(individual_intervals)
    joint_duration = interval_duration(joint_intervals)
    individual_union_duration = interval_duration(individual_union)
    cross_method_difference = joint_duration - individual_union_duration
    return {
        "metadata": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "criterion": (
                "For every target sightline, at least one active cloud must "
                "intersect the sightline before it reaches the target."
            ),
            "random_seeds": config["optimization"]["random_seeds"],
        },
        "strategy": asdict(strategy),
        "bombs": bomb_records,
        "joint_full_target_intervals_s": [
            list(value) for value in joint_intervals
        ],
        "joint_full_target_duration_s": joint_duration,
        "union_of_individually_full_intervals_s": [
            list(value) for value in individual_union
        ],
        "union_of_individually_full_duration_s": individual_union_duration,
        "joint_partial_coverage_gain_s": max(0.0, cross_method_difference),
        "joint_vs_single_method_difference_s": cross_method_difference,
        "constraint_checks": {
            "feasible": q3_is_feasible(strategy, c),
            "shared_heading_and_speed": True,
            "first_gap_s": strategy.drop_times_s[1] - strategy.drop_times_s[0],
            "second_gap_s": strategy.drop_times_s[2] - strategy.drop_times_s[1],
            "gaps_at_least_one_second": (
                strategy.drop_times_s[1] - strategy.drop_times_s[0] >= 1.0
                and strategy.drop_times_s[2] - strategy.drop_times_s[1] >= 1.0
            ),
        },
        "optimization_diagnostics": diagnostics,
    }


def merge_intervals(
    intervals: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged = [ordered[0]]
    for left, right in ordered[1:]:
        old_left, old_right = merged[-1]
        if left <= old_right:
            merged[-1] = (old_left, max(old_right, right))
        else:
            merged.append((left, right))
    return merged


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
                "heading_deg",
                "uav_speed_mps",
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
            ]
        )
        strategy = payload["strategy"]
        for bomb in payload["bombs"]:
            writer.writerow(
                [
                    strategy["heading_deg"],
                    strategy["uav_speed_mps"],
                    bomb["smoke_id"],
                    bomb["drop_time_s"],
                    bomb["fuse_delay_s"],
                    bomb["burst_time_s"],
                    *bomb["drop_pos_m"],
                    *bomb["burst_pos_m"],
                    bomb["individual_full_target_duration_s"],
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--skip-optimize", action="store_true")
    args = parser.parse_args()

    constants, config = load_config(args.config)
    if args.skip_optimize:
        stored = config["stored_strategy"]
        strategy = Q3Strategy(
            heading_deg=float(stored["heading_deg"]),
            uav_speed_mps=float(stored["uav_speed_mps"]),
            drop_times_s=tuple(map(float, stored["drop_times_s"])),
            fuse_delays_s=tuple(map(float, stored["fuse_delays_s"])),
        )
        diagnostics = {"used_stored_strategy": True}
    else:
        strategy, diagnostics = optimize_q3(constants, config)

    payload = build_payload(strategy, diagnostics, constants, config)
    write_outputs(args.output_dir, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
