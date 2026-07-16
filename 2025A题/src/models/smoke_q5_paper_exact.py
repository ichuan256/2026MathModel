"""Faithful Python reproduction of the excellent paper's MATLAB `fifth.m`.

This file intentionally preserves the appendix implementation, including its
FY5 y-coordinate (+2000), nearest-missile assignment, three drops at t/t+1/t+2,
and the final sum of 15 single-cloud durations.  These choices do not equal the
strict simultaneous objective used by the corrected Q5 model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
from pathlib import Path

import numba
import numpy as np
from numba import njit, prange


PROBLEM_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    PROBLEM_ROOT / "outputs" / "model" / "q5_paper_exact" / "results.json"
)
G = 9.8


@njit(cache=True)
def is_overwhelmed(
    missile_pos: np.ndarray,
    smoke_pos: np.ndarray,
    sample_points: np.ndarray,
) -> bool:
    dx = smoke_pos[0] - missile_pos[0]
    dy = smoke_pos[1] - missile_pos[1]
    dz = smoke_pos[2] - missile_pos[2]
    if dx * dx + dy * dy + dz * dz < 100.0:
        return False

    oc0 = missile_pos[0] - smoke_pos[0]
    oc1 = missile_pos[1] - smoke_pos[1]
    oc2 = missile_pos[2] - smoke_pos[2]
    c = oc0 * oc0 + oc1 * oc1 + oc2 * oc2 - 100.0
    for row in range(sample_points.shape[0]):
        d0 = sample_points[row, 0] - missile_pos[0]
        d1 = sample_points[row, 1] - missile_pos[1]
        d2 = sample_points[row, 2] - missile_pos[2]
        a = d0 * d0 + d1 * d1 + d2 * d2
        b = 2.0 * (oc0 * d0 + oc1 * d1 + oc2 * d2)
        discriminant = b * b - 4.0 * a * c
        if discriminant < 0.0:
            return False
        root = math.sqrt(discriminant)
        t1 = (-b - root) / (2.0 * a)
        t2 = (-b + root) / (2.0 * a)
        hit = (
            (0.0 <= t1 <= 1.0)
            or (0.0 <= t2 <= 1.0)
            or (t1 < 0.0 and t2 > 1.0)
        )
        if not hit:
            return False
    return True


@njit(cache=True)
def cloud_blocks_at(
    time_s: float,
    missile_start: np.ndarray,
    missile_unit: np.ndarray,
    burst_pos: np.ndarray,
    burst_time_s: float,
    sample_points: np.ndarray,
) -> bool:
    missile = missile_start + time_s * 300.0 * missile_unit
    cloud = burst_pos.copy()
    cloud[2] -= 3.0 * (time_s - burst_time_s)
    return is_overwhelmed(missile, cloud, sample_points)


@njit(cache=True)
def get_effective_interval(
    missile_start: np.ndarray,
    uav_start: np.ndarray,
    heading_rad: float,
    speed_mps: float,
    fly_time_s: float,
    fuse_delay_s: float,
    sample_points: np.ndarray,
) -> tuple[float, float, float]:
    if speed_mps > 140.0 or speed_mps < 70.0:
        return 0.0, 0.0, 0.0
    if fly_time_s > 65.0 or fly_time_s < 0.0:
        return 0.0, 0.0, 0.0
    if fuse_delay_s > 65.0 or fuse_delay_s < 0.0:
        return 0.0, 0.0, 0.0
    if fly_time_s + fuse_delay_s > 65.0:
        return 0.0, 0.0, 0.0

    norm_m = math.sqrt(
        missile_start[0] ** 2
        + missile_start[1] ** 2
        + missile_start[2] ** 2
    )
    missile_unit = -missile_start / norm_m
    heading = np.asarray(
        [math.cos(heading_rad), math.sin(heading_rad), 0.0]
    )
    burst_time = fly_time_s + fuse_delay_s
    burst = uav_start + burst_time * speed_mps * heading
    burst[2] -= 0.5 * G * fuse_delay_s * fuse_delay_s

    start_time = burst_time
    end_time = start_time + 20.0
    candidates = np.empty(65, dtype=np.float64)
    candidates[0] = start_time
    candidates[1] = 0.5 * (start_time + end_time)
    candidates[2] = end_time
    length = 3
    found = False
    bracket_left = 0.0
    bracket_hit = 0.0
    bracket_right = 0.0

    while not found:
        if length > 32:
            return 0.0, 0.0, 0.0
        index = 1
        while index <= length - 2:
            time_s = candidates[index]
            if cloud_blocks_at(
                time_s,
                missile_start,
                missile_unit,
                burst,
                burst_time,
                sample_points,
            ):
                found = True
                bracket_left = candidates[index - 1]
                bracket_hit = time_s
                bracket_right = candidates[index + 1]
                break
            index += 2
        if not found:
            new_length = 2 * length - 1
            refined = np.empty(65, dtype=np.float64)
            for index in range(length - 1):
                refined[2 * index] = candidates[index]
                refined[2 * index + 1] = 0.5 * (
                    candidates[index] + candidates[index + 1]
                )
            refined[new_length - 1] = candidates[length - 1]
            for index in range(new_length):
                candidates[index] = refined[index]
            length = new_length

    left = bracket_left
    right = bracket_hit
    while right - left > 1.0e-7:
        middle = 0.5 * (left + right)
        if cloud_blocks_at(
            middle,
            missile_start,
            missile_unit,
            burst,
            burst_time,
            sample_points,
        ):
            right = middle
        else:
            left = middle
    interval_start = 0.5 * (left + right)

    left = bracket_hit
    right = bracket_right
    while right - left > 1.0e-7:
        middle = 0.5 * (left + right)
        if cloud_blocks_at(
            middle,
            missile_start,
            missile_unit,
            burst,
            burst_time,
            sample_points,
        ):
            left = middle
        else:
            right = middle
    interval_end = 0.5 * (left + right)
    return interval_start, interval_end, interval_end - interval_start


@njit(cache=True)
def initial_parameters(
    uav_start: np.ndarray,
    missile_start: np.ndarray,
) -> tuple[float, float, float, float]:
    norm_m = math.sqrt(
        missile_start[0] ** 2
        + missile_start[1] ** 2
        + missile_start[2] ** 2
    )
    missile_velocity = -missile_start / norm_m * 300.0
    for step in range(12001):
        time_s = 0.005 * step
        if time_s == 0.0:
            continue
        missile = missile_start + missile_velocity * time_s
        dx = missile[0] - uav_start[0]
        dy = missile[1] - uav_start[1]
        horizontal = math.sqrt(dx * dx + dy * dy)
        speed = horizontal / time_s
        if 70.0 < speed < 140.0 and missile[2] <= uav_start[2]:
            s_square = (uav_start[2] - missile[2]) / (
                0.5 * G * time_s * time_s
            )
            if 0.0 <= s_square <= 1.0:
                s = math.sqrt(s_square)
                heading = math.atan2(dy, dx)
                fly_time = (1.0 - s) * time_s
                fuse_delay = s * time_s
                return heading, speed, fly_time, fuse_delay
    return 0.0, 0.0, 0.0, 0.0


@njit(cache=True)
def grid_search(
    missile_start: np.ndarray,
    uav_start: np.ndarray,
    theta: float,
    speed: float,
    fly_time: float,
    fuse_delay: float,
    sample_points: np.ndarray,
    grid_n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    best_intervals = np.zeros((3, 2), dtype=np.float64)
    best_durations = np.zeros(3, dtype=np.float64)
    for bomb in range(3):
        t1, t2, duration = get_effective_interval(
            missile_start,
            uav_start,
            theta,
            speed,
            fly_time + bomb,
            fuse_delay,
            sample_points,
        )
        best_intervals[bomb, 0] = t1
        best_intervals[bomb, 1] = t2
        best_durations[bomb] = duration
    best_total = best_durations.sum()
    best = np.asarray([theta, speed, fly_time, 0.0])

    for a in range(grid_n):
        d_theta = -0.3 + 0.6 * a / (grid_n - 1)
        for b in range(grid_n):
            d_speed = -20.0 + 40.0 * b / (grid_n - 1)
            for c in range(grid_n):
                d_fly = -6.0 + 12.0 * c / (grid_n - 1)
                for d in range(grid_n):
                    d_fuse = -6.0 + 12.0 * d / (grid_n - 1)
                    new_theta = min(math.pi, max(-math.pi, theta + d_theta))
                    new_speed = min(140.0, max(70.0, speed + d_speed))
                    new_fly = min(65.0, max(0.0, fly_time + d_fly))
                    new_fuse = min(65.0, max(0.0, fuse_delay + d_fuse))
                    intervals = np.zeros((3, 2), dtype=np.float64)
                    durations = np.zeros(3, dtype=np.float64)
                    for bomb in range(3):
                        t1, t2, duration = get_effective_interval(
                            missile_start,
                            uav_start,
                            new_theta,
                            new_speed,
                            new_fly + bomb,
                            new_fuse,
                            sample_points,
                        )
                        intervals[bomb, 0] = t1
                        intervals[bomb, 1] = t2
                        durations[bomb] = duration
                    total = durations.sum()
                    if total > best_total:
                        best_total = total
                        best_intervals = intervals
                        best_durations = durations
                        best = np.asarray(
                            [new_theta, new_speed, new_fly, new_fuse]
                        )
    combined = np.empty(10, dtype=np.float64)
    combined[:4] = best
    combined[4:7] = best_durations
    combined[7] = best_total
    combined[8] = theta
    combined[9] = speed
    return combined, best_intervals, best_durations


@njit(cache=True, parallel=True)
def solve_all(
    missiles: np.ndarray,
    uavs: np.ndarray,
    assignments: np.ndarray,
    sample_points: np.ndarray,
    grid_n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    solutions = np.zeros((5, 10), dtype=np.float64)
    intervals = np.zeros((5, 3, 2), dtype=np.float64)
    durations = np.zeros((5, 3), dtype=np.float64)
    for uav_index in prange(5):
        missile_index = assignments[uav_index]
        theta, speed, fly_time, fuse_delay = initial_parameters(
            uavs[uav_index], missiles[missile_index]
        )
        solution, local_intervals, local_durations = grid_search(
            missiles[missile_index],
            uavs[uav_index],
            theta,
            speed,
            fly_time,
            fuse_delay,
            sample_points,
            grid_n,
        )
        solutions[uav_index] = solution
        intervals[uav_index] = local_intervals
        durations[uav_index] = local_durations
    return solutions, intervals, durations


def target_points() -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, 300)
    rows = []
    for height in (0.0, 10.0):
        rows.extend(
            [7.0 * np.cos(angle), 200.0 + 7.0 * np.sin(angle), height]
            for angle in angles
        )
    return np.asarray(rows, dtype=np.float64)


def nearest_missile_assignments(
    missiles: np.ndarray,
    uavs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    directions = -missiles / np.linalg.norm(missiles, axis=1)[:, None]
    distance = np.zeros((3, 5), dtype=float)
    for uav_index in range(5):
        for missile_index in range(3):
            time_s = (
                missiles[missile_index, 2] - uavs[uav_index, 2]
            ) / (-directions[missile_index, 2]) / 300.0
            missile = (
                missiles[missile_index]
                + directions[missile_index] * time_s * 300.0
            )
            distance[missile_index, uav_index] = np.linalg.norm(
                uavs[uav_index, :2] - missile[:2]
            )
    return np.argmin(distance, axis=0).astype(np.int64), distance


def strategy_rows(
    solutions: np.ndarray,
    intervals: np.ndarray,
    durations: np.ndarray,
    assignments: np.ndarray,
    uavs: np.ndarray,
) -> list[dict]:
    rows = []
    for uav_index in range(5):
        theta, speed, fly_time, fuse_delay = solutions[uav_index, :4]
        direction = np.asarray([math.cos(theta), math.sin(theta), 0.0])
        for bomb in range(3):
            actual_drop_time = fly_time + bomb
            drop = uavs[uav_index] + actual_drop_time * speed * direction
            burst = drop + fuse_delay * speed * direction
            burst[2] -= 0.5 * G * fuse_delay**2
            rows.append(
                {
                    "uav_id": f"FY{uav_index + 1}",
                    "smoke_id": bomb + 1,
                    "assigned_missile": f"M{assignments[uav_index] + 1}",
                    "heading_rad": float(theta),
                    "heading_deg": float(math.degrees(theta) % 360.0),
                    "speed_mps": float(speed),
                    "drop_time_s": float(actual_drop_time),
                    "fuse_delay_s": float(fuse_delay),
                    "burst_time_s": float(actual_drop_time + fuse_delay),
                    "drop_pos_m": drop.tolist(),
                    "burst_pos_m": burst.tolist(),
                    "paper_interval_s": intervals[uav_index, bomb].tolist(),
                    "paper_individual_duration_s": float(
                        durations[uav_index, bomb]
                    ),
                }
            )
    return rows


def merge_intervals(intervals: list[list[float]]) -> list[list[float]]:
    merged: list[list[float]] = []
    for left, right in sorted(intervals):
        if right <= left:
            continue
        if not merged or left > merged[-1][1]:
            merged.append([float(left), float(right)])
        else:
            merged[-1][1] = max(merged[-1][1], float(right))
    return merged


def intersection_measure(groups: list[list[list[float]]]) -> float:
    if any(not group for group in groups):
        return 0.0
    current = groups[0]
    for group in groups[1:]:
        overlap = []
        for left_a, right_a in current:
            for left_b, right_b in group:
                left = max(left_a, left_b)
                right = min(right_a, right_b)
                if right > left:
                    overlap.append([left, right])
        current = merge_intervals(overlap)
        if not current:
            return 0.0
    return float(sum(right - left for left, right in current))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-n", type=int, default=20)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.grid_n < 2:
        raise ValueError("grid-n must be at least 2")

    missiles = np.asarray(
        [[20000.0, 0.0, 2000.0], [19000.0, 600.0, 2100.0], [18000.0, -600.0, 1900.0]]
    )
    # Faithful appendix value: FY5 y is +2000, although the problem says -2000.
    uavs = np.asarray(
        [
            [17800.0, 0.0, 1800.0],
            [12000.0, 1400.0, 1400.0],
            [6000.0, -3000.0, 700.0],
            [11000.0, 2000.0, 1800.0],
            [13000.0, 2000.0, 1300.0],
        ]
    )
    assignments, distance = nearest_missile_assignments(missiles, uavs)
    solutions, intervals, durations = solve_all(
        missiles, uavs, assignments, target_points(), args.grid_n
    )
    rows = strategy_rows(solutions, intervals, durations, assignments, uavs)
    paper_total = float(durations.sum())
    assigned_unions = {
        missile_id: merge_intervals(
            [
                row["paper_interval_s"]
                for row in rows
                if row["assigned_missile"] == missile_id
            ]
        )
        for missile_id in ("M1", "M2", "M3")
    }
    paper_text_intersection = intersection_measure(
        [assigned_unions[missile_id] for missile_id in ("M1", "M2", "M3")]
    )
    payload = {
        "metadata": {
            "method": "faithful reproduction of appendix fifth.m",
            "python": platform.python_version(),
            "numpy": np.__version__,
            "numba": numba.__version__,
            "grid_n_per_dimension": args.grid_n,
            "grid_candidate_count_per_uav": args.grid_n**4,
            "sample_point_count": 600,
            "known_appendix_inconsistencies": [
                "FY5 y-coordinate is +2000 m instead of the problem's -2000 m.",
                "The objective sums 15 individual cloud durations instead of the simultaneous three-missile intersection.",
                "The optimizer uses drop times fly_t, fly_t+1, fly_t+2; the MATLAB export later reuses durations as offsets.",
            ],
        },
        "nearest_missile_assignment": {
            f"FY{index + 1}": f"M{int(assignments[index]) + 1}"
            for index in range(5)
        },
        "distance_matrix_m": distance.tolist(),
        "paper_code_total_s": paper_total,
        "paper_reported_s": 21.0770,
        "absolute_difference_from_reported_s": abs(paper_total - 21.0770),
        "assigned_missile_interval_unions_s": assigned_unions,
        "paper_text_three_missile_intersection_s": paper_text_intersection,
        "strategy": rows,
        "per_uav_sum_s": {
            f"FY{index + 1}": float(durations[index].sum())
            for index in range(5)
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    csv_path = args.output.with_name("strategy_summary.csv")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "uav_id", "smoke_id", "assigned_missile", "heading_deg",
                "speed_mps", "drop_time_s", "fuse_delay_s", "burst_time_s",
                "paper_interval_start_s", "paper_interval_end_s",
                "paper_individual_duration_s",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["uav_id"], row["smoke_id"], row["assigned_missile"],
                    row["heading_deg"], row["speed_mps"], row["drop_time_s"],
                    row["fuse_delay_s"], row["burst_time_s"],
                    row["paper_interval_s"][0], row["paper_interval_s"][1],
                    row["paper_individual_duration_s"],
                ]
            )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
