"""2025 CUMCM A, question 4: three UAVs, one smoke bomb each, against M1.

The code uses the strict joint criterion: for every sampled point of the true
target cylinder, its sightline from M1 must meet at least one active smoke
cloud before reaching that point.  A union of three cloud shadows is allowed.
"""

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
    interval_duration,
    load_config as load_q1_q2_config,
    missile_arrival_s,
    missile_position,
    unit_heading,
)
from smoke_q3 import target_surface_points  # noqa: E402


DEFAULT_CONFIG = PROBLEM_ROOT / "configs" / "q4.json"
DEFAULT_OUTPUT_DIR = PROBLEM_ROOT / "outputs" / "model" / "q4"
UAV_IDS = ("FY1", "FY2", "FY3")


@dataclass(frozen=True)
class Q4Bomb:
    uav_id: str
    heading_deg: float
    speed_mps: float
    drop_time_s: float
    fuse_delay_s: float

    @property
    def burst_time_s(self) -> float:
        return self.drop_time_s + self.fuse_delay_s


@dataclass(frozen=True)
class Q4Strategy:
    bombs: tuple[Q4Bomb, Q4Bomb, Q4Bomb]


def load_config(path: Path) -> tuple[Constants, dict, dict[str, np.ndarray]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    constants, _ = load_q1_q2_config(PROBLEM_ROOT / "configs" / "q1_q2.json")
    uavs = {
        uav_id: np.asarray(data["uavs"][uav_id], dtype=float)
        for uav_id in UAV_IDS
    }
    return constants, data, uavs


def decode_vector(x: np.ndarray) -> Q4Strategy:
    bombs = []
    for index, uav_id in enumerate(UAV_IDS):
        offset = 4 * index
        bombs.append(
            Q4Bomb(
                uav_id=uav_id,
                heading_deg=float(x[offset] % 360.0),
                speed_mps=float(x[offset + 1]),
                drop_time_s=float(x[offset + 2]),
                fuse_delay_s=float(x[offset + 3]),
            )
        )
    return Q4Strategy(tuple(bombs))


def encode_strategy(strategy: Q4Strategy) -> np.ndarray:
    values = []
    for bomb in strategy.bombs:
        values.extend(
            [
                bomb.heading_deg,
                bomb.speed_mps,
                bomb.drop_time_s,
                bomb.fuse_delay_s,
            ]
        )
    return np.asarray(values, dtype=float)


def bomb_geometry(bomb: Q4Bomb, uav_start_m: np.ndarray, c: Constants) -> dict:
    direction = unit_heading(bomb.heading_deg)
    drop = uav_start_m + bomb.speed_mps * bomb.drop_time_s * direction
    burst = (
        drop
        + bomb.speed_mps * bomb.fuse_delay_s * direction
        - np.array(
            [0.0, 0.0, 0.5 * c.gravity_mps2 * bomb.fuse_delay_s**2],
            dtype=float,
        )
    )
    return {
        "drop_pos_m": drop,
        "burst_pos_m": burst,
        "burst_time_s": bomb.burst_time_s,
    }


def strategy_is_feasible(
    strategy: Q4Strategy,
    c: Constants,
    uavs: dict[str, np.ndarray],
) -> bool:
    arrival = missile_arrival_s(c)
    for bomb in strategy.bombs:
        if bomb.uav_id not in uavs:
            return False
        if not 0.0 <= bomb.heading_deg < 360.0:
            return False
        if not 70.0 <= bomb.speed_mps <= 140.0:
            return False
        if bomb.drop_time_s < 0.0 or bomb.fuse_delay_s < 0.0:
            return False
        geometry = bomb_geometry(bomb, uavs[bomb.uav_id], c)
        if bomb.burst_time_s > arrival:
            return False
        if float(geometry["burst_pos_m"][2]) < 0.0:
            return False
    return True


def cloud_centers(
    times: np.ndarray,
    bomb: Q4Bomb,
    uav_start_m: np.ndarray,
    c: Constants,
) -> np.ndarray:
    burst = np.asarray(bomb_geometry(bomb, uav_start_m, c)["burst_pos_m"])
    centers = np.broadcast_to(burst, times.shape + (3,)).copy()
    centers[:, 2] -= c.cloud_sink_speed_mps * (times - bomb.burst_time_s)
    return centers


def joint_coverage_series(
    times: np.ndarray,
    strategy: Q4Strategy,
    c: Constants,
    uavs: dict[str, np.ndarray],
    target_points: np.ndarray,
    *,
    chunk_size: int = 160,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return full flags, fraction covered, and a continuous search guide."""
    times = np.asarray(times, dtype=float)
    full_result = np.zeros(len(times), dtype=bool)
    fraction_result = np.zeros(len(times), dtype=float)
    guide_result = np.full(len(times), -1.0e12, dtype=float)
    radius2 = c.cloud_radius_m**2

    for start in range(0, len(times), chunk_size):
        end = min(start + chunk_size, len(times))
        t = times[start:end]
        missile = missile_position(t, c)
        ray = target_points[None, :, :] - missile[:, None, :]
        ray_length = np.linalg.norm(ray, axis=2)
        ray_unit = ray / ray_length[:, :, None]
        blocked_by_any = np.zeros(ray_length.shape, dtype=bool)
        best_margin = np.full(ray_length.shape, -1.0e12, dtype=float)

        for bomb in strategy.bombs:
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
            root = np.sqrt(np.maximum(discriminant, 0.0))
            entry = projection - root
            blocked = (
                (discriminant >= 0.0)
                & (entry >= 0.0)
                & (entry <= ray_length)
                & active[:, None]
            )
            blocked_by_any |= blocked
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


def coverage_at_time(
    t: float,
    strategy: Q4Strategy,
    c: Constants,
    uavs: dict[str, np.ndarray],
    target_points: np.ndarray,
) -> bool:
    flags, _, _ = joint_coverage_series(
        np.array([t]), strategy, c, uavs, target_points, chunk_size=1
    )
    return bool(flags[0])


def effective_intervals(
    strategy: Q4Strategy,
    c: Constants,
    uavs: dict[str, np.ndarray],
    target_points: np.ndarray,
    *,
    scan_dt_s: float,
) -> list[tuple[float, float]]:
    if not strategy_is_feasible(strategy, c, uavs):
        return []
    bursts = [bomb.burst_time_s for bomb in strategy.bombs]
    start = min(bursts)
    end = min(max(value + c.cloud_lifetime_s for value in bursts), missile_arrival_s(c))
    times = np.linspace(start, end, max(2, int(math.ceil((end - start) / scan_dt_s)) + 1))
    full, _, _ = joint_coverage_series(times, strategy, c, uavs, target_points)

    intervals: list[tuple[float, float]] = []
    current_start = start if full[0] else None
    for index in range(1, len(times)):
        if full[index] == full[index - 1]:
            continue
        left, right = float(times[index - 1]), float(times[index])
        left_value = bool(full[index - 1])
        for _ in range(48):
            middle = 0.5 * (left + right)
            if coverage_at_time(middle, strategy, c, uavs, target_points) == left_value:
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


class GridObjective:
    def __init__(self, c: Constants, uavs: dict[str, np.ndarray], points: np.ndarray, dt: float):
        self.c, self.uavs, self.points, self.dt = c, uavs, points, dt

    def __call__(self, x: np.ndarray) -> float:
        strategy = decode_vector(x)
        if not strategy_is_feasible(strategy, self.c, self.uavs):
            return 1_000.0
        bursts = [bomb.burst_time_s for bomb in strategy.bombs]
        start, end = min(bursts), min(max(value + self.c.cloud_lifetime_s for value in bursts), missile_arrival_s(self.c))
        times = np.linspace(start, end, max(2, int(math.ceil((end - start) / self.dt)) + 1))
        full, fraction, guide = joint_coverage_series(times, strategy, self.c, self.uavs, self.points)
        strict_duration = float(np.trapezoid(full.astype(float), times))
        coverage_area = float(np.trapezoid(fraction, times))
        if np.any(full):
            return -100.0 - strict_duration - 1.0e-4 * coverage_area
        return -coverage_area - 1.0e-8 * max(float(np.max(guide)), -1.0e8)


class IntervalObjective:
    def __init__(self, c: Constants, uavs: dict[str, np.ndarray], points: np.ndarray, dt: float):
        self.c, self.uavs, self.points, self.dt = c, uavs, points, dt

    def __call__(self, x: np.ndarray) -> float:
        strategy = decode_vector(x)
        return -interval_duration(effective_intervals(strategy, self.c, self.uavs, self.points, scan_dt_s=self.dt))


class SingleUAVBaselineObjective:
    """Baseline: optimize one UAV in isolation before combining the three."""

    def __init__(self, uav_id: str, c: Constants, uavs: dict[str, np.ndarray], points: np.ndarray, dt: float):
        self.uav_id, self.c, self.uavs, self.points, self.dt = uav_id, c, uavs, points, dt

    def __call__(self, x: np.ndarray) -> float:
        bomb = Q4Bomb(self.uav_id, float(x[0] % 360.0), float(x[1]), float(x[2]), float(x[3]))
        # Repeating the same cloud three times leaves the union unchanged;
        # it allows the common strict coverage evaluator to be reused.
        strategy = Q4Strategy((bomb, bomb, bomb))
        if not strategy_is_feasible(strategy, self.c, self.uavs):
            return 1_000.0
        start = bomb.burst_time_s
        end = min(start + self.c.cloud_lifetime_s, missile_arrival_s(self.c))
        times = np.linspace(start, end, max(2, int(math.ceil((end - start) / self.dt)) + 1))
        full, fraction, guide = joint_coverage_series(times, strategy, self.c, self.uavs, self.points)
        duration = float(np.trapezoid(full.astype(float), times))
        area = float(np.trapezoid(fraction, times))
        if np.any(full):
            return -100.0 - duration - 1.0e-4 * area
        return -area - 1.0e-8 * max(float(np.max(guide)), -1.0e8)


def bounds_from_config(config: dict) -> list[tuple[float, float]]:
    bound = config["optimization"]["bounds_per_uav"]
    one = [
        tuple(map(float, bound["heading_deg"])),
        tuple(map(float, bound["speed_mps"])),
        tuple(map(float, bound["drop_time_s"])),
        tuple(map(float, bound["fuse_delay_s"])),
    ]
    return one * 3


def points_from_settings(settings: dict, c: Constants) -> np.ndarray:
    return target_surface_points(
        int(settings["n_phi"]), int(settings["n_z"]), int(settings["n_radial"]), c
    )


def optimize(c: Constants, config: dict, uavs: dict[str, np.ndarray]) -> tuple[Q4Strategy, dict]:
    settings = config["optimization"]
    bounds = bounds_from_config(config)
    coarse = settings["coarse"]
    coarse_objective = GridObjective(c, uavs, points_from_settings(coarse, c), float(coarse["scan_dt_s"]))
    single_bounds = bounds[:4]
    baseline_bombs = []
    baseline_diagnostics = []
    for offset, uav_id in enumerate(UAV_IDS):
        baseline_objective = SingleUAVBaselineObjective(
            uav_id, c, uavs, points_from_settings(coarse, c), float(coarse["scan_dt_s"])
        )
        baseline_result = differential_evolution(
            baseline_objective,
            bounds=single_bounds,
            seed=int(settings["random_seeds"][offset % len(settings["random_seeds"])]),
            popsize=int(settings["baseline"]["population_size"]),
            maxiter=int(settings["baseline"]["max_iterations"]),
            polish=False,
            workers=1,
            updating="immediate",
        )
        baseline_bombs.append(
            Q4Bomb(uav_id, *map(float, baseline_result.x))
        )
        baseline_diagnostics.append({
            "uav_id": uav_id,
            "success": bool(baseline_result.success),
            "iterations": int(baseline_result.nit),
            "function_evaluations": int(baseline_result.nfev),
            "objective": float(baseline_result.fun),
        })
    baseline_strategy = Q4Strategy(tuple(baseline_bombs))
    baseline_x = encode_strategy(baseline_strategy)
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
            x0=baseline_x,
        )
        global_runs.append((int(seed), result))

    local = settings["local"]
    local_objective = GridObjective(c, uavs, points_from_settings(local, c), float(local["scan_dt_s"]))
    scored = [(float(local_objective(result.x)), seed, result) for seed, result in global_runs]
    _, selected_seed, selected = min(scored, key=lambda value: value[0])
    local_result = minimize(
        local_objective,
        selected.x,
        method="Nelder-Mead",
        options={
            "maxiter": int(local["max_iterations"]),
            "xatol": float(local["x_tolerance"]),
            "fatol": float(local["f_tolerance"]),
        },
    )
    candidate_x = local_result.x if local_result.fun < local_objective(selected.x) else selected.x
    return decode_vector(candidate_x), {
        "random_seeds": settings["random_seeds"],
        "selected_global_seed": selected_seed,
        "baseline_strategy": asdict(baseline_strategy),
        "baseline_runs": baseline_diagnostics,
        "baseline_coarse_objective": float(coarse_objective(baseline_x)),
        "global_runs": [
            {"seed": seed, "success": bool(result.success), "iterations": int(result.nit), "function_evaluations": int(result.nfev), "objective": float(result.fun)}
            for seed, result in global_runs
        ],
        "local_success": bool(local_result.success),
        "local_message": str(local_result.message),
        "local_iterations": int(local_result.nit),
    }


def build_payload(strategy: Q4Strategy, diagnostics: dict, c: Constants, config: dict, uavs: dict[str, np.ndarray]) -> dict:
    verification = config["verification"]
    points = points_from_settings(verification, c)
    intervals = effective_intervals(strategy, c, uavs, points, scan_dt_s=float(verification["scan_dt_s"]))
    bombs = []
    for bomb in strategy.bombs:
        geometry = bomb_geometry(bomb, uavs[bomb.uav_id], c)
        bombs.append({**asdict(bomb), "burst_time_s": bomb.burst_time_s, "drop_pos_m": geometry["drop_pos_m"].tolist(), "burst_pos_m": geometry["burst_pos_m"].tolist()})
    return {
        "metadata": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "criterion": "Every target sightline is blocked by at least one active cloud before the target.", "random_seeds": config["optimization"]["random_seeds"]},
        "strategy": {"bombs": bombs},
        "joint_full_target_intervals_s": [list(item) for item in intervals],
        "joint_full_target_duration_s": interval_duration(intervals),
        "constraint_checks": {"feasible": strategy_is_feasible(strategy, c, uavs), "one_bomb_per_uav": len(strategy.bombs) == 3, "uav_ids": [bomb.uav_id for bomb in strategy.bombs]},
        "optimization_diagnostics": diagnostics,
    }


def write_outputs(output_dir: Path, payload: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "strategy_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["uav_id", "heading_deg", "speed_mps", "drop_time_s", "fuse_delay_s", "burst_time_s", "drop_x_m", "drop_y_m", "drop_z_m", "burst_x_m", "burst_y_m", "burst_z_m"])
        for bomb in payload["strategy"]["bombs"]:
            writer.writerow([bomb["uav_id"], bomb["heading_deg"], bomb["speed_mps"], bomb["drop_time_s"], bomb["fuse_delay_s"], bomb["burst_time_s"], *bomb["drop_pos_m"], *bomb["burst_pos_m"]])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    constants, config, uavs = load_config(args.config)
    strategy, diagnostics = optimize(constants, config, uavs)
    payload = build_payload(strategy, diagnostics, constants, config, uavs)
    write_outputs(args.output_dir, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
