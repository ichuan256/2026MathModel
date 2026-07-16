"""Question 5 solver inspired by the displayed excellent paper.

The solver keeps the paper's hierarchy -- 15 single-UAV/single-missile
subproblems followed by a UAV assignment -- but adds two repairs:

1. assignment scores are obtained from the physical full-target evaluator,
   not only from a straight-line flight-time proxy;
2. the three missile subplans are synchronized and then re-evaluated on one
   shared time axis with every physical cloud visible to every missile.

The final reported value is therefore the strict simultaneous concealment
duration, never the sum of three separate durations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, minimize

from smoke_q5 import (
    MISSILE_IDS,
    PROBLEM_ROOT,
    UAV_IDS,
    VARIABLES_PER_UAV,
    GridObjective,
    build_payload,
    decode_vector,
    effective_intervals_for_missile,
    encode_strategy,
    geometric_baseline,
    load_config,
    make_bounds,
    paper_assignment_warm_start,
    points_from_settings,
    strategy_from_results,
    strategy_is_feasible,
    write_outputs,
)


DEFAULT_CONFIG = PROBLEM_ROOT / "configs" / "q5_paper_new.json"
DEFAULT_OUTPUT = PROBLEM_ROOT / "outputs" / "model" / "q5_paper_new"


def longest_interval_center(intervals: list[tuple[float, float]]) -> float | None:
    if not intervals:
        return None
    left, right = max(intervals, key=lambda value: value[1] - value[0])
    return 0.5 * (left + right)


def synchronize_assignment_seeds(
    vector: np.ndarray,
    mapping: dict[str, str],
    constants,
    config: dict,
    uavs: dict[str, np.ndarray],
    missiles: dict[str, np.ndarray],
    bounds: list[tuple[float, float]],
) -> list[np.ndarray]:
    """Shift assigned UAV groups toward common interval centers."""
    settings = config["optimization"]["local"]
    points = points_from_settings(settings, constants)
    dt = float(settings["scan_dt_s"])
    strategy = decode_vector(vector)
    centers: dict[str, float | None] = {}
    for missile_id in MISSILE_IDS:
        assigned = {
            uav_id for uav_id, target_id in mapping.items()
            if target_id == missile_id
        }
        bombs = [
            bomb for bomb in strategy.active_bombs if bomb.uav_id in assigned
        ]
        intervals = effective_intervals_for_missile(
            bombs,
            missiles[missile_id],
            constants,
            uavs,
            points,
            dt,
        )
        centers[missile_id] = longest_interval_center(intervals)

    finite_centers = [value for value in centers.values() if value is not None]
    target_centers = [12.0, 16.0, 20.0, 24.0, 28.0, 32.0]
    if finite_centers:
        target_centers.insert(0, float(np.median(finite_centers)))

    lower = np.asarray([item[0] for item in bounds], dtype=float)
    upper = np.asarray([item[1] for item in bounds], dtype=float)
    seeds = [np.asarray(vector, dtype=float)]
    for target_center in target_centers:
        shifted = np.asarray(vector, dtype=float).copy()
        for uav_index, uav_id in enumerate(UAV_IDS):
            center = centers[mapping[uav_id]]
            if center is None:
                continue
            first_drop_index = uav_index * VARIABLES_PER_UAV + 2
            shifted[first_drop_index] += target_center - center
        seeds.append(np.clip(shifted, lower, upper))
    return seeds


def timing_indices() -> np.ndarray:
    return np.asarray(
        [
            uav_index * VARIABLES_PER_UAV + offset
            for uav_index in range(len(UAV_IDS))
            for offset in range(2, VARIABLES_PER_UAV)
        ],
        dtype=int,
    )


def optimize_timing(
    base_vectors: list[np.ndarray],
    constants,
    config: dict,
    uavs: dict[str, np.ndarray],
    missiles: dict[str, np.ndarray],
    bounds: list[tuple[float, float]],
) -> tuple[np.ndarray, list[dict]]:
    """Optimize 30 timing variables while preserving assigned headings/speeds."""
    timing = config["optimization"]["timing_search"]
    objective = GridObjective(
        constants,
        config,
        uavs,
        missiles,
        config["optimization"]["coarse"],
    )
    indices = timing_indices()
    timing_bounds = [bounds[index] for index in indices]
    low = np.asarray([item[0] for item in timing_bounds], dtype=float)
    high = np.asarray([item[1] for item in timing_bounds], dtype=float)

    base_scores = [(float(objective(value)), value) for value in base_vectors]
    _, best_base = min(base_scores, key=lambda item: item[0])
    runs = []
    candidates = list(base_vectors)
    for seed in timing["random_seeds"]:
        rng = np.random.default_rng(int(seed))
        population_count = max(
            5, int(timing["population_size"]) * len(indices)
        )
        population = rng.uniform(
            low, high, size=(population_count, len(indices))
        )
        seed_rows = min(len(base_vectors), population_count)
        for row in range(seed_rows):
            population[row] = base_vectors[row][indices]
        remaining = population_count - seed_rows
        tight_count = remaining // 3
        medium_count = remaining // 3
        tight_scale = np.tile(
            np.asarray([0.4, 0.6, 0.6, 0.4, 0.4, 0.4]), len(UAV_IDS)
        )
        medium_scale = np.tile(
            np.asarray([2.0, 3.0, 3.0, 1.8, 1.8, 1.8]), len(UAV_IDS)
        )
        start = seed_rows
        if tight_count:
            population[start:start + tight_count] = (
                best_base[indices]
                + rng.normal(0.0, tight_scale, size=(tight_count, len(indices)))
            )
        start += tight_count
        if medium_count:
            population[start:start + medium_count] = (
                best_base[indices]
                + rng.normal(0.0, medium_scale, size=(medium_count, len(indices)))
            )
        population = np.clip(population, low, high)

        def timing_objective(values: np.ndarray) -> float:
            full = best_base.copy()
            full[indices] = values
            return float(objective(full))

        progress = {"generation": 0}

        def callback(values: np.ndarray, convergence: float) -> bool:
            progress["generation"] += 1
            generation = progress["generation"]
            if generation == 1 or generation % 25 == 0:
                print(
                    f"timing seed={seed} generation={generation} "
                    f"objective={timing_objective(values):.9f} "
                    f"convergence={convergence:.6g}",
                    flush=True,
                )
            return False

        result = differential_evolution(
            timing_objective,
            bounds=timing_bounds,
            init=population,
            x0=best_base[indices],
            seed=int(seed),
            popsize=int(timing["population_size"]),
            maxiter=int(timing["max_iterations"]),
            tol=float(timing["tolerance"]),
            atol=float(timing["absolute_tolerance"]),
            mutation=(0.5, 1.0),
            recombination=0.85,
            polish=False,
            workers=1,
            updating="immediate",
            callback=callback,
        )
        full = best_base.copy()
        full[indices] = result.x
        candidates.append(full)
        runs.append(
            {
                "seed": int(seed),
                "success": bool(result.success),
                "message": str(result.message),
                "iterations": int(result.nit),
                "function_evaluations": int(result.nfev),
                "coarse_objective": float(result.fun),
            }
        )

    local_objective = GridObjective(
        constants,
        config,
        uavs,
        missiles,
        config["optimization"]["local"],
    )
    rescored = [(float(local_objective(value)), value) for value in candidates]
    _, selected = min(rescored, key=lambda item: item[0])
    return selected, runs


def solve(
    config_path: Path,
    *,
    smoke_test: bool = False,
) -> tuple[object, dict, dict, dict, object]:
    constants, config, uavs, missiles = load_config(config_path)
    if smoke_test:
        paper = config["optimization"]["paper_decomposition"]
        paper["max_iterations"] = 1
        paper["local_max_iterations"] = 0
        config["optimization"]["timing_search"]["max_iterations"] = 1
        config["optimization"]["joint_local"]["max_iterations"] = 0
        config["verification"] = {
            "n_phi": 24,
            "n_z": 5,
            "n_radial": 3,
            "scan_dt_s": 0.03,
        }
    bounds = make_bounds(config, uavs, constants)

    fallback = geometric_baseline(uavs, constants)
    fallback_x = encode_strategy(fallback)
    warm_vectors = [fallback_x]
    for configured_path in config["optimization"].get("warm_start_results", []):
        path = Path(configured_path)
        if not path.is_absolute():
            path = PROBLEM_ROOT / path
        strategy = strategy_from_results(path)
        if not strategy_is_feasible(strategy, constants, uavs, missiles):
            raise ValueError(f"Infeasible warm start: {path}")
        warm_vectors.append(encode_strategy(strategy))

    paper_base, assignment_vectors, paper_diagnostics = (
        paper_assignment_warm_start(
            constants,
            config,
            uavs,
            missiles,
            warm_vectors[-1],
        )
    )
    mapping = paper_diagnostics["strict_joint_best_assignment"]["mapping"]
    synchronized = synchronize_assignment_seeds(
        paper_base,
        mapping,
        constants,
        config,
        uavs,
        missiles,
        bounds,
    )
    base_vectors = [*warm_vectors, paper_base, *assignment_vectors, *synchronized]
    timing_best, timing_runs = optimize_timing(
        base_vectors,
        constants,
        config,
        uavs,
        missiles,
        bounds,
    )

    local_objective = GridObjective(
        constants,
        config,
        uavs,
        missiles,
        config["optimization"]["local"],
    )
    local_cfg = config["optimization"]["joint_local"]
    local_result = None
    if int(local_cfg["max_iterations"]) > 0:
        local_result = minimize(
            local_objective,
            timing_best,
            method="Powell",
            bounds=bounds,
            options={
                "maxiter": int(local_cfg["max_iterations"]),
                "xtol": float(local_cfg["x_tolerance"]),
                "ftol": float(local_cfg["f_tolerance"]),
            },
        )
    final_candidates = [*warm_vectors, paper_base, timing_best]
    if local_result is not None:
        final_candidates.append(local_result.x)
    scores = [float(local_objective(value)) for value in final_candidates]
    best_x = final_candidates[int(np.argmin(scores))]
    diagnostics = {
        "method": "paper-inspired assignment, interval synchronization, strict joint refinement",
        "paper_decomposition": paper_diagnostics,
        "timing_search_runs": timing_runs,
        "joint_local_success": (
            None if local_result is None else bool(local_result.success)
        ),
        "joint_local_message": (
            "skipped: discontinuous objective and explicit evaluation budget"
            if local_result is None
            else str(local_result.message)
        ),
        "joint_local_iterations": (
            0 if local_result is None else int(local_result.nit)
        ),
        "joint_local_function_evaluations": (
            0 if local_result is None else int(local_result.nfev)
        ),
        "candidate_local_grid_objectives": scores,
        "selected_local_grid_objective": float(min(scores)),
    }
    return decode_vector(best_x), diagnostics, config, uavs, missiles, constants


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    strategy, diagnostics, config, uavs, missiles, constants = solve(
        args.config, smoke_test=args.smoke_test
    )
    payload = build_payload(
        strategy, diagnostics, constants, config, uavs, missiles
    )
    write_outputs(args.output_dir, payload)
    print(
        json.dumps(
            {
                "simultaneous_full_target_duration_s": payload[
                    "simultaneous_full_target_duration_s"
                ],
                "simultaneous_full_target_intervals_s": payload[
                    "simultaneous_full_target_intervals_s"
                ],
                "constraint_checks": payload["constraint_checks"],
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
