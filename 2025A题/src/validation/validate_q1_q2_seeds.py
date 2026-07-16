"""Check whether independent optimizer seeds reach the same Q2 solution region."""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, minimize


PROBLEM_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROBLEM_ROOT / "src" / "models"))

from smoke_q1_q2 import (  # noqa: E402
    DEFAULT_CONFIG,
    Q2IntervalDurationObjective,
    Q2Objective,
    Strategy,
    effective_intervals,
    interval_duration,
    load_config,
    missile_arrival_s,
    refined_interval_boundaries,
    strategy_is_feasible,
)


def main() -> None:
    constants, _ = load_config(DEFAULT_CONFIG)
    arrival = missile_arrival_s(constants)
    max_fuse = math.sqrt(
        2.0 * constants.uav_pos0_m[2] / constants.gravity_mps2
    )
    bounds = [
        (0.0, 360.0),
        (70.0, 140.0),
        (0.0, arrival),
        (0.0, max_fuse),
    ]
    rows = []
    for seed in (2025, 2026, 2027):
        coarse_objective = Q2Objective(constants, n_angle=36, scan_dt_s=0.075)
        global_result = differential_evolution(
            coarse_objective,
            bounds=bounds,
            seed=seed,
            popsize=12,
            maxiter=100,
            tol=0.0005,
            mutation=(0.5, 1.0),
            recombination=0.8,
            polish=False,
            workers=1,
            updating="immediate",
        )

        interval_objective = Q2IntervalDurationObjective(
            constants, n_angle=90, scan_dt_s=0.02
        )
        interval_result = minimize(
            interval_objective,
            global_result.x,
            method="Nelder-Mead",
            options={
                "maxiter": 700,
                "xatol": 1.0e-8,
                "fatol": 1.0e-9,
            },
        )
        x = interval_result.x
        strategy = Strategy(
            heading_deg=float(x[0] % 360.0),
            uav_speed_mps=float(x[1]),
            burst_time_s=float(x[2]),
            fuse_delay_s=float(x[3]),
        )

        boundary_used = False
        if strategy.uav_speed_mps <= 70.5 and strategy.drop_time_s <= 0.2:
            boundary_used = True

            def boundary_objective(y: np.ndarray) -> float:
                candidate = Strategy(
                    heading_deg=float(y[0] % 360.0),
                    uav_speed_mps=70.0,
                    burst_time_s=float(y[1]),
                    fuse_delay_s=float(y[1]),
                )
                if not strategy_is_feasible(candidate, constants):
                    return 1_000.0
                return -interval_duration(
                    effective_intervals(
                        candidate,
                        constants,
                        n_angle=180,
                        scan_dt_s=0.01,
                    )
                )

            boundary_result = minimize(
                boundary_objective,
                np.array([strategy.heading_deg, strategy.fuse_delay_s]),
                method="Nelder-Mead",
                options={
                    "maxiter": 350,
                    "xatol": 1.0e-9,
                    "fatol": 1.0e-10,
                },
            )
            strategy = Strategy(
                heading_deg=float(boundary_result.x[0] % 360.0),
                uav_speed_mps=70.0,
                burst_time_s=float(boundary_result.x[1]),
                fuse_delay_s=float(boundary_result.x[1]),
            )

        sampled = effective_intervals(
            strategy,
            constants,
            n_angle=720,
            scan_dt_s=0.004,
        )
        intervals = refined_interval_boundaries(sampled, strategy, constants)
        duration = interval_duration(intervals)
        rows.append(
            [
                seed,
                strategy.heading_deg,
                strategy.uav_speed_mps,
                strategy.drop_time_s,
                strategy.fuse_delay_s,
                duration,
                global_result.nit,
                global_result.nfev,
                interval_result.nit,
                interval_result.nfev,
                boundary_used,
            ]
        )
        print(rows[-1])

    output_dir = PROBLEM_ROOT / "outputs" / "validation" / "q1_q2"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "seed_sensitivity.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "seed",
                "heading_deg",
                "uav_speed_mps",
                "drop_time_s",
                "fuse_delay_s",
                "strict_duration_s",
                "global_iterations",
                "global_function_evaluations",
                "interval_iterations",
                "interval_function_evaluations",
                "boundary_refinement_used",
            ]
        )
        writer.writerows(rows)


if __name__ == "__main__":
    main()
