"""Resolution and local-perturbation checks for questions 1 and 2."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from smoke_q1_q2 import (
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT_DIR,
    Strategy,
    effective_intervals,
    interval_duration,
    load_config,
    refined_interval_boundaries,
    strategy_is_feasible,
)


def strict_duration(strategy: Strategy, constants, n_angle: int, dt: float) -> tuple:
    sampled = effective_intervals(
        strategy,
        constants,
        n_angle=n_angle,
        scan_dt_s=dt,
    )
    refined = refined_interval_boundaries(sampled, strategy, constants)
    return refined, interval_duration(refined)


def main() -> None:
    constants, config = load_config(DEFAULT_CONFIG)
    results = json.loads(
        (DEFAULT_OUTPUT_DIR / "results.json").read_text(encoding="utf-8")
    )
    q1_data = results["question_1"]["strategy"]
    q2_data = results["question_2"]["strategy"]
    q1 = Strategy(**q1_data)
    q2 = Strategy(**q2_data)

    resolutions = [
        (180, 0.010),
        (360, 0.005),
        (720, 0.004),
        (1440, 0.002),
        (2880, 0.001),
    ]
    convergence_rows = []
    for name, strategy in (("question_1", q1), ("question_2", q2)):
        for n_angle, dt in resolutions:
            intervals, duration = strict_duration(
                strategy, constants, n_angle, dt
            )
            convergence_rows.append(
                [
                    name,
                    n_angle,
                    dt,
                    duration,
                    json.dumps(intervals),
                ]
            )

    perturbations = [
        ("base", q2),
        (
            "heading_minus_0.05_deg",
            Strategy(
                q2.heading_deg - 0.05,
                q2.uav_speed_mps,
                q2.burst_time_s,
                q2.fuse_delay_s,
            ),
        ),
        (
            "heading_plus_0.05_deg",
            Strategy(
                q2.heading_deg + 0.05,
                q2.uav_speed_mps,
                q2.burst_time_s,
                q2.fuse_delay_s,
            ),
        ),
        (
            "speed_plus_0.25_mps",
            Strategy(
                q2.heading_deg,
                q2.uav_speed_mps + 0.25,
                q2.burst_time_s,
                q2.fuse_delay_s,
            ),
        ),
        (
            "drop_0.02_s_later_same_burst",
            Strategy(
                q2.heading_deg,
                q2.uav_speed_mps,
                q2.burst_time_s,
                q2.fuse_delay_s - 0.02,
            ),
        ),
        (
            "immediate_drop_fuse_minus_0.02_s",
            Strategy(
                q2.heading_deg,
                q2.uav_speed_mps,
                q2.burst_time_s - 0.02,
                q2.fuse_delay_s - 0.02,
            ),
        ),
        (
            "immediate_drop_fuse_plus_0.02_s",
            Strategy(
                q2.heading_deg,
                q2.uav_speed_mps,
                q2.burst_time_s + 0.02,
                q2.fuse_delay_s + 0.02,
            ),
        ),
    ]
    perturbation_rows = []
    for label, strategy in perturbations:
        feasible = strategy_is_feasible(strategy, constants)
        if feasible:
            intervals, duration = strict_duration(
                strategy, constants, n_angle=720, dt=0.004
            )
        else:
            intervals, duration = [], 0.0
        perturbation_rows.append(
            [
                label,
                feasible,
                strategy.heading_deg,
                strategy.uav_speed_mps,
                strategy.drop_time_s,
                strategy.fuse_delay_s,
                duration,
                json.dumps(intervals),
            ]
        )

    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (DEFAULT_OUTPUT_DIR / "resolution_convergence.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["question", "target_angles", "scan_dt_s", "duration_s", "intervals_s"]
        )
        writer.writerows(convergence_rows)

    with (DEFAULT_OUTPUT_DIR / "local_perturbations.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "case",
                "feasible",
                "heading_deg",
                "uav_speed_mps",
                "drop_time_s",
                "fuse_delay_s",
                "duration_s",
                "intervals_s",
            ]
        )
        writer.writerows(perturbation_rows)

    print(json.dumps(
        {
            "resolution_convergence": convergence_rows,
            "local_perturbations": perturbation_rows,
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
