"""Resolution, constraint, and local-perturbation checks for question 3."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


PROBLEM_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROBLEM_ROOT / "src" / "models"))

from smoke_q3 import (  # noqa: E402
    DEFAULT_CONFIG,
    Q3Strategy,
    interval_duration,
    joint_effective_intervals,
    load_config,
    q3_is_feasible,
    target_surface_points,
)


OUTPUT_DIR = PROBLEM_ROOT / "outputs" / "validation" / "q3"


def stored_strategy(config: dict) -> Q3Strategy:
    data = config["stored_strategy"]
    return Q3Strategy(
        heading_deg=float(data["heading_deg"]),
        uav_speed_mps=float(data["uav_speed_mps"]),
        drop_times_s=tuple(map(float, data["drop_times_s"])),
        fuse_delays_s=tuple(map(float, data["fuse_delays_s"])),
    )


def duration(strategy, constants, n_phi, n_z, n_radial, dt):
    points = target_surface_points(n_phi, n_z, n_radial, constants)
    intervals = joint_effective_intervals(
        strategy, constants, points, scan_dt_s=dt
    )
    return intervals, interval_duration(intervals)


def main() -> None:
    constants, config = load_config(DEFAULT_CONFIG)
    base = stored_strategy(config)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    resolution_rows = []
    for label, n_phi, n_z, n_radial, dt in [
        ("coarse", 60, 11, 8, 0.005),
        ("medium", 120, 21, 15, 0.002),
        ("fine", 180, 31, 21, 0.002),
    ]:
        intervals, total = duration(
            base, constants, n_phi, n_z, n_radial, dt
        )
        resolution_rows.append(
            [label, n_phi, n_z, n_radial, dt, total, json.dumps(intervals)]
        )

    with (OUTPUT_DIR / "resolution_convergence.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "grid",
                "n_phi",
                "n_z",
                "n_radial",
                "scan_dt_s",
                "joint_duration_s",
                "intervals_s",
            ]
        )
        writer.writerows(resolution_rows)

    perturbations = [
        ("base", base),
        (
            "heading_minus_0.05_deg",
            Q3Strategy(
                base.heading_deg - 0.05,
                base.uav_speed_mps,
                base.drop_times_s,
                base.fuse_delays_s,
            ),
        ),
        (
            "heading_plus_0.05_deg",
            Q3Strategy(
                base.heading_deg + 0.05,
                base.uav_speed_mps,
                base.drop_times_s,
                base.fuse_delays_s,
            ),
        ),
        (
            "speed_minus_0.25_mps",
            Q3Strategy(
                base.heading_deg,
                base.uav_speed_mps - 0.25,
                base.drop_times_s,
                base.fuse_delays_s,
            ),
        ),
        (
            "speed_plus_0.25_mps",
            Q3Strategy(
                base.heading_deg,
                base.uav_speed_mps + 0.25,
                base.drop_times_s,
                base.fuse_delays_s,
            ),
        ),
        (
            "second_fuse_plus_0.02_s",
            Q3Strategy(
                base.heading_deg,
                base.uav_speed_mps,
                base.drop_times_s,
                (
                    base.fuse_delays_s[0],
                    base.fuse_delays_s[1] + 0.02,
                    base.fuse_delays_s[2],
                ),
            ),
        ),
    ]
    perturbation_rows = []
    points = target_surface_points(60, 11, 8, constants)
    for label, strategy in perturbations:
        intervals = joint_effective_intervals(
            strategy, constants, points, scan_dt_s=0.005
        )
        perturbation_rows.append(
            [label, q3_is_feasible(strategy, constants), interval_duration(intervals)]
        )
    with (OUTPUT_DIR / "local_perturbations.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["perturbation", "feasible", "joint_duration_s"])
        writer.writerows(perturbation_rows)

    print(json.dumps({
        "feasible": q3_is_feasible(base, constants),
        "resolution": resolution_rows,
        "perturbations": perturbation_rows,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
