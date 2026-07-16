"""Independent resolution and contribution checks for refined question 4."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


PROBLEM_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROBLEM_ROOT / "src" / "models"))

from smoke_q4 import (  # noqa: E402
    DEFAULT_CONFIG,
    Q4Bomb,
    Q4Strategy,
    effective_intervals,
    interval_duration,
    load_config,
    target_surface_points,
)


RESULTS = PROBLEM_ROOT / "outputs" / "model" / "q4_refined" / "results.json"
OUTPUT_DIR = PROBLEM_ROOT / "outputs" / "validation" / "q4"


def main() -> None:
    constants, _, uavs = load_config(DEFAULT_CONFIG)
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    bombs = tuple(
        Q4Bomb(
            item["uav_id"],
            float(item["heading_deg"]),
            float(item["speed_mps"]),
            float(item["drop_time_s"]),
            float(item["fuse_delay_s"]),
        )
        for item in data["strategy"]["bombs"]
    )
    strategy = Q4Strategy(bombs)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for label, n_phi, n_z, n_radial, dt in [
        ("coarse", 60, 11, 8, 0.005),
        ("medium", 120, 21, 15, 0.003),
        ("fine", 180, 31, 21, 0.002),
    ]:
        points = target_surface_points(n_phi, n_z, n_radial, constants)
        intervals = effective_intervals(
            strategy, constants, uavs, points, scan_dt_s=dt
        )
        rows.append(
            [label, n_phi, n_z, n_radial, dt, interval_duration(intervals), json.dumps(intervals)]
        )
    with (OUTPUT_DIR / "resolution_convergence.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["grid", "n_phi", "n_z", "n_radial", "scan_dt_s", "duration_s", "intervals_s"]
        )
        writer.writerows(rows)

    points = target_surface_points(120, 21, 15, constants)
    contribution_rows = []
    for bomb in bombs:
        repeated = Q4Strategy((bomb, bomb, bomb))
        intervals = effective_intervals(
            repeated, constants, uavs, points, scan_dt_s=0.003
        )
        contribution_rows.append(
            [bomb.uav_id, interval_duration(intervals), json.dumps(intervals)]
        )
    with (OUTPUT_DIR / "individual_contributions.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["uav_id", "individual_duration_s", "intervals_s"])
        writer.writerows(contribution_rows)

    print(json.dumps({"resolution": rows, "individual": contribution_rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
