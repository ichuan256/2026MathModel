"""Independent grid and constraint validation for corrected Q5 results."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROBLEM_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROBLEM_ROOT / "src" / "models"))

from smoke_q1_q2 import interval_duration  # noqa: E402
from smoke_q5 import (  # noqa: E402
    BombPlan,
    Q5Strategy,
    effective_simultaneous_intervals,
    load_config,
    points_from_settings,
    strategy_is_feasible,
)


RESULT_PATH = (
    PROBLEM_ROOT / "outputs" / "model" / "q5_corrected" / "results.json"
)
OUTPUT_PATH = (
    PROBLEM_ROOT / "outputs" / "validation" / "q5_corrected" / "grid_check.json"
)


def load_saved_strategy(payload: dict) -> Q5Strategy:
    return Q5Strategy(
        tuple(
            BombPlan(
                uav_id=str(row["uav_id"]),
                smoke_id=int(row["smoke_id"]),
                target_id="ALL",
                heading_deg=float(row["heading_deg"]),
                speed_mps=float(row["speed_mps"]),
                drop_time_s=float(row["drop_time_s"]),
                fuse_delay_s=float(row["fuse_delay_s"]),
            )
            for row in payload["active_bombs"]
        )
    )


def main() -> None:
    payload = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    strategy = load_saved_strategy(payload)
    constants, _, uavs, missiles = load_config(PROBLEM_ROOT / "configs" / "q5.json")
    levels = {
        "coarse_check": {
            "n_phi": 30,
            "n_z": 7,
            "n_radial": 5,
            "scan_dt_s": 0.012,
        },
        "medium_check": {
            "n_phi": 60,
            "n_z": 11,
            "n_radial": 8,
            "scan_dt_s": 0.006,
        },
        "formal_check": {
            "n_phi": 120,
            "n_z": 21,
            "n_radial": 15,
            "scan_dt_s": 0.003,
        },
    }
    grid_results = {}
    for level, settings in levels.items():
        intervals = effective_simultaneous_intervals(
            list(strategy.active_bombs),
            missiles,
            constants,
            uavs,
            points_from_settings(settings, constants),
            float(settings["scan_dt_s"]),
        )
        grid_results[level] = {
            "settings": settings,
            "intervals_s": [list(value) for value in intervals],
            "duration_s": interval_duration(intervals),
        }

    formal_duration = float(grid_results["formal_check"]["duration_s"])
    output = {
        "strategy_feasible": strategy_is_feasible(
            strategy, constants, uavs, missiles
        ),
        "active_bomb_count": len(strategy.active_bombs),
        "grid_results": grid_results,
        "saved_vs_formal_abs_error_s": abs(
            float(payload["simultaneous_full_target_duration_s"])
            - formal_duration
        ),
        "completed_requested_generations": (
            int(payload["optimization_diagnostics"]["global_runs"][0]["iterations"])
            == 500
        ),
        "function_evaluations": int(
            payload["optimization_diagnostics"]["global_runs"][0][
                "function_evaluations"
            ]
        ),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
