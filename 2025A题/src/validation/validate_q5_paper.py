"""Independent grid validation for the paper-inspired Q5 solver."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROBLEM_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROBLEM_ROOT / "src" / "models"))

from smoke_q1_q2 import interval_duration  # noqa: E402
from smoke_q5 import (  # noqa: E402
    effective_simultaneous_intervals,
    load_config,
    points_from_settings,
    strategy_from_results,
    strategy_is_feasible,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=PROBLEM_ROOT / "outputs" / "model" / "q5_paper_new" / "results.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROBLEM_ROOT / "outputs" / "validation" / "q5_paper_new" / "grid_check.json",
    )
    args = parser.parse_args()

    payload = json.loads(args.results.read_text(encoding="utf-8"))
    strategy = strategy_from_results(args.results)
    constants, _, uavs, missiles = load_config(
        PROBLEM_ROOT / "configs" / "q5_paper_new.json"
    )
    levels = {
        "coarse": {"n_phi": 30, "n_z": 7, "n_radial": 5, "scan_dt_s": 0.012},
        "medium": {"n_phi": 60, "n_z": 11, "n_radial": 8, "scan_dt_s": 0.006},
        "formal": {"n_phi": 120, "n_z": 21, "n_radial": 15, "scan_dt_s": 0.003},
        "fine": {"n_phi": 180, "n_z": 31, "n_radial": 20, "scan_dt_s": 0.002},
    }
    grid_results = {}
    for name, settings in levels.items():
        intervals = effective_simultaneous_intervals(
            list(strategy.active_bombs),
            missiles,
            constants,
            uavs,
            points_from_settings(settings, constants),
            float(settings["scan_dt_s"]),
        )
        grid_results[name] = {
            "settings": settings,
            "intervals_s": [list(value) for value in intervals],
            "duration_s": interval_duration(intervals),
        }
        print(
            f"{name}: duration={grid_results[name]['duration_s']:.12f}s "
            f"intervals={grid_results[name]['intervals_s']}",
            flush=True,
        )

    fine_duration = float(grid_results["fine"]["duration_s"])
    timing_runs = payload["optimization_diagnostics"]["timing_search_runs"]
    output = {
        "strategy_feasible": strategy_is_feasible(
            strategy, constants, uavs, missiles
        ),
        "active_bomb_count": len(strategy.active_bombs),
        "grid_results": grid_results,
        "saved_vs_fine_abs_error_s": abs(
            float(payload["simultaneous_full_target_duration_s"])
            - fine_duration
        ),
        "timing_search_runs": timing_runs,
        "all_timing_runs_completed_budget": all(
            int(run["iterations"]) == 100 for run in timing_runs
        ),
        "constraint_checks": payload["constraint_checks"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
