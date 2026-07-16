"""Focused two-stage refinement for the question-4 strategy."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


PROBLEM_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROBLEM_ROOT / "src" / "models"))

from smoke_q4 import (  # noqa: E402
    DEFAULT_CONFIG,
    GridObjective,
    Q4Bomb,
    Q4Strategy,
    build_payload,
    decode_vector,
    encode_strategy,
    load_config,
    points_from_settings,
    write_outputs,
)


SOURCE_RESULTS = PROBLEM_ROOT / "outputs" / "model" / "q4" / "results.json"
OUTPUT_DIR = PROBLEM_ROOT / "outputs" / "model" / "q4_refined"


def main() -> None:
    constants, config, uavs = load_config(DEFAULT_CONFIG)
    source = json.loads(SOURCE_RESULTS.read_text(encoding="utf-8"))
    old = {
        item["uav_id"]: Q4Bomb(
            item["uav_id"],
            float(item["heading_deg"]),
            float(item["speed_mps"]),
            float(item["drop_time_s"]),
            float(item["fuse_delay_s"]),
        )
        for item in source["strategy"]["bombs"]
    }
    q2_verified_fy1 = Q4Bomb(
        "FY1",
        176.64108554797772,
        70.0,
        0.0,
        2.4967986905283013,
    )
    initial = Q4Strategy((q2_verified_fy1, old["FY2"], old["FY3"]))
    x0 = encode_strategy(initial)

    stage1_settings = config["optimization"]["local"]
    stage1_objective = GridObjective(
        constants,
        uavs,
        points_from_settings(stage1_settings, constants),
        float(stage1_settings["scan_dt_s"]),
    )
    stage1 = minimize(
        stage1_objective,
        x0,
        method="Nelder-Mead",
        options={
            "maxiter": 3000,
            "xatol": 1.0e-8,
            "fatol": 1.0e-8,
        },
    )
    stage1_x = stage1.x if stage1.fun < stage1_objective(x0) else x0

    stage2_settings = {
        "n_phi": 60,
        "n_z": 11,
        "n_radial": 8,
        "scan_dt_s": 0.01,
    }
    stage2_objective = GridObjective(
        constants,
        uavs,
        points_from_settings(stage2_settings, constants),
        float(stage2_settings["scan_dt_s"]),
    )
    stage2 = minimize(
        stage2_objective,
        stage1_x,
        method="Nelder-Mead",
        options={
            "maxiter": 2500,
            "xatol": 1.0e-9,
            "fatol": 1.0e-9,
        },
    )
    candidates = [x0, stage1_x, stage2.x]
    scores = [float(stage2_objective(value)) for value in candidates]
    best_x = candidates[int(np.argmin(scores))]
    strategy = decode_vector(best_x)
    diagnostics = {
        "source": str(SOURCE_RESULTS),
        "fy1_warm_start": "verified question-2 strategy",
        "stage1": {
            "success": bool(stage1.success),
            "message": str(stage1.message),
            "iterations": int(stage1.nit),
            "function_evaluations": int(stage1.nfev),
            "objective": float(stage1.fun),
        },
        "stage2": {
            "success": bool(stage2.success),
            "message": str(stage2.message),
            "iterations": int(stage2.nit),
            "function_evaluations": int(stage2.nfev),
            "objective": float(stage2.fun),
        },
        "candidate_stage2_objectives": scores,
    }
    payload = build_payload(strategy, diagnostics, constants, config, uavs)
    write_outputs(OUTPUT_DIR, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
