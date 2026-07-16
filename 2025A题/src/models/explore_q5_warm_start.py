"""Find single-cloud building blocks for a positive Q5 simultaneous baseline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

from smoke_q5 import (
    BombPlan,
    MISSILE_IDS,
    UAV_IDS,
    coverage_series_for_missile,
    load_config,
    points_from_settings,
)


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    c, config, uavs, missiles = load_config(ROOT / "configs" / "q5.json")
    points = points_from_settings(config["optimization"]["coarse"], c)
    observation_time = 15.0
    rows = []
    for missile_id in MISSILE_IDS:
        for uav_id in UAV_IDS:
            fuse_max = min(observation_time, np.sqrt(2.0 * uavs[uav_id][2] / c.gravity_mps2))

            def objective(x: np.ndarray) -> float:
                heading, speed, drop, fuse = map(float, x)
                bomb = BombPlan(
                    uav_id=uav_id,
                    smoke_id=1,
                    target_id="ALL",
                    heading_deg=heading,
                    speed_mps=speed,
                    drop_time_s=drop,
                    fuse_delay_s=fuse,
                )
                if bomb.burst_time_s > observation_time:
                    return 10.0 + bomb.burst_time_s - observation_time
                _, fraction, guide = coverage_series_for_missile(
                    np.array([observation_time]),
                    [bomb],
                    missiles[missile_id],
                    c,
                    uavs,
                    points,
                )
                if fraction[0] > 0.0:
                    return -100.0 - float(fraction[0])
                return -1.0e-8 * max(float(guide[0]), -1.0e8)

            result = differential_evolution(
                objective,
                bounds=[(0.0, 360.0), (70.0, 140.0), (0.0, observation_time), (0.0, float(fuse_max))],
                seed=2025,
                popsize=6,
                maxiter=100,
                tol=1.0e-6,
                polish=True,
            )
            best = np.asarray(result.x, dtype=float)
            best_bomb = BombPlan(
                uav_id=uav_id,
                smoke_id=1,
                target_id="ALL",
                heading_deg=float(best[0]),
                speed_mps=float(best[1]),
                drop_time_s=float(best[2]),
                fuse_delay_s=float(best[3]),
            )
            _, best_fraction, _ = coverage_series_for_missile(
                np.array([observation_time]),
                [best_bomb],
                missiles[missile_id],
                c,
                uavs,
                points,
            )
            rows.append(
                {
                    "missile_id": missile_id,
                    "uav_id": uav_id,
                    "observation_time_s": observation_time,
                    "fraction": float(best_fraction[0]),
                    "heading_deg": float(result.x[0]),
                    "speed_mps": float(result.x[1]),
                    "drop_time_s": float(result.x[2]),
                    "fuse_delay_s": float(result.x[3]),
                    "burst_time_s": float(result.x[2] + result.x[3]),
                }
            )
    path = ROOT / "outputs" / "model" / "q5_corrected" / "warm_start_candidates.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
