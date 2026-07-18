from __future__ import annotations

import json
import math
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_DIR / "src" / "models"
sys.path.insert(0, str(MODEL_DIR))

import q3_eccentric_tmax as q3


MODEL_RESULT_PATH = PROJECT_DIR / "outputs" / "model" / "q3_results.json"
VALIDATION_RESULT_PATH = (
    PROJECT_DIR / "outputs" / "validation" / "q3_validation.json"
)
REPORT_PATH = PROJECT_DIR / "reports" / "q3_validation.md"


def independent_vm_limit(
    eccentricity_mm: float,
    diameter_mm: float,
    torque_coefficient: float,
    thread_torque_fraction: float,
    yield_strength_MPa: float,
    safety_factor: float,
    effective_diameter_ratio: float,
    normal_concentration: float,
    torsional_concentration: float,
) -> float:
    effective_diameter = effective_diameter_ratio * diameter_mm
    area = math.pi * effective_diameter**2 / 4.0
    bending_modulus = math.pi * effective_diameter**3 / 32.0
    torsion_modulus = math.pi * effective_diameter**3 / 16.0
    normal_slope = normal_concentration * (
        1.0 / (torque_coefficient * diameter_mm * area)
        + eccentricity_mm
        / (torque_coefficient * diameter_mm * bending_modulus)
    )
    torsion_slope = (
        torsional_concentration
        * thread_torque_fraction
        / torsion_modulus
    )
    allowable = yield_strength_MPa / safety_factor
    return (
        allowable
        / math.sqrt(normal_slope**2 + 3.0 * torsion_slope**2)
        / 1000.0
    )


def independent_bisection_limit(
    eccentricity_mm: float,
    parameters: dict[str, float],
) -> float:
    allowable = (
        parameters["yield_strength_MPa"]
        / parameters["safety_factor"]
    )

    def residual(torque_Nm: float) -> float:
        stresses = q3.stress_components(
            torque_Nm=torque_Nm,
            eccentricity_mm=eccentricity_mm,
            diameter_mm=parameters["diameter_mm"],
            torque_coefficient=parameters["torque_coefficient"],
            thread_torque_fraction=parameters[
                "thread_torque_fraction"
            ],
            effective_diameter_ratio=parameters[
                "effective_diameter_ratio"
            ],
            normal_stress_concentration_factor=parameters[
                "normal_concentration"
            ],
            torsional_stress_concentration_factor=parameters[
                "torsional_concentration"
            ],
        )
        return stresses["von_mises_MPa"] - allowable

    low = 0.0
    high = 1.0
    while residual(high) < 0:
        high *= 2.0
        if high > 1e7:
            raise RuntimeError("failed to bracket stress root")
    for _ in range(100):
        middle = 0.5 * (low + high)
        if residual(middle) <= 0:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def close(actual: float, expected: float, tolerance: float = 1e-8) -> bool:
    return abs(actual - expected) <= tolerance * max(
        1.0,
        abs(actual),
        abs(expected),
    )


def main() -> None:
    q3.main(emit_output=False)
    results = json.loads(MODEL_RESULT_PATH.read_text(encoding="utf-8"))
    config = results["parameters"]
    q2_scenarios = {
        item["scenario"]: item
        for item in json.loads(
            q3.Q2_RESULT_PATH.read_text(encoding="utf-8")
        )["scenarios"]
    }
    checks: list[dict[str, object]] = []

    def record(name: str, passed: bool, **details: object) -> None:
        checks.append({"check": name, **details, "passed": bool(passed)})

    typical = results["appendix_typical_scenario"]
    geometry = typical["geometry"]
    record(
        "effective diameter equals 0.85d",
        close(geometry["effective_diameter_mm"], 17.0),
        actual=geometry["effective_diameter_mm"],
        expected=17.0,
    )
    first = typical["curve"][0]
    record(
        "zero eccentricity gives zero bending stress",
        close(first["stress_at_Tmax"]["bending_nominal_MPa"], 0.0),
        actual=first["stress_at_Tmax"]["bending_nominal_MPa"],
        expected=0.0,
    )

    base_parameters = {
        "diameter_mm": float(
            config["appendix_typical"]["diameter_mm"]
        ),
        "torque_coefficient": float(
            config["appendix_typical"]["torque_coefficient"]
        ),
        "thread_torque_fraction": float(
            config["thread_torque_fraction"]
        ),
        "yield_strength_MPa": float(config["yield_strength_MPa"]),
        "safety_factor": float(config["safety_factor"]),
        "effective_diameter_ratio": float(
            config["thread_effective_diameter_ratio"]
        ),
        "normal_concentration": float(
            config["normal_stress_concentration_factor"]
        ),
        "torsional_concentration": float(
            config["torsional_stress_concentration_factor"]
        ),
    }
    for ratio in (0.0, 0.1, 0.25, 0.5):
        eccentricity = ratio * base_parameters["diameter_mm"]
        model_value = q3.von_mises_torque_limit(
            eccentricity_mm=eccentricity,
            diameter_mm=base_parameters["diameter_mm"],
            torque_coefficient=base_parameters["torque_coefficient"],
            thread_torque_fraction=base_parameters[
                "thread_torque_fraction"
            ],
            yield_strength_MPa=base_parameters[
                "yield_strength_MPa"
            ],
            safety_factor=base_parameters["safety_factor"],
            effective_diameter_ratio=base_parameters[
                "effective_diameter_ratio"
            ],
            normal_stress_concentration_factor=base_parameters[
                "normal_concentration"
            ],
            torsional_stress_concentration_factor=base_parameters[
                "torsional_concentration"
            ],
        )
        direct_value = independent_vm_limit(
            eccentricity_mm=eccentricity,
            **base_parameters,
        )
        root_value = independent_bisection_limit(
            eccentricity,
            base_parameters,
        )
        record(
            f"closed form equals independent formula at e/d={ratio}",
            close(model_value, direct_value),
            model_Nm=model_value,
            independent_Nm=direct_value,
        )
        record(
            f"closed form equals stress-root solution at e/d={ratio}",
            close(model_value, root_value),
            model_Nm=model_value,
            root_Nm=root_value,
        )

    typical_limits = [
        float(row["Tmax_Nm"]) for row in typical["curve"]
    ]
    record(
        "typical Tvm is nonincreasing with eccentricity",
        all(
            later <= earlier + 1e-10
            for earlier, later in zip(
                typical_limits,
                typical_limits[1:],
            )
        ),
        first_Nm=typical_limits[0],
        last_Nm=typical_limits[-1],
    )
    vm_stress_errors = [
        abs(
            float(row["stress_at_Tmax"]["von_mises_MPa"])
            - float(config["yield_strength_MPa"])
            / float(config["safety_factor"])
        )
        for row in typical["curve"]
    ]
    record(
        "stress at Tvm equals allowable stress",
        max(vm_stress_errors) <= 1e-8,
        maximum_absolute_error_MPa=max(vm_stress_errors),
    )
    typical_threshold = float(
        typical[
            "minimum_initial_torque_feasible_until_eccentricity_mm"
        ]
    )
    typical_threshold_limit = q3.von_mises_torque_limit(
        eccentricity_mm=typical_threshold,
        diameter_mm=base_parameters["diameter_mm"],
        torque_coefficient=base_parameters["torque_coefficient"],
        thread_torque_fraction=base_parameters[
            "thread_torque_fraction"
        ],
        yield_strength_MPa=base_parameters["yield_strength_MPa"],
        safety_factor=base_parameters["safety_factor"],
        effective_diameter_ratio=base_parameters[
            "effective_diameter_ratio"
        ],
        normal_stress_concentration_factor=base_parameters[
            "normal_concentration"
        ],
        torsional_stress_concentration_factor=base_parameters[
            "torsional_concentration"
        ],
    )
    record(
        "typical minimum-torque feasibility boundary equals 150 N.m",
        close(
            typical_threshold_limit,
            float(config["minimum_initial_torque_Nm"]),
        ),
        boundary_eccentricity_mm=typical_threshold,
        limit_Nm=typical_threshold_limit,
    )

    for scenario in results["combined_scenarios"]:
        q2_scenario = q2_scenarios[scenario["scenario"]]
        curve = scenario["curve"]
        combined_limits = [
            float(row["combined_Tmax_Nm"]) for row in curve
        ]
        record(
            (
                f"{scenario['scenario']} {scenario['band_state']} "
                "combined envelope is nonincreasing"
            ),
            all(
                later <= earlier + 1e-10
                for earlier, later in zip(
                    combined_limits,
                    combined_limits[1:],
                )
            ),
            first_Nm=combined_limits[0],
            last_Nm=combined_limits[-1],
        )
        minimum_margin = min(
            float(margin)
            for row in curve
            for margin in row["constraint_margins_Nm"].values()
        )
        record(
            (
                f"{scenario['scenario']} {scenario['band_state']} "
                "all torque margins are nonnegative"
            ),
            minimum_margin >= -1e-8,
            minimum_margin_Nm=minimum_margin,
        )
        envelope_errors = []
        for row in curve:
            all_limits = {
                **row["engineering_torque_limits_Nm"],
                "thread_von_mises": row["von_mises_limit_Nm"],
            }
            envelope_errors.append(
                abs(
                    float(row["combined_Tmax_Nm"])
                    - min(float(value) for value in all_limits.values())
                )
            )
        record(
            (
                f"{scenario['scenario']} {scenario['band_state']} "
                "combined value equals exact lower envelope"
            ),
            max(envelope_errors) <= 1e-8,
            maximum_absolute_error_Nm=max(envelope_errors),
        )
        if scenario["band_state"] == "without_band":
            tray_base = q3.fixed_q2_torque_limits(
                q2_scenario,
                include_indentation=True,
            )["tray_indentation"]
            tray_errors = [
                abs(
                    float(
                        row["engineering_torque_limits_Nm"][
                            "tray_indentation"
                        ]
                    )
                    - tray_base
                    / (
                        1.0
                        + 6.0
                        * float(row["eccentricity_mm"])
                        / float(q2_scenario["tray_side_mm"])
                    )
                )
                for row in curve
            ]
            record(
                (
                    f"{scenario['scenario']} tray correction follows "
                    "the full-contact pressure formula"
                ),
                max(tray_errors) <= 1e-8,
                maximum_absolute_error_Nm=max(tray_errors),
            )
            maximum_eccentricity = max(
                float(row["eccentricity_mm"]) for row in curve
            )
            full_contact_limit = (
                float(q2_scenario["tray_side_mm"]) / 6.0
            )
            record(
                (
                    f"{scenario['scenario']} scan stays inside "
                    "the full-contact tray range"
                ),
                maximum_eccentricity <= full_contact_limit,
                maximum_eccentricity_mm=maximum_eccentricity,
                full_contact_limit_mm=full_contact_limit,
            )
        for switch_row in scenario["mode_switches"]:
            switch = float(switch_row["eccentricity_mm"])
            engineering_limits = q3.engineering_torque_limits(
                q2_scenario=q2_scenario,
                eccentricity_mm=switch,
                include_indentation=(
                    scenario["band_state"] == "without_band"
                ),
                tray_eccentric_pressure_correction=bool(
                    config["tray_eccentric_pressure_correction"]
                ),
                bond_eccentricity_reduction_alpha=float(
                    config["bond_eccentricity_reduction_alpha"]
                ),
            )
            switch_vm = q3.von_mises_torque_limit(
                eccentricity_mm=switch,
                diameter_mm=float(scenario["diameter_mm"]),
                torque_coefficient=float(
                    scenario["torque_coefficient"]
                ),
                thread_torque_fraction=float(
                    config["thread_torque_fraction"]
                ),
                yield_strength_MPa=float(
                    config["yield_strength_MPa"]
                ),
                safety_factor=float(config["safety_factor"]),
                effective_diameter_ratio=float(
                    config["thread_effective_diameter_ratio"]
                ),
                normal_stress_concentration_factor=float(
                    config[
                        "normal_stress_concentration_factor"
                    ]
                ),
                torsional_stress_concentration_factor=float(
                    config[
                        "torsional_stress_concentration_factor"
                    ]
                ),
            )
            competing_values = {
                **engineering_limits,
                "thread_von_mises": switch_vm,
            }
            first_value = competing_values[
                switch_row["from_mode"]
            ]
            second_value = competing_values[
                switch_row["to_mode"]
            ]
            record(
                (
                    f"{scenario['scenario']} {scenario['band_state']} "
                    "mode switch is continuous"
                ),
                close(first_value, second_value),
                switch_eccentricity_mm=switch,
                from_mode=switch_row["from_mode"],
                to_mode=switch_row["to_mode"],
                from_limit_Nm=first_value,
                to_limit_Nm=second_value,
            )
        threshold = scenario[
            "minimum_initial_torque_feasible_until_eccentricity_mm"
        ]
        if threshold is not None:
            threshold = float(threshold)
            engineering_limits = q3.engineering_torque_limits(
                q2_scenario=q2_scenario,
                eccentricity_mm=threshold,
                include_indentation=(
                    scenario["band_state"] == "without_band"
                ),
                tray_eccentric_pressure_correction=bool(
                    config["tray_eccentric_pressure_correction"]
                ),
                bond_eccentricity_reduction_alpha=float(
                    config["bond_eccentricity_reduction_alpha"]
                ),
            )
            vm_limit = q3.von_mises_torque_limit(
                eccentricity_mm=threshold,
                diameter_mm=float(scenario["diameter_mm"]),
                torque_coefficient=float(
                    scenario["torque_coefficient"]
                ),
                thread_torque_fraction=float(
                    config["thread_torque_fraction"]
                ),
                yield_strength_MPa=float(
                    config["yield_strength_MPa"]
                ),
                safety_factor=float(config["safety_factor"]),
                effective_diameter_ratio=float(
                    config["thread_effective_diameter_ratio"]
                ),
                normal_stress_concentration_factor=float(
                    config[
                        "normal_stress_concentration_factor"
                    ]
                ),
                torsional_stress_concentration_factor=float(
                    config[
                        "torsional_stress_concentration_factor"
                    ]
                ),
            )
            threshold_limit = min(
                *engineering_limits.values(),
                vm_limit,
            )
            record(
                (
                    f"{scenario['scenario']} {scenario['band_state']} "
                    "minimum-torque boundary equals 150 N.m"
                ),
                close(
                    threshold_limit,
                    float(config["minimum_initial_torque_Nm"]),
                ),
                boundary_eccentricity_mm=threshold,
                limit_Nm=threshold_limit,
            )

    sensitivity = results["sensitivity"]

    def sensitivity_value(
        eccentricity_ratio: float,
        effective_ratio: float,
        torque_fraction: float,
        concentration: float,
    ) -> float:
        row = next(
            item
            for item in sensitivity
            if close(
                item["eccentricity_ratio"],
                eccentricity_ratio,
            )
            and close(
                item["effective_diameter_ratio"],
                effective_ratio,
            )
            and close(
                item["thread_torque_fraction"],
                torque_fraction,
            )
            and close(
                item["stress_concentration_factor"],
                concentration,
            )
        )
        return float(row["Tmax_Nm"])

    for ratio in config["sensitivity"][
        "evaluation_eccentricity_ratios"
    ]:
        diameter_values = [
            sensitivity_value(
                float(ratio),
                effective_ratio,
                0.09,
                1.0,
            )
            for effective_ratio in (0.8, 0.85, 0.9)
        ]
        record(
            f"larger effective diameter raises Tvm at e/d={ratio}",
            diameter_values[0] < diameter_values[1] < diameter_values[2],
            values_Nm=diameter_values,
        )
        concentration_values = [
            sensitivity_value(
                float(ratio),
                0.85,
                0.09,
                concentration,
            )
            for concentration in (1.0, 1.2, 1.5)
        ]
        record(
            f"larger stress concentration lowers Tvm at e/d={ratio}",
            (
                concentration_values[0]
                > concentration_values[1]
                > concentration_values[2]
            ),
            values_Nm=concentration_values,
        )

    passed = sum(bool(check["passed"]) for check in checks)
    status = (
        "passed_with_limits" if passed == len(checks) else "failed"
    )
    validation = {
        "status": status,
        "checks_passed": passed,
        "checks_total": len(checks),
        "checks": checks,
        "limitations": [
            (
                "The e/d range 0 to 0.5 is a declared numerical "
                "scenario because the problem gives no eccentricity range."
            ),
            (
                "The nominal section model does not calibrate thread-root "
                "stress concentration or local plasticity."
            ),
            (
                "Scenario B uses Appendix 3 material strength and Ks for "
                "a 22 mm bolt; this is an explicitly marked extension."
            ),
            (
                "All construction recommendations still require a safety "
                "factor supported by engineering standards or tests."
            ),
        ],
    }
    VALIDATION_RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    VALIDATION_RESULT_PATH.write_text(
        json.dumps(validation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report_lines = [
        "# 问题三模型验证",
        "",
        f"验证状态：{'有限条件下通过' if status == 'passed_with_limits' else '未通过'}。",
        "",
        f"- 检查项：{len(checks)}。",
        f"- 通过项：{passed}。",
        "- 闭式解与独立公式、应力方程二分求根结果一致。",
        "- e = 0 时弯曲应力严格退化为 0，Tmax 随偏心距单调不增。",
        "- 工况 A、B 的综合结果逐点等于全部约束的精确下包络，约束余量非负。",
        "- 已检查主控模式切换连续性及有效直径、应力集中系数的敏感性方向。",
        "",
        "限制：偏心扫描范围属于声明的数值情景；螺纹根部真实应力集中、局部塑性和安装冲击仍需试验或有限元校准。",
        "",
    ]
    REPORT_PATH.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    if status == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
