from __future__ import annotations

import json
import math
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
RESULT_PATH = PROJECT_DIR / "outputs" / "model" / "q2_results.json"
VALIDATION_PATH = (
    PROJECT_DIR / "outputs" / "validation" / "q2_validation.json"
)
REPORT_PATH = PROJECT_DIR / "reports" / "q2_validation.md"


def close_check(name: str, actual: float, expected: float) -> dict[str, object]:
    difference = abs(actual - expected)
    tolerance = 1e-9 * max(1.0, abs(expected))
    return {
        "check": name,
        "actual": actual,
        "expected": expected,
        "absolute_difference": difference,
        "tolerance": tolerance,
        "passed": difference <= tolerance,
    }


def main() -> None:
    results = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    checks = []
    for row in results["scenarios"]:
        scenario = row["scenario"]
        E = float(row["ground_modulus_GPa"])
        nu = float(row["poisson_ratio"])
        diameter = float(row["diameter_mm"])
        k_value = float(row["torque_coefficient"])
        expected_tau = 0.25 * E + 1.0
        expected_bond = (
            math.pi
            * float(row["borehole_diameter_mm"])
            * float(row["bond_length_mm"])
            * expected_tau
            * 1e-3
        )
        expected_indent = (
            float(row["max_indentation_mm"])
            * E
            * float(row["tray_side_mm"]) ** 2
            / ((1 - nu**2) * float(row["influence_depth_mm"]))
        )
        checks.extend(
            [
                close_check(
                    f"{scenario} shear-strength formula",
                    row["bond_shear_strength_MPa"],
                    expected_tau,
                ),
                close_check(
                    f"{scenario} bond-limit formula",
                    row["pretension_limits_kN"]["bond_failure"],
                    expected_bond,
                ),
                close_check(
                    f"{scenario} indentation-limit formula",
                    row["pretension_limits_kN"]["tray_indentation"],
                    expected_indent,
                ),
            ]
        )
        limits = row["pretension_limits_kN"]
        expected_no_band = min(limits.values())
        checks.append(
            close_check(
                f"{scenario} no-band lower envelope",
                row["without_band"]["max_pretension_kN"],
                expected_no_band,
            )
        )
        checks.append(
            close_check(
                f"{scenario} torque conversion",
                row["without_band"]["Tmax_Nm"],
                k_value * diameter * expected_no_band,
            )
        )
        expected_band = limits["tray_indentation"] < min(
            limits["bolt_yield"], limits["bond_failure"]
        )
        checks.append(
            {
                "check": f"{scenario} steel-band logic",
                "actual": row["steel_band_required"],
                "expected": expected_band,
                "passed": row["steel_band_required"] == expected_band,
            }
        )
        margins = row["constraint_margins_at_without_band_Tmax_kN"]
        checks.append(
            {
                "check": f"{scenario} all constraint margins nonnegative",
                "minimum_margin_kN": min(margins.values()),
                "passed": min(margins.values()) >= -1e-10,
            }
        )

    passed = all(check["passed"] for check in checks)
    validation = {
        "status": "passed_with_limits" if passed else "failed",
        "checks": checks,
        "limitations": [
            "The effective-band Tmax is conditional because no steel-band stiffness or effective width is supplied.",
            "All reported Tmax values exceed the 350 N*m upper end of the K calibration data and are extrapolations.",
            "The theoretical upper limits are not recommended construction torques and contain no additional engineering safety reduction.",
        ],
    }
    VALIDATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    VALIDATION_PATH.write_text(
        json.dumps(validation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    REPORT_PATH.write_text(
        "\n".join(
            [
                "# 问题 2.2 模型验证",
                "",
                f"验证状态：{'有限条件下通过' if passed else '未通过'}。",
                "",
                f"- 检查项：{len(checks)}。",
                f"- 通过项：{sum(check['passed'] for check in checks)}。",
                "- 独立复算界面剪切强度、粘结上限、压陷上限、下包络、力矩转换和钢带逻辑。",
                "- 所有理论 Tmax 均超过 K 标定数据上界，属于外推。",
                "- 加钢带后的上限属于条件性结果，不能替代钢带本体设计。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
