from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_DIR / "src" / "models"
sys.path.insert(0, str(MODEL_DIR))

import q1_fit


RESULT_PATH = PROJECT_DIR / "outputs" / "model" / "q1_results.json"
VALIDATION_DIR = PROJECT_DIR / "outputs" / "validation"
VALIDATION_REPORT = PROJECT_DIR / "reports" / "model-validation.md"
VALIDATION_JSON = VALIDATION_DIR / "q1_validation.json"


def check_close(name: str, actual: float, expected: float, tolerance: float) -> dict:
    difference = abs(actual - expected)
    return {
        "check": name,
        "actual": actual,
        "expected": expected,
        "absolute_difference": difference,
        "tolerance": tolerance,
        "passed": difference <= tolerance,
    }


def main() -> None:
    saved = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    sheets = q1_fit.read_xlsx(q1_fit.SOURCE_XLSX)
    standard, field = q1_fit.extract_q1_data(sheets)
    config = json.loads(q1_fit.CONFIG_PATH.read_text(encoding="utf-8"))

    checks = []
    for row in saved["question_1_1"]["diameter_fits"]:
        diameter = int(row["diameter_mm"])
        torque = standard[diameter]["torque_Nm"]
        pretension = standard[diameter]["pretension_kN"]
        independent_slope = float(np.sum(torque * pretension) / np.sum(torque**2))
        independent_k = 1 / (diameter * independent_slope)
        checks.append(
            check_close(
                f"K independent sum-product check, d={diameter}",
                row["K"],
                independent_k,
                1e-12,
            )
        )

    continuity_checks = []
    breakpoint_checks = []
    tail_checks = []
    for condition, dataset in field.items():
        result = saved["question_1_2"][condition]
        breakpoint = float(result["estimated_breakpoint_Nm"])
        for formula in result["formula"]["point_formulas"]:
            before = formula["before"]
            after = formula["after"]
            if result["selected_model"] == "power_linear_c0":
                left = (
                    before["intercept_kN"]
                    + before["scale_kN_per_Nm_power"]
                    * breakpoint ** before["exponent"]
                )
            else:
                left = (
                    before["intercept_kN"]
                    + before["linear_kN_per_Nm"] * breakpoint
                    + before["quadratic_kN_per_Nm2"] * breakpoint**2
                )
            right = (
                after["intercept_kN"]
                + after["slope_kN_per_Nm"] * breakpoint
            )
            continuity_checks.append(
                check_close(
                    f"{condition} point {formula['point_id']} continuity",
                    left,
                    right,
                    1e-9,
                )
            )

        torque_levels = dataset["torque_Nm"]
        response = dataset["pretension_kN"]
        point_count = response.shape[1]
        torque = np.repeat(torque_levels, point_count)
        point_id = np.tile(np.arange(point_count), len(torque_levels))
        y = response.reshape(-1)
        step = float(config["breakpoint_grid_step_Nm"])
        grid = np.arange(torque_levels[2], torque_levels[-3] + step / 2, step)
        _, profile = q1_fit.profile_piecewise(
            torque,
            point_id,
            y,
            point_count,
            result["selected_model"],
            grid,
            int(config["minimum_torque_levels_each_side"]),
            tuple(float(value) for value in config["power_exponent_bounds"]),
            float(config["power_exponent_tolerance"]),
        )
        independently_selected = min(profile, key=lambda row: row["BIC"])[
            "breakpoint_Nm"
        ]
        breakpoint_checks.append(
            check_close(
                f"{condition} breakpoint profile minimum",
                breakpoint,
                independently_selected,
                1e-12,
            )
        )

        practical = float(result["practical_tested_threshold_Nm"])
        independent_tail = q1_fit.tail_diagnostics(
            torque_levels, response, practical
        )
        minimum_r2 = min(row["R2"] for row in independent_tail)
        tail_checks.append(
            {
                "check": f"{condition} tail minimum R2 >= 0.95",
                "minimum_R2": minimum_r2,
                "threshold_Nm": practical,
                "passed": minimum_r2 >= 0.95,
            }
        )

    all_checks = checks + continuity_checks + breakpoint_checks + tail_checks
    passed = all(item["passed"] for item in all_checks)
    validation = {
        "status": "passed_with_limits" if passed else "failed",
        "checks": all_checks,
        "limitations": [
            "No full data-audit report exists; validation is limited to Q1 source ranges.",
            "Breakpoint precision is limited by the 25 N·m experimental grid.",
            "Tail linearity threshold is R2 >= 0.95 for each measurement point.",
        ],
    }

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    VALIDATION_JSON.write_text(
        json.dumps(validation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    VALIDATION_REPORT.write_text(
        "\n".join(
            [
                "# 问题 1 模型验证",
                "",
                f"验证状态：{'有限条件下通过' if passed else '未通过'}。",
                "",
                "## 验证内容",
                "",
                "- 从原始 Excel 重新提取问题 1 数据。",
                "- 用独立求和公式复算三个直径的 K。",
                "- 检查分段公式在临界点处连续。",
                "- 重新执行断点 BIC 剖面搜索。",
                "- 新增幂函数、对数函数和三次函数后，按相同断点网格比较 BIC、AICc、整组留一 RMSE 与断点稳定范围。",
                "- 检查实用临界点后每个测点的线性拟合 R² 不低于 0.95。",
                "",
                "## 结果",
                "",
                f"- 检查项总数：{len(all_checks)}。",
                f"- 通过项：{sum(item['passed'] for item in all_checks)}。",
                f"- 未通过项：{sum(not item['passed'] for item in all_checks)}。",
                "",
                "## 限制",
                "",
                "- 尚无完整数据审计报告，本验证仅覆盖问题 1 使用的数据区。",
                "- 原实验力矩步长为 25 N·m，模型断点的小数精度不可解释为实验测量精度。",
                "- 5 个测点的空间或重复试验结构未由题面说明。",
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
