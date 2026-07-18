from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_DIR / "src" / "models"
sys.path.insert(0, str(MODEL_DIR))

import q1_fit


SOURCE_XLSX = PROJECT_DIR / "A-附件.xlsx"
Q1_RESULT_PATH = PROJECT_DIR / "outputs" / "model" / "q1_results.json"
RESULT_PATH = PROJECT_DIR / "outputs" / "model" / "q2_results.json"
REPORT_PATH = PROJECT_DIR / "reports" / "q2_model.md"
MINIMUM_INITIAL_TORQUE_NM = 150.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_table4() -> list[dict[str, float | str]]:
    sheets = q1_fit.read_xlsx(SOURCE_XLSX)
    sheet_name = next(name for name in sheets if name.startswith("Sheet4_"))
    rows = sheets[sheet_name]
    scenarios = []
    for column, scenario_id in ((1, "A"), (2, "B")):
        scenarios.append(
            {
                "scenario": scenario_id,
                "source_label": str(rows[0][column]),
                "diameter_mm": float(rows[1][column]),
                "yield_load_kN": float(rows[2][column]),
                "borehole_diameter_mm": float(rows[3][column]),
                "bond_length_mm": float(rows[4][column]),
                "ground_modulus_GPa": float(rows[5][column]),
                "poisson_ratio": float(rows[6][column]),
                "tray_side_mm": float(rows[7][column]),
                "max_indentation_mm": float(rows[8][column]),
                "influence_depth_mm": float(rows[9][column]),
            }
        )
    return scenarios


def torque_from_pretension(
    pretension_kN: float, diameter_mm: float, torque_coefficient: float
) -> float:
    return torque_coefficient * pretension_kN * diameter_mm


def analyze_scenario(
    scenario: dict[str, float | str],
    q1_diameter_fits: dict[int, dict[str, object]],
) -> dict[str, object]:
    diameter = float(scenario["diameter_mm"])
    q1_fit_row = q1_diameter_fits[int(diameter)]
    torque_coefficient = float(q1_fit_row["K"])
    k_low, k_high = (
        float(value) for value in q1_fit_row["K_95pct_CI"]
    )
    yield_limit = float(scenario["yield_load_kN"])
    ground_modulus = float(scenario["ground_modulus_GPa"])
    poisson_ratio = float(scenario["poisson_ratio"])
    borehole_diameter = float(scenario["borehole_diameter_mm"])
    bond_length = float(scenario["bond_length_mm"])
    tray_side = float(scenario["tray_side_mm"])
    max_indentation = float(scenario["max_indentation_mm"])
    influence_depth = float(scenario["influence_depth_mm"])

    bond_shear_strength = 0.25 * ground_modulus + 1.0
    bond_limit = (
        math.pi
        * borehole_diameter
        * bond_length
        * bond_shear_strength
        * 1e-3
    )
    indentation_limit = (
        max_indentation
        * ground_modulus
        * tray_side**2
        / ((1 - poisson_ratio**2) * influence_depth)
    )
    limits = {
        "bolt_yield": yield_limit,
        "bond_failure": bond_limit,
        "tray_indentation": indentation_limit,
    }
    no_band_mode = min(limits, key=limits.get)
    no_band_limit = limits[no_band_mode]
    structural_limits = {
        "bolt_yield": yield_limit,
        "bond_failure": bond_limit,
    }
    structural_mode = min(structural_limits, key=structural_limits.get)
    structural_limit = structural_limits[structural_mode]
    band_required = indentation_limit < structural_limit

    no_band_torque = torque_from_pretension(
        no_band_limit, diameter, torque_coefficient
    )
    conditional_band_torque = torque_from_pretension(
        structural_limit, diameter, torque_coefficient
    )
    calibration_max_torque = 350.0
    return {
        **scenario,
        "torque_coefficient": torque_coefficient,
        "torque_coefficient_95pct_CI": [k_low, k_high],
        "bond_shear_strength_MPa": bond_shear_strength,
        "pretension_limits_kN": limits,
        "without_band": {
            "max_pretension_kN": no_band_limit,
            "Tmax_Nm": no_band_torque,
            "Tmax_from_K_95pct_CI_Nm": [
                torque_from_pretension(no_band_limit, diameter, k_low),
                torque_from_pretension(no_band_limit, diameter, k_high),
            ],
            "governing_mode": no_band_mode,
            "meets_minimum_initial_torque_150_Nm": (
                no_band_torque >= MINIMUM_INITIAL_TORQUE_NM
            ),
            "exceeds_K_calibration_torque_range": (
                no_band_torque > calibration_max_torque
            ),
        },
        "steel_band_required": band_required,
        "with_effective_band_conditional": {
            "assumption": (
                "A qualified steel band removes local tray indentation "
                "from the governing constraints; no band stiffness model "
                "is supplied in the problem."
            ),
            "max_pretension_kN": structural_limit,
            "Tmax_Nm": conditional_band_torque,
            "Tmax_from_K_95pct_CI_Nm": [
                torque_from_pretension(structural_limit, diameter, k_low),
                torque_from_pretension(structural_limit, diameter, k_high),
            ],
            "governing_mode": structural_mode,
            "exceeds_K_calibration_torque_range": (
                conditional_band_torque > calibration_max_torque
            ),
        },
        "constraint_margins_at_without_band_Tmax_kN": {
            mode: limit - no_band_limit for mode, limit in limits.items()
        },
    }


def build_report(results: dict[str, object]) -> str:
    mode_names = {
        "bolt_yield": "锚杆屈服",
        "bond_failure": "锚固粘结破坏",
        "tray_indentation": "托盘压陷",
    }
    lines = [
        "# 问题 2.2 最大允许预紧力矩计算",
        "",
        "计算口径：无钢带时取屈服、粘结和压陷三个预紧力上限的最小值；钢带必要性按压陷上限是否低于结构上限判断。",
        "",
        "| 工况 | 屈服/kN | 粘结/kN | 压陷/kN | 无钢带 Tmax/(N·m) | 主控模式 | 钢带 |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for row in results["scenarios"]:
        limits = row["pretension_limits_kN"]
        no_band = row["without_band"]
        lines.append(
            f"| {row['scenario']} | {limits['bolt_yield']:.3f} | "
            f"{limits['bond_failure']:.3f} | "
            f"{limits['tray_indentation']:.3f} | "
            f"{no_band['Tmax_Nm']:.3f} | "
            f"{mode_names[no_band['governing_mode']]} | "
            f"{'必须' if row['steel_band_required'] else '不必须'} |"
        )
    lines.extend(
        [
            "",
            "若钢带能够有效消除局部托盘压陷约束，则其条件性上限只由锚杆屈服与锚固粘结共同决定。题面未给钢带刚度和等效承载宽度，因此该结果不能解释为钢带承载力的独立计算值。",
            "",
            "两种工况的力矩上限均高于 K 标定试验的 350 N·m 上界，属于外推结果，需在论文中明确限制。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    q1_results = json.loads(Q1_RESULT_PATH.read_text(encoding="utf-8"))
    q1_diameter_fits = {
        int(row["diameter_mm"]): row
        for row in q1_results["question_1_1"]["diameter_fits"]
    }
    scenarios = [
        analyze_scenario(row, q1_diameter_fits) for row in extract_table4()
    ]
    results = {
        "metadata": {
            "source_file": SOURCE_XLSX.name,
            "source_sha256": sha256_file(SOURCE_XLSX),
            "q1_result_file": str(Q1_RESULT_PATH.relative_to(PROJECT_DIR)),
            "python_version": platform.python_version(),
            "random_seed": None,
            "minimum_initial_torque_Nm": MINIMUM_INITIAL_TORQUE_NM,
        },
        "formulas": {
            "bond_shear_strength_MPa": "0.25 * E_GPa + 1.0",
            "bond_limit_kN": (
                "pi * D_hole_mm * L_bond_mm * tau_MPa * 1e-3"
            ),
            "indentation_limit_kN": (
                "delta_max_mm * E_GPa * b_mm^2 "
                "/ ((1 - nu^2) * h0_mm)"
            ),
            "torque_limit_Nm": "K * P_limit_kN * d_mm",
        },
        "scenarios": scenarios,
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    REPORT_PATH.write_text(build_report(results), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
