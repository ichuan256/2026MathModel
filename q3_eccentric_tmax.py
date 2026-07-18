from __future__ import annotations

import json
import math
import platform
from pathlib import Path
from typing import Callable, Iterable


PROJECT_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_DIR / "configs" / "q3_eccentricity.json"
Q2_RESULT_PATH = PROJECT_DIR / "outputs" / "model" / "q2_results.json"
RESULT_PATH = PROJECT_DIR / "outputs" / "model" / "q3_results.json"
REPORT_PATH = PROJECT_DIR / "reports" / "q3_model.md"


def geometry(
    diameter_mm: float,
    effective_diameter_ratio: float,
) -> dict[str, float]:
    effective_diameter = effective_diameter_ratio * diameter_mm
    return {
        "effective_diameter_mm": effective_diameter,
        "effective_area_mm2": math.pi * effective_diameter**2 / 4.0,
        "bending_section_modulus_mm3": (
            math.pi * effective_diameter**3 / 32.0
        ),
        "torsion_section_modulus_mm3": (
            math.pi * effective_diameter**3 / 16.0
        ),
    }


def stress_components(
    torque_Nm: float,
    eccentricity_mm: float,
    diameter_mm: float,
    torque_coefficient: float,
    thread_torque_fraction: float,
    effective_diameter_ratio: float,
    normal_stress_concentration_factor: float = 1.0,
    torsional_stress_concentration_factor: float = 1.0,
) -> dict[str, float]:
    section = geometry(diameter_mm, effective_diameter_ratio)
    torque_Nmm = 1000.0 * torque_Nm
    pretension_N = torque_Nmm / (torque_coefficient * diameter_mm)
    axial_nominal = pretension_N / section["effective_area_mm2"]
    bending_nominal = (
        pretension_N
        * eccentricity_mm
        / section["bending_section_modulus_mm3"]
    )
    torsion_nominal = (
        thread_torque_fraction
        * torque_Nmm
        / section["torsion_section_modulus_mm3"]
    )
    normal_local = normal_stress_concentration_factor * (
        axial_nominal + bending_nominal
    )
    torsion_local = (
        torsional_stress_concentration_factor * torsion_nominal
    )
    equivalent = math.sqrt(normal_local**2 + 3.0 * torsion_local**2)
    return {
        "pretension_kN": pretension_N / 1000.0,
        "axial_nominal_MPa": axial_nominal,
        "bending_nominal_MPa": bending_nominal,
        "torsion_nominal_MPa": torsion_nominal,
        "normal_after_concentration_MPa": normal_local,
        "torsion_after_concentration_MPa": torsion_local,
        "von_mises_MPa": equivalent,
    }


def von_mises_torque_limit(
    eccentricity_mm: float,
    diameter_mm: float,
    torque_coefficient: float,
    thread_torque_fraction: float,
    yield_strength_MPa: float,
    safety_factor: float,
    effective_diameter_ratio: float,
    normal_stress_concentration_factor: float = 1.0,
    torsional_stress_concentration_factor: float = 1.0,
) -> float:
    if eccentricity_mm < 0:
        raise ValueError("eccentricity_mm must be nonnegative")
    if min(
        diameter_mm,
        torque_coefficient,
        yield_strength_MPa,
        safety_factor,
        effective_diameter_ratio,
    ) <= 0:
        raise ValueError("diameter, K, strength, safety factor and ratio must be positive")
    if thread_torque_fraction < 0:
        raise ValueError("thread_torque_fraction must be nonnegative")
    section = geometry(diameter_mm, effective_diameter_ratio)
    normal_per_Nmm = normal_stress_concentration_factor * (
        1.0 / (
            torque_coefficient
            * diameter_mm
            * section["effective_area_mm2"]
        )
        + eccentricity_mm
        / (
            torque_coefficient
            * diameter_mm
            * section["bending_section_modulus_mm3"]
        )
    )
    torsion_per_Nmm = (
        torsional_stress_concentration_factor
        * thread_torque_fraction
        / section["torsion_section_modulus_mm3"]
    )
    equivalent_per_Nmm = math.sqrt(
        normal_per_Nmm**2 + 3.0 * torsion_per_Nmm**2
    )
    allowable_stress = yield_strength_MPa / safety_factor
    return allowable_stress / equivalent_per_Nmm / 1000.0


def linspace(start: float, stop: float, count: int) -> list[float]:
    if count < 2:
        raise ValueError("count must be at least 2")
    step = (stop - start) / (count - 1)
    return [start + index * step for index in range(count)]


def bisect_torque_threshold(
    torque_limit: Callable[[float], float],
    target_torque_Nm: float,
    low_e_mm: float,
    high_e_mm: float,
    tolerance_mm: float = 1e-10,
) -> float | None:
    low_value = torque_limit(low_e_mm) - target_torque_Nm
    high_value = torque_limit(high_e_mm) - target_torque_Nm
    if low_value < 0:
        return 0.0
    if high_value >= 0:
        return None
    low = low_e_mm
    high = high_e_mm
    for _ in range(100):
        middle = 0.5 * (low + high)
        middle_value = torque_limit(middle) - target_torque_Nm
        if abs(high - low) <= tolerance_mm:
            return middle
        if middle_value >= 0:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def fixed_q2_torque_limits(
    q2_scenario: dict[str, object],
    include_indentation: bool,
) -> dict[str, float]:
    diameter = float(q2_scenario["diameter_mm"])
    coefficient = float(q2_scenario["torque_coefficient"])
    pretension_limits = q2_scenario["pretension_limits_kN"]
    modes = ["bolt_yield", "bond_failure"]
    if include_indentation:
        modes.append("tray_indentation")
    return {
        mode: coefficient
        * diameter
        * float(pretension_limits[mode])
        for mode in modes
    }


def engineering_torque_limits(
    q2_scenario: dict[str, object],
    eccentricity_mm: float,
    include_indentation: bool,
    tray_eccentric_pressure_correction: bool,
    bond_eccentricity_reduction_alpha: float,
) -> dict[str, float]:
    base_limits = fixed_q2_torque_limits(
        q2_scenario,
        include_indentation=include_indentation,
    )
    diameter = float(q2_scenario["diameter_mm"])
    if bond_eccentricity_reduction_alpha < 0:
        raise ValueError("bond reduction alpha must be nonnegative")
    bond_factor = 1.0 / (
        1.0
        + bond_eccentricity_reduction_alpha
        * eccentricity_mm
        / diameter
    )
    base_limits["bond_failure"] *= bond_factor
    if include_indentation and tray_eccentric_pressure_correction:
        tray_side = float(q2_scenario["tray_side_mm"])
        if eccentricity_mm > tray_side / 6.0:
            raise ValueError(
                "full-contact tray correction requires e <= b/6"
            )
        base_limits["tray_indentation"] /= (
            1.0 + 6.0 * eccentricity_mm / tray_side
        )
    return base_limits


def bisect_competing_modes(
    first_mode: str,
    second_mode: str,
    low_e_mm: float,
    high_e_mm: float,
    q2_scenario: dict[str, object],
    include_indentation: bool,
    config: dict[str, object],
    vm_arguments: dict[str, float],
    tolerance_mm: float = 1e-10,
) -> float | None:
    def limits(eccentricity_mm: float) -> dict[str, float]:
        return {
            **engineering_torque_limits(
                q2_scenario=q2_scenario,
                eccentricity_mm=eccentricity_mm,
                include_indentation=include_indentation,
                tray_eccentric_pressure_correction=bool(
                    config["tray_eccentric_pressure_correction"]
                ),
                bond_eccentricity_reduction_alpha=float(
                    config["bond_eccentricity_reduction_alpha"]
                ),
            ),
            "thread_von_mises": von_mises_torque_limit(
                eccentricity_mm=eccentricity_mm,
                **vm_arguments,
            ),
        }

    def difference(eccentricity_mm: float) -> float:
        current = limits(eccentricity_mm)
        return current[first_mode] - current[second_mode]

    low_value = difference(low_e_mm)
    high_value = difference(high_e_mm)
    if low_value == 0:
        return low_e_mm
    if high_value == 0:
        return high_e_mm
    if low_value * high_value > 0:
        return None
    low = low_e_mm
    high = high_e_mm
    for _ in range(100):
        middle = 0.5 * (low + high)
        middle_value = difference(middle)
        if abs(high - low) <= tolerance_mm:
            return middle
        if low_value * middle_value <= 0:
            high = middle
        else:
            low = middle
            low_value = middle_value
    return 0.5 * (low + high)


def combined_scenario(
    q2_scenario: dict[str, object],
    config: dict[str, object],
    include_indentation: bool,
) -> dict[str, object]:
    scenario_id = str(q2_scenario["scenario"])
    diameter = float(q2_scenario["diameter_mm"])
    coefficient = float(q2_scenario["torque_coefficient"])
    ratio_grid = linspace(
        float(config["eccentricity_ratio_min"]),
        float(config["eccentricity_ratio_max"]),
        int(config["eccentricity_grid_points"]),
    )
    base_limits = fixed_q2_torque_limits(
        q2_scenario,
        include_indentation=include_indentation,
    )
    base_mode = min(base_limits, key=base_limits.get)
    base_limit = base_limits[base_mode]
    vm_arguments = {
        "diameter_mm": diameter,
        "torque_coefficient": coefficient,
        "thread_torque_fraction": float(
            config["thread_torque_fraction"]
        ),
        "yield_strength_MPa": float(config["yield_strength_MPa"]),
        "safety_factor": float(config["safety_factor"]),
        "effective_diameter_ratio": float(
            config["thread_effective_diameter_ratio"]
        ),
        "normal_stress_concentration_factor": float(
            config["normal_stress_concentration_factor"]
        ),
        "torsional_stress_concentration_factor": float(
            config["torsional_stress_concentration_factor"]
        ),
    }
    curve = []
    for ratio in ratio_grid:
        eccentricity = ratio * diameter
        vm_limit = von_mises_torque_limit(
            eccentricity_mm=eccentricity,
            **vm_arguments,
        )
        engineering_limits = engineering_torque_limits(
            q2_scenario=q2_scenario,
            eccentricity_mm=eccentricity,
            include_indentation=include_indentation,
            tray_eccentric_pressure_correction=bool(
                config["tray_eccentric_pressure_correction"]
            ),
            bond_eccentricity_reduction_alpha=float(
                config["bond_eccentricity_reduction_alpha"]
            ),
        )
        all_limits = {
            **engineering_limits,
            "thread_von_mises": vm_limit,
        }
        governing_mode = min(all_limits, key=all_limits.get)
        combined_limit = all_limits[governing_mode]
        stresses = stress_components(
            torque_Nm=combined_limit,
            eccentricity_mm=eccentricity,
            diameter_mm=diameter,
            torque_coefficient=coefficient,
            thread_torque_fraction=float(
                config["thread_torque_fraction"]
            ),
            effective_diameter_ratio=float(
                config["thread_effective_diameter_ratio"]
            ),
            normal_stress_concentration_factor=float(
                config["normal_stress_concentration_factor"]
            ),
            torsional_stress_concentration_factor=float(
                config["torsional_stress_concentration_factor"]
            ),
        )
        curve.append(
            {
                "eccentricity_ratio": ratio,
                "eccentricity_mm": eccentricity,
                "engineering_torque_limits_Nm": engineering_limits,
                "engineering_envelope_Nm": min(
                    engineering_limits.values()
                ),
                "von_mises_limit_Nm": vm_limit,
                "combined_Tmax_Nm": combined_limit,
                "governing_mode": governing_mode,
                "constraint_margins_Nm": {
                    mode: limit - combined_limit
                    for mode, limit in all_limits.items()
                },
                "stress_at_combined_Tmax": stresses,
            }
        )
    switches = []
    for previous, current in zip(curve, curve[1:]):
        first_mode = str(previous["governing_mode"])
        second_mode = str(current["governing_mode"])
        if first_mode == second_mode:
            continue
        switch = bisect_competing_modes(
            first_mode=first_mode,
            second_mode=second_mode,
            low_e_mm=float(previous["eccentricity_mm"]),
            high_e_mm=float(current["eccentricity_mm"]),
            q2_scenario=q2_scenario,
            include_indentation=include_indentation,
            config=config,
            vm_arguments=vm_arguments,
        )
        switches.append(
            {
                "from_mode": first_mode,
                "to_mode": second_mode,
                "eccentricity_mm": switch,
                "eccentricity_ratio": (
                    None if switch is None else switch / diameter
                ),
            }
        )
    def combined_limit_at(eccentricity_mm: float) -> float:
        engineering_limits = engineering_torque_limits(
            q2_scenario=q2_scenario,
            eccentricity_mm=eccentricity_mm,
            include_indentation=include_indentation,
            tray_eccentric_pressure_correction=bool(
                config["tray_eccentric_pressure_correction"]
            ),
            bond_eccentricity_reduction_alpha=float(
                config["bond_eccentricity_reduction_alpha"]
            ),
        )
        vm_limit = von_mises_torque_limit(
            eccentricity_mm=eccentricity_mm,
            **vm_arguments,
        )
        return min(*engineering_limits.values(), vm_limit)

    minimum_torque_threshold = bisect_torque_threshold(
        torque_limit=combined_limit_at,
        target_torque_Nm=float(config["minimum_initial_torque_Nm"]),
        low_e_mm=ratio_grid[0] * diameter,
        high_e_mm=ratio_grid[-1] * diameter,
    )
    return {
        "scenario": scenario_id,
        "band_state": (
            "without_band"
            if include_indentation
            else "with_effective_band_conditional"
        ),
        "diameter_mm": diameter,
        "torque_coefficient": coefficient,
        "q2_torque_limits_at_zero_e_Nm": base_limits,
        "q2_governing_mode_at_zero_e": base_mode,
        "q2_envelope_at_zero_e_Nm": base_limit,
        "mode_switches": switches,
        "minimum_initial_torque_feasible_until_eccentricity_mm": (
            minimum_torque_threshold
        ),
        "minimum_initial_torque_feasible_until_eccentricity_ratio": (
            None
            if minimum_torque_threshold is None
            else minimum_torque_threshold / diameter
        ),
        "curve": curve,
    }


def typical_scenario(
    config: dict[str, object],
) -> dict[str, object]:
    typical = config["appendix_typical"]
    diameter = float(typical["diameter_mm"])
    coefficient = float(typical["torque_coefficient"])
    ratios = linspace(
        float(config["eccentricity_ratio_min"]),
        float(config["eccentricity_ratio_max"]),
        int(config["eccentricity_grid_points"]),
    )
    curve = []
    for ratio in ratios:
        eccentricity = ratio * diameter
        limit = von_mises_torque_limit(
            eccentricity_mm=eccentricity,
            diameter_mm=diameter,
            torque_coefficient=coefficient,
            thread_torque_fraction=float(
                config["thread_torque_fraction"]
            ),
            yield_strength_MPa=float(config["yield_strength_MPa"]),
            safety_factor=float(config["safety_factor"]),
            effective_diameter_ratio=float(
                config["thread_effective_diameter_ratio"]
            ),
            normal_stress_concentration_factor=float(
                config["normal_stress_concentration_factor"]
            ),
            torsional_stress_concentration_factor=float(
                config["torsional_stress_concentration_factor"]
            ),
        )
        curve.append(
            {
                "eccentricity_ratio": ratio,
                "eccentricity_mm": eccentricity,
                "Tmax_Nm": limit,
                "stress_at_Tmax": stress_components(
                    torque_Nm=limit,
                    eccentricity_mm=eccentricity,
                    diameter_mm=diameter,
                    torque_coefficient=coefficient,
                    thread_torque_fraction=float(
                        config["thread_torque_fraction"]
                    ),
                    effective_diameter_ratio=float(
                        config["thread_effective_diameter_ratio"]
                    ),
                    normal_stress_concentration_factor=float(
                        config["normal_stress_concentration_factor"]
                    ),
                    torsional_stress_concentration_factor=float(
                        config["torsional_stress_concentration_factor"]
                    ),
                ),
            }
        )
    vm_arguments = {
        "diameter_mm": diameter,
        "torque_coefficient": coefficient,
        "thread_torque_fraction": float(
            config["thread_torque_fraction"]
        ),
        "yield_strength_MPa": float(config["yield_strength_MPa"]),
        "safety_factor": float(config["safety_factor"]),
        "effective_diameter_ratio": float(
            config["thread_effective_diameter_ratio"]
        ),
        "normal_stress_concentration_factor": float(
            config["normal_stress_concentration_factor"]
        ),
        "torsional_stress_concentration_factor": float(
            config["torsional_stress_concentration_factor"]
        ),
    }
    minimum_torque_threshold = bisect_torque_threshold(
        torque_limit=lambda eccentricity_mm: von_mises_torque_limit(
            eccentricity_mm=eccentricity_mm,
            **vm_arguments,
        ),
        target_torque_Nm=float(config["minimum_initial_torque_Nm"]),
        low_e_mm=ratios[0] * diameter,
        high_e_mm=ratios[-1] * diameter,
    )
    return {
        "diameter_mm": diameter,
        "torque_coefficient": coefficient,
        "geometry": geometry(
            diameter,
            float(config["thread_effective_diameter_ratio"]),
        ),
        "minimum_initial_torque_feasible_until_eccentricity_mm": (
            minimum_torque_threshold
        ),
        "minimum_initial_torque_feasible_until_eccentricity_ratio": (
            None
            if minimum_torque_threshold is None
            else minimum_torque_threshold / diameter
        ),
        "curve": curve,
    }


def sensitivity_results(
    config: dict[str, object],
) -> list[dict[str, object]]:
    typical = config["appendix_typical"]
    diameter = float(typical["diameter_mm"])
    coefficient = float(typical["torque_coefficient"])
    sensitivity = config["sensitivity"]
    rows = []
    for ratio in sensitivity["evaluation_eccentricity_ratios"]:
        eccentricity = float(ratio) * diameter
        for effective_ratio in sensitivity[
            "thread_effective_diameter_ratios"
        ]:
            for torque_fraction in sensitivity[
                "thread_torque_fractions"
            ]:
                for concentration in sensitivity[
                    "stress_concentration_factors"
                ]:
                    limit = von_mises_torque_limit(
                        eccentricity_mm=eccentricity,
                        diameter_mm=diameter,
                        torque_coefficient=coefficient,
                        thread_torque_fraction=float(torque_fraction),
                        yield_strength_MPa=float(
                            config["yield_strength_MPa"]
                        ),
                        safety_factor=float(config["safety_factor"]),
                        effective_diameter_ratio=float(effective_ratio),
                        normal_stress_concentration_factor=float(
                            concentration
                        ),
                        torsional_stress_concentration_factor=float(
                            concentration
                        ),
                    )
                    rows.append(
                        {
                            "eccentricity_ratio": float(ratio),
                            "effective_diameter_ratio": float(
                                effective_ratio
                            ),
                            "thread_torque_fraction": float(
                                torque_fraction
                            ),
                            "stress_concentration_factor": float(
                                concentration
                            ),
                            "Tmax_Nm": limit,
                        }
                    )
    return rows


def selected_points(
    curve: Iterable[dict[str, object]],
    ratios: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5),
) -> list[dict[str, object]]:
    curve_rows = list(curve)
    return [
        min(
            curve_rows,
            key=lambda row: abs(
                float(row["eccentricity_ratio"]) - target
            ),
        )
        for target in ratios
    ]


def build_report(results: dict[str, object]) -> str:
    mode_names = {
        "bolt_yield": "杆体屈服",
        "bond_failure": "锚固粘结",
        "tray_indentation": "托盘压陷",
        "thread_von_mises": "螺纹段联合应力",
    }
    typical = results["appendix_typical_scenario"]
    lines = [
        "# 问题三偏心联合受力模型",
        "",
        "## 1. 建模口径",
        "",
        "- 内部单位统一为 N、mm、N·mm 和 MPa。",
        "- 偏心扫描采用 0 ≤ e/d ≤ 0.5。该范围是数值验证情景，不是题面给定的工程容差。",
        "- 工况 A、B 的 K 来自问题一；屈服强度 500 MPa、螺纹有效扭矩系数 0.09 和有效直径 0.85d 来自附录 3。",
        "- 附录 3 的屈服强度上限与附件表 4 的整杆屈服载荷同时保留，综合包络取两者及工程约束的最小值。",
        "",
        "## 2. 解析模型",
        "",
        "令 q = de/d。有效截面面积、抗弯截面系数和抗扭截面系数分别为 π(de)^2/4、π(de)^3/32 和 π(de)^3/16。",
        "",
        "各应力均与力矩 T 成正比，因此无需迭代即可得到联合应力上限：",
        "",
        "```text",
        "a(e) = Kt_sigma × [1/(K d Ae) + e/(K d Wb)]",
        "b    = Kt_tau × Ks/Wt",
        "Tvm(e) = sigma_allow / sqrt(a(e)^2 + 3 b^2)",
        "```",
        "",
        "式中 Tvm 的内部单位为 N·mm，展示时除以 1000 得 N·m。名义模型取两个应力集中系数为 1；引入应力集中后仍使用同一公式。",
        "",
        "综合模型为各力矩上限的下包络：杆体屈服、锚固粘结、托盘压陷和螺纹段联合应力四者取最小值。合格钢带情景仅条件性移除托盘压陷约束。",
        "",
        "对方形托盘，在 e ≤ b/6 且仍保持全接触的线性压力分布近似下，最大接触压力为平均压力的 1+6e/b 倍，因此压陷力矩上限修正为 Tindent(0)/(1+6e/b)。锚固界面折减写成 1/(1+alpha_b e/d)；由于题面没有界面偏心试验或截面参数，基准计算取 alpha_b=0，并保留该参数供后续校准。",
        "",
        "## 3. 附录 3 典型场景",
        "",
        "| e/d | e/mm | Tvm/(N·m) |",
        "|---:|---:|---:|",
    ]
    for row in selected_points(typical["curve"]):
        lines.append(
            f"| {row['eccentricity_ratio']:.2f} | "
            f"{row['eccentricity_mm']:.3f} | "
            f"{row['Tmax_Nm']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 4. 工况 A、B 综合安全包络",
            "",
            "| 工况 | 钢带状态 | e/d | Tmax/(N·m) | 主控模式 |",
            "|---|---|---:|---:|---|",
        ]
    )
    for scenario in results["combined_scenarios"]:
        for row in selected_points(scenario["curve"]):
            lines.append(
                f"| {scenario['scenario']} | "
                f"{scenario['band_state']} | "
                f"{row['eccentricity_ratio']:.2f} | "
                f"{row['combined_Tmax_Nm']:.3f} | "
                f"{mode_names[row['governing_mode']]} |"
            )
    lines.extend(["", "主控模式切换点：", ""])
    for scenario in results["combined_scenarios"]:
        if not scenario["mode_switches"]:
            lines.append(
                f"- 工况 {scenario['scenario']}、{scenario['band_state']}：扫描范围内无模式切换。"
            )
            continue
        for switch in scenario["mode_switches"]:
            lines.append(
                f"- 工况 {scenario['scenario']}、{scenario['band_state']}："
                f"e = {switch['eccentricity_mm']:.6f} mm"
                f"（e/d = {switch['eccentricity_ratio']:.6f}）时，"
                f"{mode_names[switch['from_mode']]}转为"
                f"{mode_names[switch['to_mode']]}。"
            )
    lines.extend(
        [
            "",
            "与 150 N·m 最低初始力矩的可行性比较：",
            "",
            (
                f"- 附录典型场景在 e = "
                f"{typical['minimum_initial_torque_feasible_until_eccentricity_mm']:.6f} mm"
                f"（e/d = {typical['minimum_initial_torque_feasible_until_eccentricity_ratio']:.6f}）"
                "后，理论上限低于 150 N·m。"
            ),
        ]
    )
    for scenario in results["combined_scenarios"]:
        threshold = scenario[
            "minimum_initial_torque_feasible_until_eccentricity_mm"
        ]
        if threshold is None:
            lines.append(
                f"- 工况 {scenario['scenario']}、{scenario['band_state']}："
                "扫描范围内始终不低于 150 N·m。"
            )
        else:
            lines.append(
                f"- 工况 {scenario['scenario']}、{scenario['band_state']}："
                f"e = {threshold:.6f} mm"
                f"（e/d = "
                f"{scenario['minimum_initial_torque_feasible_until_eccentricity_ratio']:.6f}）"
                "后，理论上限低于 150 N·m。"
            )
    lines.extend(
        [
            "",
            "## 5. 适用性与局限",
            "",
            "适用于准静态、小变形、线弹性条件下的趋势分析、参数筛查和名义截面初步设计。模型显式反映轴拉、偏心弯曲和施工扭转的共同作用。",
            "",
            "局限包括：螺纹牙载荷分配与接触摩擦被等效化；螺纹根部真实应力集中、局部塑性、安装冲击、疲劳、腐蚀和残余应力未被直接刻画；偏心距不能替代安装倾角；K 和 Ks 被视为常数；钢带与偏心引起的非均匀接触没有足够参数。",
            "",
            "因此名义模型结果是给定假设下的理论上限。工程应用应采用有依据的安全系数和应力集中系数，并以高力矩、偏心加载试验校准。",
            "",
        ]
    )
    return "\n".join(lines)


def main(emit_output: bool = True) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    q2_results = json.loads(Q2_RESULT_PATH.read_text(encoding="utf-8"))
    combined = []
    for scenario in q2_results["scenarios"]:
        combined.append(
            combined_scenario(
                scenario,
                config,
                include_indentation=True,
            )
        )
        combined.append(
            combined_scenario(
                scenario,
                config,
                include_indentation=False,
            )
        )
    results = {
        "metadata": {
            "config_file": str(CONFIG_PATH.relative_to(PROJECT_DIR)),
            "q2_result_file": str(Q2_RESULT_PATH.relative_to(PROJECT_DIR)),
            "python_version": platform.python_version(),
            "random_seed": config["random_seed"],
            "eccentricity_scan_is_declared_scenario_not_problem_fact": True,
        },
        "parameters": config,
        "formulas": {
            "effective_diameter_mm": "q * d",
            "effective_area_mm2": "pi * de^2 / 4",
            "bending_section_modulus_mm3": "pi * de^3 / 32",
            "torsion_section_modulus_mm3": "pi * de^3 / 16",
            "von_mises_limit_Nm": (
                "(sigma_y/n) / sqrt("
                "(Kt_sigma*(1/(K*d*Ae)+e/(K*d*Wb)))^2"
                "+3*(Kt_tau*Ks/Wt)^2) / 1000"
            ),
            "combined_limit_Nm": (
                "min(T_yield, eta_b(e)*T_bond, "
                "T_indent/(1+6e/b), T_vm(e))"
            ),
        },
        "appendix_typical_scenario": typical_scenario(config),
        "combined_scenarios": combined,
        "sensitivity": sensitivity_results(config),
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    REPORT_PATH.write_text(build_report(results), encoding="utf-8")
    if emit_output:
        print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
