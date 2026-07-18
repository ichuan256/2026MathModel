from __future__ import annotations

import hashlib
import json
import platform
import re
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
from scipy.optimize import minimize_scalar


PROJECT_DIR = Path(__file__).resolve().parents[2]
SOURCE_XLSX = PROJECT_DIR / "A-附件.xlsx"
CONFIG_PATH = PROJECT_DIR / "configs" / "q1_fit.json"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "model"
REPORT_DIR = PROJECT_DIR / "reports"
RESULT_PATH = OUTPUT_DIR / "q1_results.json"
REPORT_PATH = REPORT_DIR / "q1_model.md"

MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"

MODEL_FAMILIES = (
    "linear",
    "hinge",
    "quadratic_linear_c1",
    "quadratic_linear_c0",
    "power_linear_c0",
    "log_linear_c0",
    "cubic_linear_c0",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def column_number(cell_reference: str) -> int:
    letters = re.match(r"[A-Z]+", cell_reference).group(0)
    number = 0
    for letter in letters:
        number = number * 26 + ord(letter) - ord("A") + 1
    return number


def read_xlsx(path: Path) -> dict[str, list[list[object | None]]]:
    with zipfile.ZipFile(path) as archive:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall(f"{{{MAIN_NS}}}si"):
                shared_strings.append(
                    "".join(node.text or "" for node in item.iter(f"{{{MAIN_NS}}}t"))
                )

        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relationships = ET.fromstring(
            archive.read("xl/_rels/workbook.xml.rels")
        )
        relationship_targets = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in relationships.findall(f"{{{PACKAGE_REL_NS}}}Relationship")
        }

        sheets: dict[str, list[list[object | None]]] = {}
        for sheet in workbook.find(f"{{{MAIN_NS}}}sheets"):
            name = sheet.attrib["name"]
            relationship_id = sheet.attrib[f"{{{REL_NS}}}id"]
            target = relationship_targets[relationship_id].replace("\\", "/")
            if target.startswith("/"):
                sheet_path = target.lstrip("/")
            elif target.startswith("xl/"):
                sheet_path = target
            else:
                sheet_path = f"xl/{target}"

            sheet_root = ET.fromstring(archive.read(sheet_path))
            row_maps: dict[int, dict[int, object | None]] = {}
            max_column = 0
            for row in sheet_root.iter(f"{{{MAIN_NS}}}row"):
                row_number = int(row.attrib["r"])
                row_map: dict[int, object | None] = {}
                for cell in row.findall(f"{{{MAIN_NS}}}c"):
                    reference = cell.attrib["r"]
                    col = column_number(reference)
                    max_column = max(max_column, col)
                    cell_type = cell.attrib.get("t")
                    value_node = cell.find(f"{{{MAIN_NS}}}v")
                    if cell_type == "inlineStr":
                        inline = cell.find(f"{{{MAIN_NS}}}is")
                        value = (
                            "".join(
                                node.text or ""
                                for node in inline.iter(f"{{{MAIN_NS}}}t")
                            )
                            if inline is not None
                            else ""
                        )
                    elif value_node is None:
                        value = None
                    elif cell_type == "s":
                        value = shared_strings[int(value_node.text)]
                    elif cell_type == "b":
                        value = value_node.text == "1"
                    else:
                        numeric = float(value_node.text)
                        value = int(numeric) if numeric.is_integer() else numeric
                    row_map[col] = value
                row_maps[row_number] = row_map

            max_row = max(row_maps) if row_maps else 0
            rows = []
            for row_number in range(1, max_row + 1):
                row_map = row_maps.get(row_number, {})
                rows.append(
                    [row_map.get(col) for col in range(1, max_column + 1)]
                )
            sheets[name] = rows
        return sheets


def extract_q1_data(
    sheets: dict[str, list[list[object | None]]],
) -> tuple[dict[int, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]]]:
    standard_rows = sheets["Sheet1_标准实验数据"]
    standard: dict[int, dict[str, list[float]]] = {}
    current_diameter = None
    for row in standard_rows[1:22]:
        if row[0] is not None:
            current_diameter = int(row[0])
            standard[current_diameter] = {"torque_Nm": [], "pretension_kN": []}
        standard[current_diameter]["torque_Nm"].append(float(row[1]))
        standard[current_diameter]["pretension_kN"].append(float(row[2]))

    standard_array = {
        diameter: {
            field: np.array(values, dtype=float)
            for field, values in group.items()
        }
        for diameter, group in standard.items()
    }

    field = {}
    for key, sheet_name, last_row in (
        ("rock", "Sheet2_岩石工况数据", 13),
        ("coal", "Sheet3_煤体工况数据", 16),
    ):
        rows = sheets[sheet_name][1:last_row]
        field[key] = {
            "torque_Nm": np.array([float(row[0]) for row in rows], dtype=float),
            "pretension_kN": np.array(
                [[float(value) for value in row[1:6]] for row in rows],
                dtype=float,
            ),
        }
    return standard_array, field


def aicc(y: np.ndarray, fitted: np.ndarray, parameter_count: int) -> float:
    n = len(y)
    rss = float(np.sum((y - fitted) ** 2))
    base = n * np.log(rss / n) + 2 * parameter_count
    return float(
        base
        + 2
        * parameter_count
        * (parameter_count + 1)
        / (n - parameter_count - 1)
    )


def zero_intercept_fit(
    torque: np.ndarray, pretension: np.ndarray, diameter: int
) -> dict[str, object]:
    slope = float(torque @ pretension / (torque @ torque))
    fitted = slope * torque
    residual = pretension - fitted
    rss = float(residual @ residual)
    n = len(torque)
    slope_se = float(np.sqrt((rss / (n - 1)) / (torque @ torque)))
    t_critical_df6 = 2.446912
    slope_ci = (
        slope - t_critical_df6 * slope_se,
        slope + t_critical_df6 * slope_se,
    )
    k_value = 1 / (diameter * slope)
    k_ci = (
        1 / (diameter * slope_ci[1]),
        1 / (diameter * slope_ci[0]),
    )
    loo_predictions = []
    for held_out in range(n):
        keep = np.arange(n) != held_out
        loo_slope = float(
            torque[keep] @ pretension[keep] / (torque[keep] @ torque[keep])
        )
        loo_predictions.append(loo_slope * torque[held_out])
    loo_predictions = np.array(loo_predictions)
    tss = float(np.sum((pretension - np.mean(pretension)) ** 2))
    return {
        "diameter_mm": diameter,
        "formula_P_from_T": {
            "intercept_kN": 0.0,
            "slope_kN_per_Nm": slope,
        },
        "formula_T_from_P": {
            "slope_Nm_per_kN": 1 / slope,
        },
        "K": k_value,
        "K_95pct_CI": list(k_ci),
        "slope_95pct_CI": list(slope_ci),
        "train_R2_centered": 1 - rss / tss,
        "train_RMSE_kN": float(np.sqrt(np.mean(residual**2))),
        "leave_one_out_RMSE_kN": float(
            np.sqrt(np.mean((pretension - loo_predictions) ** 2))
        ),
        "residuals_kN": residual.tolist(),
    }


def global_q11_model_comparison(
    standard: dict[int, dict[str, np.ndarray]],
) -> list[dict[str, object]]:
    diameters = sorted(standard)
    torque = np.concatenate([standard[d]["torque_Nm"] for d in diameters])
    y = np.concatenate([standard[d]["pretension_kN"] for d in diameters])
    diameter = np.concatenate(
        [
            np.full(len(standard[d]["torque_Nm"]), d, dtype=float)
            for d in diameters
        ]
    )

    designs = {
        "common_K_zero_intercept": (torque / diameter)[:, None],
        "separate_K_zero_intercept": np.column_stack(
            [(diameter == d) * torque for d in diameters]
        ),
        "separate_K_group_intercepts": np.column_stack(
            [(diameter == d).astype(float) for d in diameters]
            + [(diameter == d) * torque for d in diameters]
        ),
        "separate_quadratic": np.column_stack(
            [(diameter == d).astype(float) for d in diameters]
            + [(diameter == d) * torque for d in diameters]
            + [(diameter == d) * torque**2 for d in diameters]
        ),
    }

    results = []
    for name, design in designs.items():
        beta = np.linalg.lstsq(design, y, rcond=None)[0]
        fitted = design @ beta
        residual = y - fitted
        rss = float(residual @ residual)
        loo = np.empty_like(y)
        for held_out in range(len(y)):
            keep = np.arange(len(y)) != held_out
            fold_beta = np.linalg.lstsq(design[keep], y[keep], rcond=None)[0]
            loo[held_out] = design[held_out] @ fold_beta
        parameter_count = design.shape[1]
        bic = len(y) * np.log(rss / len(y)) + parameter_count * np.log(len(y))
        results.append(
            {
                "model": name,
                "parameter_count": parameter_count,
                "BIC": float(bic),
                "AICc": aicc(y, fitted, parameter_count),
                "train_RMSE_kN": float(np.sqrt(np.mean(residual**2))),
                "leave_one_out_RMSE_kN": float(
                    np.sqrt(np.mean((y - loo) ** 2))
                ),
            }
        )
    return results


def make_piecewise_design(
    torque: np.ndarray,
    point_id: np.ndarray,
    point_count: int,
    family: str,
    breakpoint: float | None = None,
    power_exponent: float | None = None,
) -> np.ndarray:
    if family == "linear":
        local = np.column_stack([np.ones_like(torque), torque])
    elif family == "hinge":
        local = np.column_stack(
            [np.ones_like(torque), torque, np.maximum(torque - breakpoint, 0)]
        )
    elif family == "quadratic_linear_c1":
        curved = np.where(
            torque <= breakpoint,
            torque**2,
            2 * breakpoint * torque - breakpoint**2,
        )
        local = np.column_stack([np.ones_like(torque), torque, curved])
    elif family == "quadratic_linear_c0":
        before = torque <= breakpoint
        local = np.column_stack(
            [
                np.ones_like(torque),
                np.where(before, torque, breakpoint),
                np.where(before, torque**2, breakpoint**2),
                np.where(before, 0, torque - breakpoint),
            ]
        )
    elif family == "power_linear_c0":
        if power_exponent is None:
            raise ValueError("power_exponent is required for power model")
        capped = np.minimum(torque, breakpoint)
        local = np.column_stack(
            [
                np.ones_like(torque),
                capped**power_exponent,
                np.maximum(torque - breakpoint, 0),
            ]
        )
    elif family == "log_linear_c0":
        capped = np.minimum(torque, breakpoint)
        local = np.column_stack(
            [
                np.ones_like(torque),
                np.log1p(capped),
                np.maximum(torque - breakpoint, 0),
            ]
        )
    elif family == "cubic_linear_c0":
        capped = np.minimum(torque, breakpoint)
        local = np.column_stack(
            [
                np.ones_like(torque),
                capped,
                capped**2,
                capped**3,
                np.maximum(torque - breakpoint, 0),
            ]
        )
    else:
        raise ValueError(f"Unknown model family: {family}")

    width = local.shape[1]
    design = np.zeros((len(torque), point_count * width))
    for row in range(len(torque)):
        start = point_id[row] * width
        design[row, start : start + width] = local[row]
    return design


def fit_piecewise(
    torque: np.ndarray,
    point_id: np.ndarray,
    y: np.ndarray,
    point_count: int,
    family: str,
    breakpoint: float | None = None,
    power_exponent: float | None = None,
) -> dict[str, object]:
    design = make_piecewise_design(
        torque,
        point_id,
        point_count,
        family,
        breakpoint,
        power_exponent,
    )
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    fitted = design @ beta
    residual = y - fitted
    rss = float(residual @ residual)
    n = len(y)
    parameter_count = (
        design.shape[1]
        + (family != "linear")
        + (family == "power_linear_c0")
    )
    bic = n * np.log(rss / n) + parameter_count * np.log(n)
    return {
        "beta": beta,
        "fitted": fitted,
        "residual": residual,
        "rss": rss,
        "BIC": float(bic),
        "AICc": aicc(y, fitted, int(parameter_count)),
        "parameter_count": int(parameter_count),
    }


def valid_breakpoints(
    unique_torque: np.ndarray,
    grid: np.ndarray,
    minimum_levels_each_side: int,
) -> list[float]:
    return [
        float(value)
        for value in grid
        if np.sum(unique_torque <= value) >= minimum_levels_each_side
        and np.sum(unique_torque > value) >= minimum_levels_each_side
    ]


def profile_piecewise(
    torque: np.ndarray,
    point_id: np.ndarray,
    y: np.ndarray,
    point_count: int,
    family: str,
    grid: np.ndarray,
    minimum_levels_each_side: int,
    power_exponent_bounds: tuple[float, float],
    power_exponent_tolerance: float,
) -> tuple[dict[str, object], list[dict[str, float]]]:
    candidates = valid_breakpoints(
        np.unique(torque), grid, minimum_levels_each_side
    )
    profile = []
    fits = {}
    for breakpoint in candidates:
        power_exponent = None
        if family == "power_linear_c0":
            exponent_search = minimize_scalar(
                lambda exponent: fit_piecewise(
                    torque,
                    point_id,
                    y,
                    point_count,
                    family,
                    breakpoint,
                    float(exponent),
                )["rss"],
                bounds=power_exponent_bounds,
                method="bounded",
                options={"xatol": power_exponent_tolerance},
            )
            power_exponent = float(exponent_search.x)
        fit = fit_piecewise(
            torque,
            point_id,
            y,
            point_count,
            family,
            breakpoint,
            power_exponent,
        )
        fit["power_exponent"] = power_exponent
        fits[(breakpoint, power_exponent)] = fit
        profile.append(
            {
                "breakpoint_Nm": breakpoint,
                "power_exponent": power_exponent,
                "BIC": fit["BIC"],
                "AICc": fit["AICc"],
                "RSS": fit["rss"],
            }
        )
    best_row = min(profile, key=lambda row: row["BIC"])
    return (
        fits[
            (
                best_row["breakpoint_Nm"],
                best_row["power_exponent"],
            )
        ],
        profile,
    )


def grouped_leave_one_level_out(
    torque_levels: np.ndarray,
    response: np.ndarray,
    family: str,
    grid: np.ndarray,
    minimum_levels_each_side: int,
    power_exponent_bounds: tuple[float, float],
    power_exponent_tolerance: float,
) -> dict[str, object]:
    level_count, point_count = response.shape
    predictions = np.empty_like(response)
    chosen_breakpoints = []
    chosen_power_exponents = []
    for held_out in range(level_count):
        keep = np.arange(level_count) != held_out
        train_torque = np.repeat(torque_levels[keep], point_count)
        train_point = np.tile(np.arange(point_count), np.sum(keep))
        train_y = response[keep].reshape(-1)
        if family == "linear":
            fit = fit_piecewise(
                train_torque, train_point, train_y, point_count, family
            )
            breakpoint = None
            power_exponent = None
        else:
            fit, profile = profile_piecewise(
                train_torque,
                train_point,
                train_y,
                point_count,
                family,
                grid,
                minimum_levels_each_side,
                power_exponent_bounds,
                power_exponent_tolerance,
            )
            best_row = min(profile, key=lambda row: row["BIC"])
            breakpoint = best_row["breakpoint_Nm"]
            power_exponent = best_row["power_exponent"]
            chosen_breakpoints.append(breakpoint)
            if power_exponent is not None:
                chosen_power_exponents.append(power_exponent)
        test_torque = np.repeat(torque_levels[held_out], point_count)
        test_point = np.arange(point_count)
        test_design = make_piecewise_design(
            test_torque,
            test_point,
            point_count,
            family,
            breakpoint,
            power_exponent,
        )
        predictions[held_out] = test_design @ fit["beta"]
    residual = response - predictions
    return {
        "grouped_LOO_RMSE_kN": float(np.sqrt(np.mean(residual**2))),
        "grouped_LOO_MAE_kN": float(np.mean(np.abs(residual))),
        "fold_breakpoint_range_Nm": (
            [
                float(np.min(chosen_breakpoints)),
                float(np.max(chosen_breakpoints)),
            ]
            if chosen_breakpoints
            else None
        ),
        "fold_breakpoint_median_Nm": (
            float(np.median(chosen_breakpoints))
            if chosen_breakpoints
            else None
        ),
        "fold_power_exponent_range": (
            [
                float(np.min(chosen_power_exponents)),
                float(np.max(chosen_power_exponents)),
            ]
            if chosen_power_exponents
            else None
        ),
    }


def tail_diagnostics(
    torque_levels: np.ndarray,
    response: np.ndarray,
    threshold: float,
) -> list[dict[str, float]]:
    tail = torque_levels >= threshold
    diagnostics = []
    for point in range(response.shape[1]):
        design = np.column_stack([np.ones(np.sum(tail)), torque_levels[tail]])
        y = response[tail, point]
        beta = np.linalg.lstsq(design, y, rcond=None)[0]
        fitted = design @ beta
        residual = y - fitted
        rss = float(residual @ residual)
        tss = float(np.sum((y - np.mean(y)) ** 2))
        diagnostics.append(
            {
                "point_id": point + 1,
                "intercept_kN": float(beta[0]),
                "slope_kN_per_Nm": float(beta[1]),
                "RMSE_kN": float(np.sqrt(np.mean(residual**2))),
                "R2": float(1 - rss / tss),
            }
        )
    return diagnostics


def formulas_from_qlinear_c0(
    beta: np.ndarray, point_count: int, breakpoint: float
) -> dict[str, object]:
    coefficients = beta.reshape(point_count, 4)
    formulas = []
    for point, (a, b, c, slope_after) in enumerate(coefficients, start=1):
        value_at_breakpoint = a + b * breakpoint + c * breakpoint**2
        post_intercept = value_at_breakpoint - slope_after * breakpoint
        formulas.append(
            {
                "point_id": point,
                "before": {
                    "intercept_kN": float(a),
                    "linear_kN_per_Nm": float(b),
                    "quadratic_kN_per_Nm2": float(c),
                },
                "after": {
                    "intercept_kN": float(post_intercept),
                    "slope_kN_per_Nm": float(slope_after),
                },
                "continuity_value_kN": float(value_at_breakpoint),
            }
        )
    mean_coefficients = np.mean(coefficients, axis=0)
    a, b, c, slope_after = mean_coefficients
    value_at_breakpoint = a + b * breakpoint + c * breakpoint**2
    return {
        "point_formulas": formulas,
        "mean_formula": {
            "before": {
                "intercept_kN": float(a),
                "linear_kN_per_Nm": float(b),
                "quadratic_kN_per_Nm2": float(c),
            },
            "after": {
                "intercept_kN": float(
                    value_at_breakpoint - slope_after * breakpoint
                ),
                "slope_kN_per_Nm": float(slope_after),
            },
            "continuity_value_kN": float(value_at_breakpoint),
        },
    }


def formulas_from_power_c0(
    beta: np.ndarray,
    point_count: int,
    breakpoint: float,
    exponent: float,
) -> dict[str, object]:
    coefficients = beta.reshape(point_count, 3)
    formulas = []
    for point, (a, scale, slope_after) in enumerate(
        coefficients, start=1
    ):
        value_at_breakpoint = a + scale * breakpoint**exponent
        post_intercept = value_at_breakpoint - slope_after * breakpoint
        formulas.append(
            {
                "point_id": point,
                "before": {
                    "intercept_kN": float(a),
                    "scale_kN_per_Nm_power": float(scale),
                    "exponent": float(exponent),
                },
                "after": {
                    "intercept_kN": float(post_intercept),
                    "slope_kN_per_Nm": float(slope_after),
                },
                "continuity_value_kN": float(value_at_breakpoint),
            }
        )
    a, scale, slope_after = np.mean(coefficients, axis=0)
    value_at_breakpoint = a + scale * breakpoint**exponent
    return {
        "point_formulas": formulas,
        "mean_formula": {
            "before": {
                "intercept_kN": float(a),
                "scale_kN_per_Nm_power": float(scale),
                "exponent": float(exponent),
            },
            "after": {
                "intercept_kN": float(
                    value_at_breakpoint - slope_after * breakpoint
                ),
                "slope_kN_per_Nm": float(slope_after),
            },
            "continuity_value_kN": float(value_at_breakpoint),
        },
    }


def analyze_field_dataset(
    dataset: dict[str, np.ndarray], config: dict[str, object]
) -> dict[str, object]:
    torque_levels = dataset["torque_Nm"]
    response = dataset["pretension_kN"]
    level_count, point_count = response.shape
    torque = np.repeat(torque_levels, point_count)
    point_id = np.tile(np.arange(point_count), level_count)
    y = response.reshape(-1)
    step = float(config["breakpoint_grid_step_Nm"])
    grid = np.arange(torque_levels[2], torque_levels[-3] + step / 2, step)
    minimum_levels = int(config["minimum_torque_levels_each_side"])
    delta_bic_limit = float(config["confidence_profile_delta_BIC"])
    power_exponent_bounds = tuple(
        float(value) for value in config["power_exponent_bounds"]
    )
    power_exponent_tolerance = float(config["power_exponent_tolerance"])

    comparisons = []
    selected_fits = {}
    profiles = {}
    for family in MODEL_FAMILIES:
        if family == "linear":
            fit = fit_piecewise(torque, point_id, y, point_count, family)
            breakpoint = None
            profile_range = None
        else:
            fit, profile = profile_piecewise(
                torque,
                point_id,
                y,
                point_count,
                family,
                grid,
                minimum_levels,
                power_exponent_bounds,
                power_exponent_tolerance,
            )
            profiles[family] = profile
            best = min(profile, key=lambda row: row["BIC"])
            breakpoint = best["breakpoint_Nm"]
            power_exponent = best["power_exponent"]
            profile_values = [
                row["breakpoint_Nm"]
                for row in profile
                if row["BIC"] - best["BIC"] <= delta_bic_limit
            ]
            profile_range = [
                float(min(profile_values)),
                float(max(profile_values)),
            ]
        if family == "linear":
            power_exponent = None
        selected_fits[family] = (fit, breakpoint, power_exponent)
        cv = grouped_leave_one_level_out(
            torque_levels,
            response,
            family,
            grid,
            minimum_levels,
            power_exponent_bounds,
            power_exponent_tolerance,
        )
        candidate_threshold = (
            float(torque_levels[0])
            if breakpoint is None
            else float(torque_levels[torque_levels >= breakpoint][0])
        )
        candidate_tail = tail_diagnostics(
            torque_levels, response, candidate_threshold
        )
        comparisons.append(
            {
                "model": family,
                "breakpoint_Nm": breakpoint,
                "power_exponent": power_exponent,
                "parameter_count": fit["parameter_count"],
                "BIC": fit["BIC"],
                "AICc": fit["AICc"],
                "train_RMSE_kN": float(np.sqrt(fit["rss"] / len(y))),
                "practical_threshold_Nm": candidate_threshold,
                "tail_min_R2": float(
                    min(row["R2"] for row in candidate_tail)
                ),
                "profile_delta_BIC_le_2_range_Nm": profile_range,
                **cv,
            }
        )

    minimum_tail_r2 = float(config["minimum_tail_R2_each_point"])
    eligible = [
        row
        for row in comparisons
        if row["breakpoint_Nm"] is not None
        and row["tail_min_R2"] >= minimum_tail_r2
    ]
    if not eligible:
        raise ValueError("No piecewise candidate passes the tail R2 constraint")
    selected = min(eligible, key=lambda row: row["BIC"])
    selected_fit, breakpoint, selected_power_exponent = selected_fits[
        selected["model"]
    ]
    practical_threshold = float(
        torque_levels[torque_levels >= breakpoint][0]
    )
    robustness_upper = max(
        selected["profile_delta_BIC_le_2_range_Nm"][1],
        selected["fold_breakpoint_range_Nm"][1],
    )
    conservative_threshold = float(
        torque_levels[torque_levels >= robustness_upper][0]
    )
    quadratic_fit, quadratic_breakpoint, _ = selected_fits[
        "quadratic_linear_c0"
    ]
    quadratic_formula = formulas_from_qlinear_c0(
        quadratic_fit["beta"], point_count, quadratic_breakpoint
    )
    if selected["model"] == "power_linear_c0":
        formula = formulas_from_power_c0(
            selected_fit["beta"],
            point_count,
            breakpoint,
            selected_power_exponent,
        )
    elif selected["model"] == "quadratic_linear_c0":
        formula = quadratic_formula
    else:
        raise ValueError(
            f"Formula export is not implemented for {selected['model']}"
        )
    return {
        "selected_model": selected["model"],
        "selected_power_exponent": selected_power_exponent,
        "estimated_breakpoint_Nm": breakpoint,
        "practical_tested_threshold_Nm": practical_threshold,
        "conservative_tested_threshold_Nm": conservative_threshold,
        "model_comparison": comparisons,
        "quadratic_linear_c0_formula": quadratic_formula,
        "formula": formula,
        "tail_diagnostics_at_practical_threshold": tail_diagnostics(
            torque_levels, response, practical_threshold
        ),
    }


def format_number(value: float, digits: int = 6) -> str:
    text = f"{value:.{digits}f}"
    return text if digits == 0 else text.rstrip("0").rstrip(".")


def build_report(results: dict[str, object]) -> str:
    q11 = results["question_1_1"]
    rock = results["question_1_2"]["rock"]
    coal = results["question_1_2"]["coal"]
    lines = [
        "# 问题 1 模型建立与求解",
        "",
        "## 1. 数据与运行口径",
        "",
        "- 原始输入：`A-附件.xlsx`，只读。",
        "- 问题 1.1：T 为实验设定量，P 为响应量；按直径分别拟合过原点线性模型。",
        "- 问题 1.2：岩石、煤体分别估计一个共享临界点，5 个测点保留各自参数。",
        "- 断点模型：临界点前比较二次、幂函数、对数和三次，临界点后线性，并在临界点处连续。",
        "- 断点选择：先要求尾段各测点 R² 不低于 0.95，再按 BIC 最小选择；同时报告 AICc、整组留一误差和断点稳定范围。",
        "- 算法为确定性网格搜索，不使用随机种子。",
        "",
        "## 2. 问题 1.1 拟合结果",
        "",
        "主模型为：`P = beta(d) × T`，且 `K = 1 / (d × beta(d))`。",
        "",
        "| 直径 / mm | P(T) 公式 | T(P) 公式 | K | K 的 95% 区间 | 留一 RMSE / kN |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for row in q11["diameter_fits"]:
        slope = row["formula_P_from_T"]["slope_kN_per_Nm"]
        inverse = row["formula_T_from_P"]["slope_Nm_per_kN"]
        k_low, k_high = row["K_95pct_CI"]
        lines.append(
            f"| {row['diameter_mm']} | P = {format_number(slope)}T | "
            f"T = {format_number(inverse)}P | {format_number(row['K'])} | "
            f"[{format_number(k_low)}, {format_number(k_high)}] | "
            f"{format_number(row['leave_one_out_RMSE_kN'], 3)} |"
        )

    lines.extend(
        [
            "",
            "## 3. 问题 1.2 分段模型",
            "",
            "对每个测点 j 使用：",
            "",
            "```text",
            "T ≤ Tc：Pj = aj + bj × T + cj × T²",
            "T > Tc：Pj = Aj + sj × T",
            "```",
            "",
            "两段在 Tc 处连续。下表中的平均公式是 5 个测点公式的等权平均，断点仍由全部测点联合估计。",
            "",
            "| 工况 | 估计断点 / N·m | ΔBIC≤2 区间 / N·m | 留一断点范围 / N·m | 实测档位 | 保守档位 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for label, result in (("岩石", rock), ("煤体", coal)):
        selected = next(
            row
            for row in result["model_comparison"]
            if row["model"] == result["selected_model"]
        )
        profile = selected["profile_delta_BIC_le_2_range_Nm"]
        fold = selected["fold_breakpoint_range_Nm"]
        lines.append(
            f"| {label} | {format_number(result['estimated_breakpoint_Nm'], 1)} | "
            f"{format_number(profile[0], 1)}–{format_number(profile[1], 1)} | "
            f"{format_number(fold[0], 1)}–{format_number(fold[1], 1)} | "
            f"{format_number(result['practical_tested_threshold_Nm'], 0)} | "
            f"{format_number(result['conservative_tested_threshold_Nm'], 0)} |"
        )

    lines.extend(["", "### 3.1 平均拟合公式", ""])
    for label, result in (("岩石", rock), ("煤体", coal)):
        tc = result["estimated_breakpoint_Nm"]
        mean_formula = result["formula"]["mean_formula"]
        before = mean_formula["before"]
        after = mean_formula["after"]
        if result["selected_model"] == "power_linear_c0":
            before_formula = (
                f"T ≤ Tc：P_mean = "
                f"{format_number(before['intercept_kN'])} + "
                f"{format_number(before['scale_kN_per_Nm_power'], 9)} × "
                f"T^{format_number(before['exponent'], 6)}"
            )
        else:
            before_formula = (
                f"T ≤ Tc：P_mean = "
                f"{format_number(before['intercept_kN'])} "
                f"+ {format_number(before['linear_kN_per_Nm'])} × T "
                f"+ ({format_number(before['quadratic_kN_per_Nm2'], 9)}) × T²"
            )
        lines.extend(
            [
                f"**{label}工况，Tc = {format_number(tc, 1)} N·m：**",
                "",
                "```text",
                before_formula,
                f"T > Tc：P_mean = {format_number(after['intercept_kN'])} "
                f"+ {format_number(after['slope_kN_per_Nm'])} × T",
                "```",
                "",
            ]
        )

    lines.extend(
        [
            "## 4. 模型比较摘要",
            "",
            "| 工况 | 模型 | BIC | 整组留一 RMSE / kN |",
            "|---|---|---:|---:|",
        ]
    )
    name_map = {
        "linear": "整段线性",
        "hinge": "连续折线",
        "quadratic_linear_c1": "前二次—后线性，斜率连续",
        "quadratic_linear_c0": "前二次—后线性，仅函数连续",
        "power_linear_c0": "前幂函数—后线性",
        "log_linear_c0": "前对数—后线性",
        "cubic_linear_c0": "前三次—后线性",
    }
    for label, result in (("岩石", rock), ("煤体", coal)):
        for row in result["model_comparison"]:
            lines.append(
                f"| {label} | {name_map[row['model']]} | "
                f"{format_number(row['BIC'], 2)} | "
                f"{format_number(row['grouped_LOO_RMSE_kN'], 3)} |"
            )

    lines.extend(
        [
            "",
            "## 5. 限制",
            "",
            "- 当前项目没有完整 `data-audit` 报告，本次仅对问题 1 所需工作表进行结构和数值范围检查。",
            f"- 临界点搜索步长为 1 N·m，但原实验力矩档位间隔为 25 N·m；{format_number(rock['estimated_breakpoint_Nm'], 0)} 和 {format_number(coal['estimated_breakpoint_Nm'], 0)} N·m 是模型断点估计，不应冒充同精度的实验观测。",
            "- 5 个测点的物理位置和相关结构未说明，因此共享断点、测点特有系数是最小稳健处理。",
            "- 问题 1.1 的 18 mm 组波动较大，K 的区间明显宽于其他两组。",
            "",
            "## 6. 复现",
            "",
            "从项目根目录运行：",
            "",
            "```text",
            "python xiaosai/2026A题/src/models/q1_fit.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    sheets = read_xlsx(SOURCE_XLSX)
    standard, field = extract_q1_data(sheets)

    required_diameters = {18, 20, 22}
    if set(standard) != required_diameters:
        raise ValueError(f"Unexpected diameter groups: {sorted(standard)}")
    for diameter, group in standard.items():
        if len(group["torque_Nm"]) != 7:
            raise ValueError(f"Diameter {diameter} does not have 7 rows")
    if field["rock"]["pretension_kN"].shape != (12, 5):
        raise ValueError("Rock data shape is not 12 × 5")
    if field["coal"]["pretension_kN"].shape != (15, 5):
        raise ValueError("Coal data shape is not 15 × 5")

    q11_fits = [
        zero_intercept_fit(
            standard[diameter]["torque_Nm"],
            standard[diameter]["pretension_kN"],
            diameter,
        )
        for diameter in sorted(standard)
    ]
    results = {
        "metadata": {
            "source_file": str(SOURCE_XLSX.relative_to(PROJECT_DIR)),
            "source_sha256": sha256_file(SOURCE_XLSX),
            "config": config,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "random_seed": None,
        },
        "question_1_1": {
            "selected_model": "separate_diameter_zero_intercept_OLS",
            "diameter_fits": q11_fits,
            "model_comparison": global_q11_model_comparison(standard),
        },
        "question_1_2": {
            condition: analyze_field_dataset(dataset, config)
            for condition, dataset in field.items()
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    REPORT_PATH.write_text(build_report(results), encoding="utf-8")

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
