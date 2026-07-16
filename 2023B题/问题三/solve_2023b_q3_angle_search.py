from __future__ import annotations

import csv
import html
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


NM = 1852.0
THETA = math.radians(120)
GAMMA = THETA / 2
ALPHA = math.radians(1.5)
CENTER_DEPTH = 110.0

X_MIN, X_MAX = -2 * NM, 2 * NM
Y_MIN, Y_MAX = -1 * NM, 1 * NM

ETA_TARGET = 0.10
PHI_MIN_DEG, PHI_MAX_DEG = 80.0, 100.0
COARSE_STEP_DEG = 0.1
GOLDEN_HALF_WIDTH_DEG = 0.2
GOLDEN_ITERATIONS = 80

BASE_DIR = Path(__file__).resolve().parent
PROBLEM_DIR = BASE_DIR
LINES_CSV_PATH = PROBLEM_DIR / "q3_angle_search_lines.csv"
HISTORY_CSV_PATH = PROBLEM_DIR / "q3_angle_search_history.csv"
PNG_PATH = PROBLEM_DIR / "q3_angle_search_layout.png"
SVG_PATH = PROBLEM_DIR / "q3_angle_search_layout.svg"


def depth_at_x(x: float) -> float:
    return CENTER_DEPTH - x * math.tan(ALPHA)


def rectangle_vertices() -> list[tuple[float, float]]:
    return [
        (X_MIN, Y_MIN),
        (X_MIN, Y_MAX),
        (X_MAX, Y_MIN),
        (X_MAX, Y_MAX),
    ]


def rectangle_polygon() -> list[tuple[float, float]]:
    return [
        (X_MIN, Y_MIN),
        (X_MAX, Y_MIN),
        (X_MAX, Y_MAX),
        (X_MIN, Y_MAX),
    ]


def unit_vectors(phi_deg: float) -> tuple[tuple[float, float], tuple[float, float]]:
    phi = math.radians(phi_deg)
    along = (math.cos(phi), math.sin(phi))
    normal = (-math.sin(phi), math.cos(phi))
    return along, normal


def projection_bounds(normal: tuple[float, float]) -> tuple[float, float]:
    values = [x * normal[0] + y * normal[1] for x, y in rectangle_vertices()]
    return min(values), max(values)


def clip_polygon_by_half_plane(
    polygon: list[tuple[float, float]],
    normal: tuple[float, float],
    limit: float,
    keep_less: bool,
) -> list[tuple[float, float]]:
    def signed(point: tuple[float, float]) -> float:
        value = point[0] * normal[0] + point[1] * normal[1] - limit
        return value if keep_less else -value

    output = []
    for start, end in zip(polygon, polygon[1:] + polygon[:1]):
        ds, de = signed(start), signed(end)
        start_inside, end_inside = ds <= 1e-10, de <= 1e-10
        if start_inside:
            output.append(start)
        if start_inside != end_inside:
            ratio = ds / (ds - de)
            output.append(
                (
                    start[0] + ratio * (end[0] - start[0]),
                    start[1] + ratio * (end[1] - start[1]),
                )
            )
    return output


def covered_polygon(
    left: float,
    right: float,
    normal: tuple[float, float],
) -> list[tuple[float, float]]:
    polygon = rectangle_polygon()
    polygon = clip_polygon_by_half_plane(polygon, normal, right, keep_less=True)
    return clip_polygon_by_half_plane(polygon, normal, left, keep_less=False)


def required_survey_segment(
    r: float,
    left: float,
    right: float,
    along: tuple[float, float],
    normal: tuple[float, float],
) -> tuple[float, float, float, float, float] | None:
    polygon = covered_polygon(left, right, normal)
    if not polygon:
        return None
    t_values = [x * along[0] + y * along[1] for x, y in polygon]
    t_low, t_high = min(t_values), max(t_values)
    x1 = r * normal[0] + t_low * along[0]
    y1 = r * normal[1] + t_low * along[1]
    x2 = r * normal[0] + t_high * along[0]
    y2 = r * normal[1] + t_high * along[1]
    return x1, y1, x2, y2, t_high - t_low


def line_rectangle_segment(
    r: float, along: tuple[float, float], normal: tuple[float, float]
) -> tuple[float, float, float, float] | None:
    t_low = -1e30
    t_high = 1e30

    for coord, lo, hi in ((0, X_MIN, X_MAX), (1, Y_MIN, Y_MAX)):
        base = r * normal[coord]
        direction = along[coord]
        if abs(direction) < 1e-12:
            if base < lo or base > hi:
                return None
            continue
        a = (lo - base) / direction
        b = (hi - base) / direction
        t_low = max(t_low, min(a, b))
        t_high = min(t_high, max(a, b))

    if t_low > t_high:
        return None
    x1 = r * normal[0] + t_low * along[0]
    y1 = r * normal[1] + t_low * along[1]
    x2 = r * normal[0] + t_high * along[0]
    y2 = r * normal[1] + t_high * along[1]
    return x1, y1, x2, y2


def min_depth_on_line(r: float, along: tuple[float, float], normal: tuple[float, float]) -> float:
    segment = line_rectangle_segment(r, along, normal)
    if segment is None:
        return max(1.0, depth_at_x(r * normal[0]))
    x1, _, x2, _ = segment
    return min(depth_at_x(x1), depth_at_x(x2))


def interval_at(
    r: float, along: tuple[float, float], normal: tuple[float, float]
) -> tuple[float, float, float, float, float]:
    alpha_perp = math.atan(math.tan(ALPHA) * abs(normal[0]))
    depth = min_depth_on_line(r, along, normal)
    common = depth * math.sin(GAMMA) * math.cos(alpha_perp)
    deep_half = common / math.cos(GAMMA + alpha_perp)
    shallow_half = common / math.cos(GAMMA - alpha_perp)
    left = r - shallow_half
    right = r + deep_half
    return left, right, shallow_half + deep_half, depth, math.degrees(alpha_perp)


def bisect_root(func, lo: float, hi: float, iterations: int = 80) -> float:
    flo = func(lo)
    fhi = func(hi)
    if flo == 0:
        return lo
    if fhi == 0:
        return hi
    if flo * fhi > 0:
        raise ValueError("root not bracketed")

    for _ in range(iterations):
        mid = (lo + hi) / 2
        fmid = func(mid)
        if flo * fmid <= 0:
            hi = mid
            fhi = fmid
        else:
            lo = mid
            flo = fmid
    return (lo + hi) / 2


def bracketed_root(
    func,
    lo: float,
    hi: float,
    lower_limit: float,
    upper_limit: float,
    expansions: int = 80,
) -> float:
    flo = func(lo)
    fhi = func(hi)
    step = hi - lo
    for _ in range(expansions):
        if flo == 0:
            return lo
        if fhi == 0:
            return hi
        if flo * fhi < 0:
            return bisect_root(func, lo, hi)
        if abs(flo) < abs(fhi):
            lo = max(lower_limit, lo - step)
            flo = func(lo)
        else:
            hi = min(upper_limit, hi + step)
            fhi = func(hi)
        step *= 1.35
        if lo <= lower_limit and hi >= upper_limit and flo * fhi > 0:
            break
    raise ValueError("root not bracketed")


def build_lines(phi_deg: float, eta_target: float = ETA_TARGET) -> list[dict[str, float]]:
    along, normal = unit_vectors(phi_deg)
    r_min, r_max = projection_bounds(normal)
    pad = 2200.0

    def first_objective(r: float) -> float:
        _, right, _, _, _ = interval_at(r, along, normal)
        return right - r_max

    try:
        r = bracketed_root(
            first_objective,
            r_max - pad,
            r_max + pad,
            r_max - 5 * pad,
            r_max + 5 * pad,
        )
    except ValueError:
        return []

    lines = []
    for _ in range(400):
        left, right, width, depth, alpha_perp_deg = interval_at(r, along, normal)
        segment = required_survey_segment(r, left, right, along, normal)
        if segment is None:
            return []

        x1, y1, x2, y2, length = segment
        lines.append(
            {
                "r": r,
                "left": left,
                "right": right,
                "width": width,
                "depth": depth,
                "alpha_perp_deg": alpha_perp_deg,
                "length": length,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )

        if left <= r_min:
            break

        prev_left = left
        prev_width = width

        def next_objective(candidate_r: float) -> float:
            _, cand_right, cand_width, _, _ = interval_at(candidate_r, along, normal)
            overlap = cand_right - prev_left
            return overlap - eta_target * min(prev_width, cand_width)

        try:
            r = bracketed_root(
                next_objective,
                r - max(width * 2, 1000.0),
                r - 1.0,
                r_min - 5 * pad,
                r - 1.0,
            )
        except ValueError:
            return []

    return lines


def evaluate_phi(phi_deg: float) -> tuple[float, list[dict[str, float]], float, float]:
    lines = build_lines(phi_deg)
    if not lines:
        return float("inf"), [], float("nan"), float("nan")

    overlap_rates = [
        (b["right"] - a["left"]) / min(a["width"], b["width"])
        for a, b in zip(lines, lines[1:])
    ]
    if not overlap_rates:
        return float("inf"), [], float("nan"), float("nan")

    total_length_nm = sum(line["length"] for line in lines) / NM
    min_overlap = min(overlap_rates) * 100
    max_overlap = max(overlap_rates) * 100
    feasible = min_overlap >= 10.0 - 1e-7 and max_overlap <= 20.0 + 1e-7
    if not feasible:
        return float("inf"), lines, min_overlap, max_overlap
    return total_length_nm, lines, min_overlap, max_overlap


def coarse_search() -> list[dict[str, float]]:
    results = []
    steps = round((PHI_MAX_DEG - PHI_MIN_DEG) / COARSE_STEP_DEG)
    print(
        f"[1/4] 开始粗搜索: {PHI_MIN_DEG:.1f}° 到 {PHI_MAX_DEG:.1f}°, "
        f"步长 {COARSE_STEP_DEG:.3f}°, 共 {steps + 1} 个角度",
        flush=True,
    )
    for index in range(steps + 1):
        phi = PHI_MIN_DEG + index * COARSE_STEP_DEG
        total, lines, min_overlap, max_overlap = evaluate_phi(phi)
        if index == 0 or index == steps or index % 20 == 0:
            total_text = "不可行" if not math.isfinite(total) else f"{total:.6f} 海里"
            print(
                f"  粗搜索进度 {index + 1:>3}/{steps + 1}: "
                f"phi={phi:.3f}°, total={total_text}, lines={len(lines)}",
                flush=True,
            )
        results.append(
            {
                "stage": "coarse",
                "phi_deg": phi,
                "total_length_nm": total,
                "line_count": len(lines),
                "overlap_min_percent": min_overlap,
                "overlap_max_percent": max_overlap,
            }
        )
    return results


def golden_section_search(lo: float, hi: float) -> list[dict[str, float]]:
    print(
        f"[2/4] 开始黄金分割精修: 区间 [{lo:.6f}°, {hi:.6f}°], "
        f"迭代 {GOLDEN_ITERATIONS} 次",
        flush=True,
    )
    inv_phi = (math.sqrt(5) - 1) / 2
    c = hi - inv_phi * (hi - lo)
    d = lo + inv_phi * (hi - lo)
    fc, lines_c, min_c, max_c = evaluate_phi(c)
    fd, lines_d, min_d, max_d = evaluate_phi(d)
    history = []

    for iteration in range(1, GOLDEN_ITERATIONS + 1):
        if fc <= fd:
            hi = d
            d, fd, lines_d, min_d, max_d = c, fc, lines_c, min_c, max_c
            c = hi - inv_phi * (hi - lo)
            fc, lines_c, min_c, max_c = evaluate_phi(c)
            best_phi, best_total, best_lines, best_min, best_max = d, fd, lines_d, min_d, max_d
        else:
            lo = c
            c, fc, lines_c, min_c, max_c = d, fd, lines_d, min_d, max_d
            d = lo + inv_phi * (hi - lo)
            fd, lines_d, min_d, max_d = evaluate_phi(d)
            best_phi, best_total, best_lines, best_min, best_max = c, fc, lines_c, min_c, max_c

        history.append(
            {
                "stage": "golden",
                "iteration": iteration,
                "phi_deg": best_phi,
                "total_length_nm": best_total,
                "line_count": len(best_lines),
                "overlap_min_percent": best_min,
                "overlap_max_percent": best_max,
                "lo_deg": lo,
                "hi_deg": hi,
            }
        )
        if iteration == 1 or iteration == GOLDEN_ITERATIONS or iteration % 10 == 0:
            total_text = "不可行" if not math.isfinite(best_total) else f"{best_total:.9f} 海里"
            print(
                f"  黄金分割进度 {iteration:>3}/{GOLDEN_ITERATIONS}: "
                f"best_phi={best_phi:.9f}°, total={total_text}, "
                f"区间宽度={hi - lo:.9f}°",
                flush=True,
            )
    return history


def optimize_phi() -> tuple[float, float, list[dict[str, float]], list[dict[str, float]]]:
    history = coarse_search()
    feasible = [row for row in history if math.isfinite(row["total_length_nm"])]
    if not feasible:
        raise RuntimeError("no feasible angle found")

    best_coarse = min(feasible, key=lambda row: row["total_length_nm"])
    print(
        f"  粗搜索最优: phi={best_coarse['phi_deg']:.6f}°, "
        f"total={best_coarse['total_length_nm']:.6f} 海里, "
        f"lines={best_coarse['line_count']}",
        flush=True,
    )
    lo = max(PHI_MIN_DEG, best_coarse["phi_deg"] - GOLDEN_HALF_WIDTH_DEG)
    hi = min(PHI_MAX_DEG, best_coarse["phi_deg"] + GOLDEN_HALF_WIDTH_DEG)

    golden_history = golden_section_search(lo, hi)
    history.extend(golden_history)

    best_row = min(
        [row for row in history if math.isfinite(row["total_length_nm"])],
        key=lambda row: row["total_length_nm"],
    )
    best_phi = best_row["phi_deg"]
    best_total, best_lines, _, _ = evaluate_phi(best_phi)
    return best_phi, best_total, best_lines, history


def save_lines(phi_deg: float, lines: list[dict[str, float]]) -> None:
    fieldnames = [
        "index",
        "method",
        "phi_deg",
        "eta_target",
        "x1_m",
        "y1_m",
        "x2_m",
        "y2_m",
        "length_nm",
        "r_m",
        "depth_m",
        "width_m",
        "cover_left_m",
        "cover_right_m",
    ]
    with LINES_CSV_PATH.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for index, line in enumerate(lines, start=1):
            writer.writerow(
                {
                    "index": index,
                    "method": "angle_search_golden_recursive",
                    "phi_deg": round(phi_deg, 9),
                    "eta_target": ETA_TARGET,
                    "x1_m": round(line["x1"], 2),
                    "y1_m": round(line["y1"], 2),
                    "x2_m": round(line["x2"], 2),
                    "y2_m": round(line["y2"], 2),
                    "length_nm": round(line["length"] / NM, 6),
                    "r_m": round(line["r"], 2),
                    "depth_m": round(line["depth"], 2),
                    "width_m": round(line["width"], 2),
                    "cover_left_m": round(line["left"], 2),
                    "cover_right_m": round(line["right"], 2),
                }
            )


def save_history(history: list[dict[str, float]]) -> None:
    fieldnames = [
        "stage",
        "iteration",
        "phi_deg",
        "total_length_nm",
        "line_count",
        "overlap_min_percent",
        "overlap_max_percent",
        "lo_deg",
        "hi_deg",
    ]
    with HISTORY_CSV_PATH.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def clip_half_plane(
    polygon: list[tuple[float, float]],
    normal: tuple[float, float],
    limit: float,
    keep_less: bool,
) -> list[tuple[float, float]]:
    def signed(point: tuple[float, float]) -> float:
        value = point[0] * normal[0] + point[1] * normal[1] - limit
        return value if keep_less else -value

    output = []
    for start, end in zip(polygon, polygon[1:] + polygon[:1]):
        ds, de = signed(start), signed(end)
        start_inside, end_inside = ds <= 1e-10, de <= 1e-10
        if start_inside:
            output.append(start)
        if start_inside != end_inside:
            ratio = ds / (ds - de)
            output.append(
                (
                    start[0] + ratio * (end[0] - start[0]),
                    start[1] + ratio * (end[1] - start[1]),
                )
            )
    return output


def stripe_polygon(normal: tuple[float, float], left_m: float, right_m: float) -> list[tuple[float, float]]:
    polygon = [
        (X_MIN / NM, Y_MIN / NM),
        (X_MAX / NM, Y_MIN / NM),
        (X_MAX / NM, Y_MAX / NM),
        (X_MIN / NM, Y_MAX / NM),
    ]
    left_nm, right_nm = left_m / NM, right_m / NM
    polygon = clip_half_plane(polygon, normal, right_nm, keep_less=True)
    return clip_half_plane(polygon, normal, left_nm, keep_less=False)


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    ]
    for font_path in candidates:
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def blend(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))


def depth_color(depth: float, min_depth: float, max_depth: float) -> tuple[int, int, int]:
    t = (depth - min_depth) / (max_depth - min_depth)
    if t < 0.5:
        return blend((217, 240, 240), (78, 165, 181), t * 2)
    return blend((78, 165, 181), (23, 74, 104), (t - 0.5) * 2)


def rgb(color: tuple[int, int, int]) -> str:
    return f"rgb({color[0]},{color[1]},{color[2]})"


def draw_info_panel(
    draw: ImageDraw.ImageDraw,
    panel_x: int,
    panel_y: int,
    panel_w: int,
    title_font: ImageFont.ImageFont,
    text_font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
    summary: str,
    min_depth: float,
    max_depth: float,
) -> None:
    panel_h = 470
    draw.rounded_rectangle(
        (panel_x, panel_y, panel_x + panel_w, panel_y + panel_h),
        radius=10,
        fill=(255, 255, 255, 235),
        outline=(185, 199, 206, 255),
    )
    draw.text((panel_x + 22, panel_y + 22), "结果摘要", font=title_font, fill=(15, 23, 42))
    draw.multiline_text((panel_x + 22, panel_y + 66), summary, font=text_font, fill=(16, 44, 61), spacing=8)

    legend_y = panel_y + 280
    draw.text((panel_x + 22, legend_y), "海水深度 / m", font=small_font, fill=(71, 85, 105))
    for i in range(220):
        d = min_depth + (max_depth - min_depth) * i / 219
        draw.line(
            (panel_x + 22 + i, legend_y + 30, panel_x + 22 + i, legend_y + 50),
            fill=depth_color(d, min_depth, max_depth) + (255,),
        )
    draw.text((panel_x + 22, legend_y + 58), f"{min_depth:.1f}", font=small_font, fill=(71, 85, 105))
    draw.text((panel_x + 198, legend_y + 58), f"{max_depth:.1f}", font=small_font, fill=(71, 85, 105))
    for item_index, (color, label) in enumerate([((189, 230, 232), "浅青色：覆盖区域"), ((231, 94, 54), "红橙色：重叠覆盖区域"), ((247, 243, 232), "白色：实际测线")]):
        y0 = legend_y + 92 + item_index * 28
        draw.rounded_rectangle((panel_x + 22, y0, panel_x + 50, y0 + 16), radius=3, fill=color + (255,), outline=(100, 116, 139, 180))
        draw.text((panel_x + 62, y0 - 2), label, font=small_font, fill=(30, 41, 59))


def world_to_pixel(x_nm: float, y_nm: float) -> tuple[float, float]:
    plot_left, plot_top = 84, 56
    plot_w, plot_h = 1080, 540
    px = plot_left + (x_nm - (-2.08)) / 4.16 * plot_w
    py = plot_top + (1.08 - y_nm) / 2.16 * plot_h
    return px, py


def svg_points(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def draw_svg(phi_deg: float, total_length_nm: float, lines: list[dict[str, float]]) -> None:
    print("[绘图] 生成角度搜索 SVG 矢量图...", flush=True)
    phi = math.radians(phi_deg)
    normal = (-math.sin(phi), math.cos(phi))
    plot_left, plot_top = 84, 56
    plot_w, plot_h = 1080, 540
    min_depth = depth_at_x(X_MAX)
    max_depth = depth_at_x(X_MIN)
    overlap_rates = [
        (b["right"] - a["left"]) / min(a["width"], b["width"]) * 100
        for a, b in zip(lines, lines[1:])
    ]
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1240" height="680" viewBox="0 0 1240 680">',
        "<style><![CDATA[text{font-family:'Microsoft YaHei','SimHei',Arial,sans-serif}.thin{vector-effect:non-scaling-stroke}.fixed-label{font-size:14px}.axis{font-size:17px;font-weight:700}.note{font-size:15px;font-weight:700}]]></style>",
        '<rect width="1240" height="680" fill="#ffffff"/>',
    ]
    for i in range(120):
        x_nm = -2.08 + i / 119 * 4.16
        x = plot_left + i / 120 * plot_w
        parts.append(f'<rect x="{x:.2f}" y="{plot_top}" width="{plot_w / 120 + 0.8:.2f}" height="{plot_h}" fill="{rgb(depth_color(depth_at_x(x_nm * NM), min_depth, max_depth))}" opacity="0.88"/>')
    for tick in np.linspace(-2, 2, 9):
        px, _ = world_to_pixel(float(tick), 0)
        parts.append(f'<line class="thin" x1="{px:.2f}" y1="{plot_top}" x2="{px:.2f}" y2="{plot_top + plot_h}" stroke="#ffffff" stroke-width="1" opacity="0.55"/>')
        parts.append(f'<text x="{px:.2f}" y="{plot_top + plot_h + 24}" class="fixed-label" fill="#475569" text-anchor="middle">{tick:g}</text>')
    for tick in np.linspace(-1, 1, 5):
        _, py = world_to_pixel(0, float(tick))
        parts.append(f'<line class="thin" x1="{plot_left}" y1="{py:.2f}" x2="{plot_left + plot_w}" y2="{py:.2f}" stroke="#ffffff" stroke-width="1" opacity="0.55"/>')
        parts.append(f'<text x="{plot_left - 22}" y="{py + 5:.2f}" class="fixed-label" fill="#475569" text-anchor="middle">{tick:g}</text>')

    for line in lines:
        polygon = stripe_polygon(normal, line["left"], line["right"])
        if polygon:
            parts.append(f'<polygon points="{svg_points([world_to_pixel(x, y) for x, y in polygon])}" fill="#bde6e8" opacity="0.38"/>')

    for previous, current in zip(lines, lines[1:]):
        polygon = stripe_polygon(normal, previous["left"], current["right"])
        if polygon:
            parts.append(f'<polygon points="{svg_points([world_to_pixel(x, y) for x, y in polygon])}" fill="#e75e36" opacity="0.62"/>')

    for index, line in enumerate(lines, start=1):
        p1 = world_to_pixel(line["x1"] / NM, line["y1"] / NM)
        p2 = world_to_pixel(line["x2"] / NM, line["y2"] / NM)
        parts.append(f'<line class="thin" x1="{p1[0]:.2f}" y1="{p1[1]:.2f}" x2="{p2[0]:.2f}" y2="{p2[1]:.2f}" stroke="#fff7ed" stroke-width="1.6" stroke-linecap="round" opacity="0.96"/>')
        if index == 1 or index == len(lines) or index % 5 == 0:
            mx, my = world_to_pixel((line["x1"] + line["x2"]) / (2 * NM), (line["y1"] + line["y2"]) / (2 * NM))
            parts.append(f'<rect class="thin" x="{mx - 13:.2f}" y="{my - 11:.2f}" width="26" height="22" rx="5" fill="#ffffff" opacity="0.86" stroke="#94a3b8" stroke-width="1"/>')
            parts.append(f'<text x="{mx:.2f}" y="{my + 5:.2f}" class="fixed-label" fill="#122a3a" text-anchor="middle">{index}</text>')

    border = [world_to_pixel(-2, -1), world_to_pixel(2, -1), world_to_pixel(2, 1), world_to_pixel(-2, 1)]
    parts.append(f'<polygon class="thin" points="{svg_points(border)}" fill="none" stroke="#102c3d" stroke-width="3"/>')
    parts.append(f'<text x="{plot_left + plot_w / 2}" y="{plot_top + plot_h + 55}" class="axis" fill="#1e293b" text-anchor="middle">东西方向 / n mile（西深，东浅）</text>')
    parts.append(f'<text x="{plot_left}" y="{plot_top - 12}" class="axis" fill="#1e293b">南北方向 / n mile</text>')

    parts.append(f'<text x="{plot_left + 18}" y="{plot_top + 30}" class="note" fill="#ffffff">深水侧（西）</text>')
    parts.append(f'<text x="{plot_left + plot_w - 18}" y="{plot_top + 30}" class="note" fill="#25384a" text-anchor="end">浅水侧（东）</text>')
    lx, ly = plot_left + 390, plot_top + 16
    parts.append(f'<rect x="{lx}" y="{ly}" width="330" height="38" rx="5" fill="#ffffff" fill-opacity="0.86" stroke="#b9c7ce"/>')
    parts.append(f'<rect x="{lx + 16}" y="{ly + 11}" width="26" height="14" fill="#e75e36" fill-opacity="0.72"/>')
    parts.append(f'<text x="{lx + 49}" y="{ly + 24}" class="fixed-label" fill="#263442">重叠区域</text>')
    parts.append(f'<line x1="{lx + 142}" y1="{ly + 18}" x2="{lx + 174}" y2="{ly + 18}" stroke="#fff7ed" stroke-width="3"/>')
    parts.append(f'<text x="{lx + 184}" y="{ly + 24}" class="fixed-label" fill="#263442">实际测线（编号）</text>')
    parts.append("</svg>")
    SVG_PATH.write_text("\n".join(parts), encoding="utf-8")


def save_plot(phi_deg: float, total_length_nm: float, lines: list[dict[str, float]]) -> None:
    phi = math.radians(phi_deg)
    normal = (-math.sin(phi), math.cos(phi))

    image = Image.new("RGB", (1240, 680), (255, 255, 255))
    draw = ImageDraw.Draw(image, "RGBA")
    label_font = load_font(17, bold=True)
    small_font = load_font(14)

    plot_left, plot_top = 84, 56
    plot_w, plot_h = 1080, 540
    min_depth = depth_at_x(X_MAX)
    max_depth = depth_at_x(X_MIN)
    for i in range(plot_w):
        x_nm = -2.08 + i / plot_w * 4.16
        depth = 110.0 - x_nm * NM * math.tan(ALPHA)
        color = depth_color(depth, min_depth, max_depth)
        draw.line((plot_left + i, plot_top, plot_left + i, plot_top + plot_h), fill=color + (225,))

    for tick in np.linspace(-2, 2, 9):
        px, _ = world_to_pixel(float(tick), 0)
        draw.line((px, plot_top, px, plot_top + plot_h), fill=(255, 255, 255, 75), width=1)
        draw.text((px - 12, plot_top + plot_h + 8), f"{tick:g}", font=small_font, fill=(71, 85, 105))
    for tick in np.linspace(-1, 1, 5):
        _, py = world_to_pixel(0, float(tick))
        draw.line((plot_left, py, plot_left + plot_w, py), fill=(255, 255, 255, 75), width=1)
        draw.text((plot_left - 40, py - 8), f"{tick:g}", font=small_font, fill=(71, 85, 105))

    for line in lines:
        polygon = stripe_polygon(normal, line["left"], line["right"])
        if polygon:
            pixels = [world_to_pixel(x, y) for x, y in polygon]
            draw.polygon(pixels, fill=(189, 230, 232, 78))

    for previous, current in zip(lines, lines[1:]):
        polygon = stripe_polygon(normal, previous["left"], current["right"])
        if polygon:
            pixels = [world_to_pixel(x, y) for x, y in polygon]
            draw.polygon(pixels, fill=(231, 94, 54, 145))

    for index, line in enumerate(lines, start=1):
        x1, y1 = line["x1"] / NM, line["y1"] / NM
        x2, y2 = line["x2"] / NM, line["y2"] / NM
        p1 = world_to_pixel(x1, y1)
        p2 = world_to_pixel(x2, y2)
        draw.line((p1, p2), fill=(247, 243, 232, 245), width=2)
        if index == 1 or index == len(lines) or index % 5 == 0:
            pm = world_to_pixel((x1 + x2) / 2, (y1 + y2) / 2)
            label = str(index)
            bbox = draw.textbbox((0, 0), label, font=small_font)
            box = (pm[0] - 13, pm[1] - 11, pm[0] + 13, pm[1] + 11)
            draw.rounded_rectangle(box, radius=5, fill=(255, 255, 255, 215))
            draw.text(
                (pm[0] - (bbox[2] - bbox[0]) / 2, pm[1] - (bbox[3] - bbox[1]) / 2 - 1),
                label,
                font=small_font,
                fill=(18, 42, 58),
            )

    border_points = [world_to_pixel(-2, -1), world_to_pixel(2, -1), world_to_pixel(2, 1), world_to_pixel(-2, 1)]
    draw.line(border_points + [border_points[0]], fill=(16, 44, 61, 255), width=3)

    draw.text((plot_left + plot_w / 2 - 120, plot_top + plot_h + 42), "东西方向 / n mile（西深，东浅）", font=label_font, fill=(30, 41, 59))
    draw.text((plot_left, plot_top - 30), "南北方向 / n mile", font=label_font, fill=(30, 41, 59))

    draw.text((plot_left + 18, plot_top + 13), "深水侧（西）", font=small_font, fill=(255, 255, 255))
    deep_label = "浅水侧（东）"
    deep_box = draw.textbbox((0, 0), deep_label, font=small_font)
    draw.text((plot_left + plot_w - 18 - (deep_box[2] - deep_box[0]), plot_top + 13), deep_label, font=small_font, fill=(37, 56, 74))
    legend_x, legend_y = plot_left + 390, plot_top + 16
    draw.rounded_rectangle((legend_x, legend_y, legend_x + 330, legend_y + 38), radius=5, fill=(255, 255, 255, 220), outline=(185, 199, 206, 255))
    draw.rectangle((legend_x + 16, legend_y + 11, legend_x + 42, legend_y + 25), fill=(231, 94, 54, 180))
    draw.text((legend_x + 49, legend_y + 9), "重叠区域", font=small_font, fill=(38, 52, 66))
    draw.line((legend_x + 142, legend_y + 18, legend_x + 174, legend_y + 18), fill=(247, 243, 232, 255), width=3)
    draw.text((legend_x + 184, legend_y + 9), "实际测线（编号）", font=small_font, fill=(38, 52, 66))

    image.save(PNG_PATH, dpi=(300, 300))
    draw_svg(phi_deg, total_length_nm, lines)


def main() -> None:
    print("=== 2023B 问题三：方向角搜索程序启动 ===", flush=True)
    best_phi, best_total, best_lines, history = optimize_phi()
    print("[3/4] 搜索完成，开始保存 CSV 和绘图", flush=True)
    save_lines(best_phi, best_lines)
    save_history(history)
    save_plot(best_phi, best_total, best_lines)
    print("[4/4] 文件保存完成，输出最终结果", flush=True)

    overlap_rates = [
        (b["right"] - a["left"]) / min(a["width"], b["width"]) * 100
        for a, b in zip(best_lines, best_lines[1:])
    ]
    total_at_90, lines_at_90, min_90, max_90 = evaluate_phi(90.0)

    print("method: angle_search_golden_recursive")
    print("best_phi_deg:", round(best_phi, 9))
    print("best_total_length_nm:", round(best_total, 9))
    print("line_count:", len(best_lines))
    print("overlap_min_percent:", round(min(overlap_rates), 9))
    print("overlap_max_percent:", round(max(overlap_rates), 9))
    print("phi_90_total_length_nm:", round(total_at_90, 9))
    print("phi_90_line_count:", len(lines_at_90))
    print("phi_90_overlap_min_percent:", round(min_90, 9))
    print("phi_90_overlap_max_percent:", round(max_90, 9))
    print("saved_lines:", LINES_CSV_PATH)
    print("saved_history:", HISTORY_CSV_PATH)
    print("saved_png:", PNG_PATH)
    print("saved_svg:", SVG_PATH)


if __name__ == "__main__":
    main()
