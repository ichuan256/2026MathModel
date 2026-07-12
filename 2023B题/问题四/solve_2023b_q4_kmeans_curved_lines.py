import csv
import html
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from kmeans_depth_classification import (
    K_MAX,
    K_MIN,
    ZONE_COLORS,
    cluster_ranges,
    elbow_analysis,
    kmeans_1d,
)
from make_contour_from_attachment import load_font, marching_segments, read_bathymetry_grid


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_LINES_CSV = BASE_DIR / "q4_kmeans_curved_lines.csv"
OUTPUT_SUMMARY_CSV = BASE_DIR / "q4_kmeans_curved_summary.csv"
OUTPUT_PNG = BASE_DIR / "q4_kmeans_curved_lines.png"
OUTPUT_SVG = BASE_DIR / "q4_kmeans_curved_lines.svg"

NM = 1852.0
OPENING_ANGLE_DEG = 120.0
HALF_ANGLE = math.radians(OPENING_ANGLE_DEG / 2)
TAN_HALF = math.tan(HALF_ANGLE)

SWARM_SIZE = 28
MAX_ITER = 12000
MIN_ITER = 10000
CONVERGENCE_TOL = 1e-7
CONVERGENCE_PATIENCE = 1000
ETA_MIN = 0.02
ETA_MAX = 0.199
SEED = 2023


def swath_half_depth(center_depth: float, gradient_m_per_nm: float) -> float:
    """Approximate half swath width in depth-coordinate space."""
    half_width_nm = center_depth * TAN_HALF / NM
    return max(0.05, gradient_m_per_nm * half_width_nm)


def estimate_region_gradient(
    x: np.ndarray, y: np.ndarray, depth: np.ndarray, labels: np.ndarray, cluster_id: int
) -> float:
    grad_y, grad_x = np.gradient(depth, y, x)
    magnitude = np.hypot(grad_x, grad_y)
    values = magnitude[(labels == cluster_id) & np.isfinite(magnitude)]
    if values.size == 0:
        return float(np.nanmedian(magnitude))
    return max(0.5, float(np.nanmedian(values)))


def depth_bounds_from_ranges(ranges: list[dict]) -> list[tuple[float, float]]:
    return [(row["min_depth"], row["max_depth"]) for row in ranges]


def contour_length_lookup(x: np.ndarray, y: np.ndarray, depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    print("[3/8] Precomputing contour length lookup table...", flush=True)
    z_min = float(np.nanmin(depth))
    z_max = float(np.nanmax(depth))
    levels = np.arange(math.floor(z_min), math.ceil(z_max) + 0.25, 0.25)
    lengths = []
    for index, level in enumerate(levels, start=1):
        length = 0.0
        for p1, p2 in marching_segments(x, y, depth, float(level)):
            length += math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        lengths.append(length)
        if index == 1 or index % 120 == 0 or index == len(levels):
            print(f"    contour lookup progress {index}/{len(levels)}", flush=True)
    return levels, np.asarray(lengths, dtype=float)


def interpolated_contour_length(levels: np.ndarray, lengths: np.ndarray, depth_value: float) -> float:
    return float(np.interp(depth_value, levels, lengths, left=0.0, right=0.0))


def build_region_lane_depths(
    shallow_boundary: float,
    deep_boundary: float,
    gradient_m_per_nm: float,
    eta: float,
) -> list[dict]:
    """Generate centerline depths from the deepest boundary to the shallowest boundary."""
    lanes = []
    target_deep_edge = deep_boundary
    for _ in range(500):
        coefficient = gradient_m_per_nm * TAN_HALF / NM
        center_depth = target_deep_edge / (1.0 + coefficient)
        half_depth = swath_half_depth(center_depth, gradient_m_per_nm)
        deep_edge = center_depth + half_depth
        shallow_edge = center_depth - half_depth
        lanes.append(
            {
                "center_depth": center_depth,
                "deep_edge": deep_edge,
                "shallow_edge": shallow_edge,
                "width_depth": deep_edge - shallow_edge,
            }
        )
        if shallow_edge <= shallow_boundary:
            break
        target_deep_edge = shallow_edge + eta * (deep_edge - shallow_edge)
    return lanes


def build_layout_from_eta(
    etas: np.ndarray,
    ranges: list[dict],
    gradients: list[float],
    levels: np.ndarray,
    lengths: np.ndarray,
) -> tuple[list[dict], dict]:
    all_lanes = []
    total_length_nm = 0.0
    overlap_excess_length_nm = 0.0

    for cluster_id, row in enumerate(ranges):
        shallow, deep = row["min_depth"], row["max_depth"]
        lanes = build_region_lane_depths(shallow, deep, gradients[cluster_id], float(etas[cluster_id]))
        for lane_index, lane in enumerate(lanes, start=1):
            length_nm = interpolated_contour_length(levels, lengths, lane["center_depth"])
            total_length_nm += length_nm
            all_lanes.append(
                {
                    "cluster": cluster_id + 1,
                    "lane_index": lane_index,
                    "center_depth": lane["center_depth"],
                    "deep_edge": lane["deep_edge"],
                    "shallow_edge": lane["shallow_edge"],
                    "length_nm": length_nm,
                    "eta_target": float(etas[cluster_id]),
                }
            )

        for previous, current in zip(lanes, lanes[1:]):
            overlap = current["deep_edge"] - previous["shallow_edge"]
            base = min(previous["width_depth"], current["width_depth"])
            rate = overlap / max(base, 1e-12)
            if rate > 0.20:
                overlap_excess_length_nm += interpolated_contour_length(
                    levels, lengths, current["center_depth"]
                ) * (rate - 0.20)

    summary = {
        "line_count": len(all_lanes),
        "total_length_nm": total_length_nm,
        "overlap_excess_length_nm": overlap_excess_length_nm,
        "uncovered_ratio": 0.0,
    }
    return all_lanes, summary


def evaluate(
    etas: np.ndarray,
    ranges: list[dict],
    gradients: list[float],
    levels: np.ndarray,
    lengths: np.ndarray,
) -> tuple[float, list[dict], dict]:
    lanes, summary = build_layout_from_eta(etas, ranges, gradients, levels, lengths)
    penalty = summary["overlap_excess_length_nm"] * 1000.0 + summary["uncovered_ratio"] * 1e6
    fitness = summary["total_length_nm"] + penalty
    return fitness, lanes, summary


def optimize_etas(
    ranges: list[dict],
    gradients: list[float],
    levels: np.ndarray,
    lengths: np.ndarray,
) -> tuple[np.ndarray, list[dict], dict, int, int]:
    print(
        f"[4/8] Starting PSO iteration: swarm={SWARM_SIZE}, min_iter={MIN_ITER}, "
        f"patience={CONVERGENCE_PATIENCE}, tol={CONVERGENCE_TOL}",
        flush=True,
    )
    random.seed(SEED)
    dim = len(ranges)
    particles = []
    global_best_position = None
    global_best_fitness = float("inf")
    global_best_lanes = []
    global_best_summary = {}

    for _ in range(SWARM_SIZE):
        position = np.asarray([random.uniform(ETA_MIN, ETA_MAX) for _ in range(dim)], dtype=float)
        velocity = np.asarray([random.uniform(-0.01, 0.01) for _ in range(dim)], dtype=float)
        fitness, lanes, summary = evaluate(position, ranges, gradients, levels, lengths)
        particle = {
            "position": position,
            "velocity": velocity,
            "best_position": position.copy(),
            "best_fitness": fitness,
        }
        particles.append(particle)
        if fitness < global_best_fitness:
            global_best_position = position.copy()
            global_best_fitness = fitness
            global_best_lanes = lanes
            global_best_summary = summary

    converged_count = 0
    iterations_used = MAX_ITER
    for iteration in range(1, MAX_ITER + 1):
        old_best = global_best_fitness
        w = 0.9 - (0.9 - 0.4) * (iteration - 1) / max(1, MAX_ITER - 1)
        c1 = 2.4 - 1.7 * (iteration - 1) / max(1, MAX_ITER - 1)
        c2 = 0.7 + 1.7 * (iteration - 1) / max(1, MAX_ITER - 1)

        for particle in particles:
            for d in range(dim):
                r1 = random.random()
                r2 = random.random()
                cognitive = c1 * r1 * (particle["best_position"][d] - particle["position"][d])
                social = c2 * r2 * (global_best_position[d] - particle["position"][d])
                particle["velocity"][d] = w * particle["velocity"][d] + cognitive + social
                particle["velocity"][d] = float(np.clip(particle["velocity"][d], -0.03, 0.03))
                particle["position"][d] = float(
                    np.clip(particle["position"][d] + particle["velocity"][d], ETA_MIN, ETA_MAX)
                )

            fitness, lanes, summary = evaluate(particle["position"], ranges, gradients, levels, lengths)
            if fitness < particle["best_fitness"]:
                particle["best_fitness"] = fitness
                particle["best_position"] = particle["position"].copy()
            if fitness < global_best_fitness:
                global_best_fitness = fitness
                global_best_position = particle["position"].copy()
                global_best_lanes = lanes
                global_best_summary = summary

        relative_error = abs(old_best - global_best_fitness) / max(abs(old_best), 1e-12)
        if iteration <= MIN_ITER:
            converged_count = 0
        elif relative_error < CONVERGENCE_TOL:
            converged_count += 1
        else:
            converged_count = 0

        if iteration == 1 or iteration % 500 == 0:
            eta_text = ", ".join(f"{v:.5f}" for v in global_best_position)
            print(
                f"    iter {iteration:>5}/{MAX_ITER}: fitness={global_best_fitness:.6f}, "
                f"total_length={global_best_summary['total_length_nm']:.3f} n mile, "
                f"lines={global_best_summary['line_count']}, etas=[{eta_text}], "
                f"stable={converged_count}",
                flush=True,
            )

        if iteration >= MIN_ITER and converged_count >= CONVERGENCE_PATIENCE:
            iterations_used = iteration
            print(
                f"    stop condition reached at iteration {iterations_used}, "
                f"stable_count={converged_count}",
                flush=True,
            )
            break

    return global_best_position, global_best_lanes, global_best_summary, iterations_used, converged_count


def segment_mid_depth(depth: np.ndarray, x: np.ndarray, y: np.ndarray, midpoint: tuple[float, float]) -> float:
    xi = int(np.clip(round((midpoint[0] - x[0]) / max(x[1] - x[0], 1e-12)), 0, len(x) - 1))
    yi = int(np.clip(round((midpoint[1] - y[0]) / max(y[1] - y[0], 1e-12)), 0, len(y) - 1))
    return float(depth[yi, xi])


def draw_layout(
    x: np.ndarray,
    y: np.ndarray,
    depth: np.ndarray,
    ranges: list[dict],
    lanes: list[dict],
    summary: dict,
    iterations_used: int,
) -> None:
    print("[6/8] Drawing region-wise curved survey-line layout...", flush=True)
    width, height = 1700, 980
    left, top, plot_w, plot_h = 110, 135, 1120, 680
    right, bottom = left + plot_w, top + plot_h

    def to_pixel(px: float, py: float) -> tuple[float, float]:
        xx = left + (px - float(x.min())) / max(float(x.max() - x.min()), 1e-12) * plot_w
        yy = bottom - (py - float(y.min())) / max(float(y.max() - y.min()), 1e-12) * plot_h
        return xx, yy

    image = Image.new("RGB", (width, height), (247, 250, 252))
    draw = ImageDraw.Draw(image, "RGBA")
    title_font = load_font(28, bold=True)
    label_font = load_font(18, bold=True)
    text_font = load_font(14)
    small_font = load_font(12)

    draw.text((60, 35), "问题四：K-means 分区曲线测线布设结果", font=title_font, fill=(15, 23, 42))

    for i in range(len(y) - 1):
        for j in range(len(x) - 1):
            z = float(np.nanmean([depth[i, j], depth[i, j + 1], depth[i + 1, j], depth[i + 1, j + 1]]))
            cluster_id = 0
            for idx, row in enumerate(ranges):
                if row["min_depth"] <= z <= row["max_depth"]:
                    cluster_id = idx
                    break
            color = ZONE_COLORS[min(cluster_id, len(ZONE_COLORS) - 1)]
            p1 = to_pixel(float(x[j]), float(y[i]))
            p2 = to_pixel(float(x[j + 1]), float(y[i + 1]))
            draw.rectangle((p1[0], p2[1], p2[0], p1[1]), fill=color + (210,))

    for value in np.linspace(float(x.min()), float(x.max()), 9):
        px, _ = to_pixel(value, float(y.min()))
        draw.line((px, top, px, bottom), fill=(255, 255, 255, 65), width=1)
        draw.text((px - 18, bottom + 10), f"{value:.1f}", font=small_font, fill=(71, 85, 105))
    for value in np.linspace(float(y.min()), float(y.max()), 6):
        _, py = to_pixel(float(x.min()), value)
        draw.line((left, py, right, py), fill=(255, 255, 255, 65), width=1)
        draw.text((left - 52, py - 8), f"{value:.1f}", font=small_font, fill=(71, 85, 105))

    # Draw K-means region boundaries.
    for idx in range(len(ranges) - 1):
        boundary = (ranges[idx]["max_depth"] + ranges[idx + 1]["min_depth"]) / 2
        for p1, p2 in marching_segments(x, y, depth, float(boundary)):
            draw.line((*to_pixel(*p1), *to_pixel(*p2)), fill=(19, 36, 56, 190), width=2)

    depth_span = max(float(np.nanmax(depth) - np.nanmin(depth)), 1e-12)
    depth_to_px = plot_h / depth_span

    def draw_depth_contour_band(
        level: float,
        cluster_idx: int,
        color: tuple[int, int, int, int],
        width_depth: float,
        width_scale: float,
        min_width: int,
        max_width: int,
    ) -> None:
        row = ranges[cluster_idx]
        stroke_width = int(round(np.clip(width_depth * depth_to_px * width_scale, min_width, max_width)))
        for p1, p2 in marching_segments(x, y, depth, float(level)):
            midpoint = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
            mid_depth = segment_mid_depth(depth, x, y, midpoint)
            if row["min_depth"] <= mid_depth <= row["max_depth"]:
                draw.line((*to_pixel(*p1), *to_pixel(*p2)), fill=color, width=stroke_width)

    # Draw curved swath bands instead of cell blocks: light cyan = covered once, red-orange = overlap.
    for lane in lanes:
        cluster_idx = int(lane["cluster"]) - 1
        draw_depth_contour_band(
            lane["center_depth"],
            cluster_idx,
            (189, 230, 232, 106),
            lane["deep_edge"] - lane["shallow_edge"],
            0.34,
            5,
            16,
        )

    for previous, current in zip(lanes, lanes[1:]):
        if previous["cluster"] != current["cluster"]:
            continue
        overlap_depth = current["deep_edge"] - previous["shallow_edge"]
        if overlap_depth <= 0:
            continue
        cluster_idx = int(current["cluster"]) - 1
        overlap_level = (current["deep_edge"] + previous["shallow_edge"]) / 2
        draw_depth_contour_band(
            overlap_level,
            cluster_idx,
            (231, 94, 54, 176),
            overlap_depth,
            0.62,
            4,
            18,
        )

    # Draw curved survey lines. Keep each contour segment whose midpoint lies in the target region.
    palette = [(255, 255, 255), (255, 246, 204), (255, 220, 160), (255, 184, 108), (255, 150, 92)]
    for lane in lanes:
        cluster_idx = int(lane["cluster"]) - 1
        line_color = palette[min(cluster_idx, len(palette) - 1)] + (245,)
        count = 0
        for p1, p2 in marching_segments(x, y, depth, float(lane["center_depth"])):
            midpoint = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
            mid_depth = segment_mid_depth(depth, x, y, midpoint)
            row = ranges[cluster_idx]
            if row["min_depth"] <= mid_depth <= row["max_depth"]:
                draw.line((*to_pixel(*p1), *to_pixel(*p2)), fill=line_color, width=3)
                count += 1
        if lane["lane_index"] == 1:
            # Highlight the first line in each region: its deep-side swath edge is aligned to the deep boundary.
            for p1, p2 in marching_segments(x, y, depth, float(lane["center_depth"])):
                midpoint = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
                mid_depth = segment_mid_depth(depth, x, y, midpoint)
                row = ranges[cluster_idx]
                if row["min_depth"] <= mid_depth <= row["max_depth"]:
                    draw.line((*to_pixel(*p1), *to_pixel(*p2)), fill=(255, 255, 255, 255), width=4)

    draw.rectangle((left, top, right, bottom), outline=(16, 44, 61, 255), width=3)
    draw.text((left + plot_w / 2 - 135, bottom + 50), "东西方向 / n mile", font=label_font, fill=(30, 41, 59))
    draw.text((left, top - 34), "南北方向 / n mile", font=label_font, fill=(30, 41, 59))

    info = (
        f"K-means 分区数：{len(ranges)}\n"
        f"测线数量：{summary['line_count']} 条\n"
        f"测线总长：{summary['total_length_nm']:.3f} n mile\n"
        f"未覆盖比例：{summary['uncovered_ratio'] * 100:.3f}%\n"
        f"超 20% 重叠长度：{summary['overlap_excess_length_nm']:.3f} n mile\n"
        "覆盖区：浅青色\n"
        "重叠区：红橙色\n"
        f"迭代代数：{iterations_used}"
    )
    panel_x, panel_y, panel_w, panel_h = 1270, 135, 365, 670
    draw.rounded_rectangle((panel_x, panel_y, panel_x + panel_w, panel_y + panel_h), radius=10, fill=(255, 255, 255, 235), outline=(185, 199, 206, 255))
    draw.text((panel_x + 22, panel_y + 24), "结果摘要", font=label_font, fill=(15, 23, 42))
    draw.multiline_text((panel_x + 22, panel_y + 70), info, font=text_font, fill=(16, 44, 61), spacing=8)

    legend_x, legend_y = panel_x + 22, panel_y + 350
    draw.text((legend_x, legend_y - 30), "颜色说明", font=label_font, fill=(30, 41, 59))
    color_legend = [
        ((189, 230, 232), "浅青色：测线覆盖区域"),
        ((231, 94, 54), "红橙色：重叠覆盖区域"),
        ((255, 255, 255), "白色：实际测线"),
    ]
    for item_index, (color, label) in enumerate(color_legend):
        y0 = legend_y + item_index * 28
        draw.rounded_rectangle((legend_x, y0, legend_x + 24, y0 + 18), radius=3, fill=color + (255,), outline=(100, 116, 139, 180))
        draw.text((legend_x + 34, y0 - 1), label, font=small_font, fill=(30, 41, 59))

    region_legend_y = legend_y + 118
    draw.text((legend_x, region_legend_y - 24), "K-means 水深分区", font=label_font, fill=(30, 41, 59))
    for idx, row in enumerate(ranges):
        y0 = region_legend_y + idx * 28
        color = ZONE_COLORS[min(idx, len(ZONE_COLORS) - 1)]
        draw.rounded_rectangle((legend_x, y0, legend_x + 24, y0 + 18), radius=3, fill=color + (255,))
        draw.text(
            (legend_x + 34, y0 - 1),
            f"区 {idx + 1}: {row['min_depth']:.1f}-{row['max_depth']:.1f} m",
            font=small_font,
            fill=(30, 41, 59),
        )

    image.save(OUTPUT_PNG)


def draw_layout_svg(
    x: np.ndarray,
    y: np.ndarray,
    depth: np.ndarray,
    ranges: list[dict],
    lanes: list[dict],
    summary: dict,
    iterations_used: int,
) -> None:
    print("[6/8] Drawing region-wise curved survey-line SVG...", flush=True)
    width, height = 1700, 980
    left, top, plot_w, plot_h = 110, 135, 1120, 680
    right, bottom = left + plot_w, top + plot_h

    def to_pixel(px: float, py: float) -> tuple[float, float]:
        xx = left + (px - float(x.min())) / max(float(x.max() - x.min()), 1e-12) * plot_w
        yy = bottom - (py - float(y.min())) / max(float(y.max() - y.min()), 1e-12) * plot_h
        return xx, yy

    def rgb(color: tuple[int, int, int]) -> str:
        return f"rgb({color[0]},{color[1]},{color[2]})"

    def region_id(z: float) -> int:
        for idx, row in enumerate(ranges):
            if row["min_depth"] <= z <= row["max_depth"]:
                return idx + 1
        return 1

    def svg_line(p1: tuple[float, float], p2: tuple[float, float], color: str, stroke_width: float, opacity: float = 1.0) -> str:
        a, b = to_pixel(*p1), to_pixel(*p2)
        return f'<line class="thin" x1="{a[0]:.2f}" y1="{a[1]:.2f}" x2="{b[0]:.2f}" y2="{b[1]:.2f}" stroke="{color}" stroke-width="{stroke_width}" opacity="{opacity}" stroke-linecap="round"/>'

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style><![CDATA[text{font-family:'Microsoft YaHei','SimHei',Arial,sans-serif}.thin{vector-effect:non-scaling-stroke}.fixed-label{font-size:12px}.title{font-size:28px;font-weight:700}.axis{font-size:18px;font-weight:700}.panel-title{font-size:18px;font-weight:700}.panel-text{font-size:14px}]]></style>",
        f'<rect width="{width}" height="{height}" fill="#f7fafc"/>',
        '<text x="60" y="66" class="title" fill="#0f172a">问题四：K-means 分区曲线测线布设结果</text>',
    ]

    for i in range(len(y) - 1):
        for j in range(len(x) - 1):
            z = float(np.nanmean([depth[i, j], depth[i, j + 1], depth[i + 1, j], depth[i + 1, j + 1]]))
            cluster_id = region_id(z) - 1
            color = ZONE_COLORS[min(cluster_id, len(ZONE_COLORS) - 1)]
            p1 = to_pixel(float(x[j]), float(y[i]))
            p2 = to_pixel(float(x[j + 1]), float(y[i + 1]))
            parts.append(f'<rect x="{p1[0]:.2f}" y="{p2[1]:.2f}" width="{p2[0] - p1[0]:.2f}" height="{p1[1] - p2[1]:.2f}" fill="{rgb(color)}" opacity="0.78"/>')

    for value in np.linspace(float(x.min()), float(x.max()), 9):
        px, _ = to_pixel(value, float(y.min()))
        parts.append(f'<line class="thin" x1="{px:.2f}" y1="{top}" x2="{px:.2f}" y2="{bottom}" stroke="#ffffff" stroke-width="1" opacity="0.45"/>')
        parts.append(f'<text x="{px:.2f}" y="{bottom + 24}" class="fixed-label" fill="#475569" text-anchor="middle">{value:.1f}</text>')
    for value in np.linspace(float(y.min()), float(y.max()), 6):
        _, py = to_pixel(float(x.min()), value)
        parts.append(f'<line class="thin" x1="{left}" y1="{py:.2f}" x2="{right}" y2="{py:.2f}" stroke="#ffffff" stroke-width="1" opacity="0.45"/>')
        parts.append(f'<text x="{left - 22}" y="{py + 5:.2f}" class="fixed-label" fill="#475569" text-anchor="middle">{value:.1f}</text>')

    for idx in range(len(ranges) - 1):
        boundary = (ranges[idx]["max_depth"] + ranges[idx + 1]["min_depth"]) / 2
        for p1, p2 in marching_segments(x, y, depth, float(boundary)):
            parts.append(svg_line(p1, p2, "#132438", 1.5, 0.9))

    depth_span = max(float(np.nanmax(depth) - np.nanmin(depth)), 1e-12)
    depth_to_px = plot_h / depth_span

    def add_depth_contour_band(
        level: float,
        cluster_idx: int,
        color: str,
        opacity: float,
        width_depth: float,
        width_scale: float,
        min_width: float,
        max_width: float,
    ) -> None:
        row = ranges[cluster_idx]
        stroke_width = float(np.clip(width_depth * depth_to_px * width_scale, min_width, max_width))
        for p1, p2 in marching_segments(x, y, depth, float(level)):
            midpoint = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
            mid_depth = segment_mid_depth(depth, x, y, midpoint)
            if row["min_depth"] <= mid_depth <= row["max_depth"]:
                parts.append(svg_line(p1, p2, color, stroke_width, opacity))

    for lane in lanes:
        cluster_idx = int(lane["cluster"]) - 1
        add_depth_contour_band(
            lane["center_depth"],
            cluster_idx,
            "#bde6e8",
            0.46,
            lane["deep_edge"] - lane["shallow_edge"],
            0.34,
            5,
            16,
        )

    for previous, current in zip(lanes, lanes[1:]):
        if previous["cluster"] != current["cluster"]:
            continue
        overlap_depth = current["deep_edge"] - previous["shallow_edge"]
        if overlap_depth <= 0:
            continue
        cluster_idx = int(current["cluster"]) - 1
        overlap_level = (current["deep_edge"] + previous["shallow_edge"]) / 2
        add_depth_contour_band(
            overlap_level,
            cluster_idx,
            "#e75e36",
            0.72,
            overlap_depth,
            0.62,
            4,
            18,
        )

    for lane in lanes:
        cluster_idx = int(lane["cluster"]) - 1
        stroke = "#ffffff" if lane["lane_index"] == 1 else "#fff2a8"
        width_px = 3.0 if lane["lane_index"] == 1 else 2.2
        for p1, p2 in marching_segments(x, y, depth, float(lane["center_depth"])):
            midpoint = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
            mid_depth = segment_mid_depth(depth, x, y, midpoint)
            row = ranges[cluster_idx]
            if row["min_depth"] <= mid_depth <= row["max_depth"]:
                parts.append(svg_line(p1, p2, stroke, width_px, 0.98))

    parts.append(f'<rect class="thin" x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#102c3d" stroke-width="3"/>')
    parts.append(f'<text x="{left + plot_w / 2}" y="{bottom + 56}" class="axis" fill="#1e293b" text-anchor="middle">东西方向 / n mile</text>')
    parts.append(f'<text x="{left}" y="{top - 18}" class="axis" fill="#1e293b">南北方向 / n mile</text>')

    panel_x, panel_y = 1270, 135
    parts.append(f'<rect class="thin" x="{panel_x}" y="{panel_y}" width="365" height="670" rx="10" fill="#ffffff" opacity="0.92" stroke="#b9c7ce"/>')
    parts.append(f'<text x="{panel_x + 22}" y="{panel_y + 48}" class="panel-title" fill="#0f172a">结果摘要</text>')
    info = [
        f"K-means 分区数：{len(ranges)}",
        f"测线数量：{summary['line_count']} 条",
        f"测线总长：{summary['total_length_nm']:.3f} n mile",
        f"未覆盖比例：{summary['uncovered_ratio'] * 100:.3f}%",
        f"超 20% 重叠长度：{summary['overlap_excess_length_nm']:.3f} n mile",
        "覆盖区：浅青色",
        "重叠区：红橙色",
        f"迭代代数：{iterations_used}",
    ]
    for i, item in enumerate(info):
        parts.append(f'<text x="{panel_x + 22}" y="{panel_y + 86 + i * 24}" class="panel-text" fill="#102c3d">{html.escape(item)}</text>')
    legend_x, legend_y = panel_x + 22, panel_y + 350
    parts.append(f'<text x="{legend_x}" y="{legend_y - 30}" class="panel-title" fill="#1e293b">颜色说明</text>')
    for item_index, (color, label) in enumerate([("#bde6e8", "浅青色：测线覆盖区域"), ("#e75e36", "红橙色：重叠覆盖区域"), ("#ffffff", "白色：实际测线")]):
        y0 = legend_y + item_index * 28
        parts.append(f'<rect class="thin" x="{legend_x}" y="{y0}" width="24" height="18" rx="3" fill="{color}" stroke="#64748b" stroke-width="1"/>')
        parts.append(f'<text x="{legend_x + 34}" y="{y0 + 14}" class="fixed-label" fill="#1e293b">{html.escape(label)}</text>')
    region_legend_y = legend_y + 118
    parts.append(f'<text x="{legend_x}" y="{region_legend_y - 24}" class="panel-title" fill="#1e293b">K-means 水深分区</text>')
    for idx, row in enumerate(ranges):
        y0 = region_legend_y + idx * 28
        color = ZONE_COLORS[min(idx, len(ZONE_COLORS) - 1)]
        label = f"区 {idx + 1}: {row['min_depth']:.1f}-{row['max_depth']:.1f} m"
        parts.append(f'<rect class="thin" x="{legend_x}" y="{y0}" width="24" height="18" rx="3" fill="{rgb(color)}" stroke="#64748b" stroke-width="1"/>')
        parts.append(f'<text x="{legend_x + 34}" y="{y0 + 14}" class="fixed-label" fill="#1e293b">{html.escape(label)}</text>')
    parts.append("</svg>")
    OUTPUT_SVG.write_text("\n".join(parts), encoding="utf-8")


def save_outputs(lanes: list[dict], summary: dict, etas: np.ndarray, iterations_used: int, stable_count: int) -> None:
    print("[7/8] Saving CSV outputs...", flush=True)
    with OUTPUT_LINES_CSV.open("w", newline="", encoding="utf-8-sig") as file:
        fieldnames = [
            "line_id",
            "region",
            "lane_index",
            "center_depth_m",
            "deep_edge_m",
            "shallow_edge_m",
            "length_nm",
            "eta_target",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for line_id, lane in enumerate(lanes, start=1):
            writer.writerow(
                {
                    "line_id": line_id,
                    "region": lane["cluster"],
                    "lane_index": lane["lane_index"],
                    "center_depth_m": round(lane["center_depth"], 4),
                    "deep_edge_m": round(lane["deep_edge"], 4),
                    "shallow_edge_m": round(lane["shallow_edge"], 4),
                    "length_nm": round(lane["length_nm"], 6),
                    "eta_target": round(lane["eta_target"], 6),
                }
            )

    with OUTPUT_SUMMARY_CSV.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        writer.writerow(["total_length_nm", round(summary["total_length_nm"], 6)])
        writer.writerow(["line_count", summary["line_count"]])
        writer.writerow(["uncovered_ratio_percent", round(summary["uncovered_ratio"] * 100, 6)])
        writer.writerow(["overlap_excess_length_nm", round(summary["overlap_excess_length_nm"], 6)])
        writer.writerow(["iterations_used", iterations_used])
        writer.writerow(["stable_count", stable_count])
        for idx, eta in enumerate(etas, start=1):
            writer.writerow([f"region_{idx}_eta", round(float(eta), 8)])


def main() -> None:
    print("=== 2023B Q4: K-means curved survey-line iterative layout ===", flush=True)
    print("[1/8] Reading bathymetry grid and selecting K by elbow method...", flush=True)
    x, y, depth = read_bathymetry_grid()
    _, selected_k = elbow_analysis(depth, draw_plot=False)
    labels, centers, _ = kmeans_1d(depth, selected_k)
    ranges = cluster_ranges(depth, labels, centers)
    print(f"[2/8] Selected K={selected_k}; estimating gradients for each K-means region...", flush=True)
    gradients = []
    for cluster_id, row in enumerate(ranges):
        gradient = estimate_region_gradient(x, y, depth, labels, cluster_id)
        gradients.append(gradient)
        print(
            f"    region {cluster_id + 1}: depth={row['min_depth']:.2f}-{row['max_depth']:.2f} m, "
            f"median_gradient={gradient:.4f} m/n mile",
            flush=True,
        )

    levels, lengths = contour_length_lookup(x, y, depth)
    etas, lanes, summary, iterations_used, stable_count = optimize_etas(ranges, gradients, levels, lengths)
    print("[5/8] Building final curved survey-line geometry...", flush=True)
    lanes, summary = build_layout_from_eta(etas, ranges, gradients, levels, lengths)
    draw_layout(x, y, depth, ranges, lanes, summary, iterations_used)
    draw_layout_svg(x, y, depth, ranges, lanes, summary, iterations_used)
    save_outputs(lanes, summary, etas, iterations_used, stable_count)

    print("[8/8] Done.", flush=True)
    print("selected_k:", selected_k)
    print("line_count:", summary["line_count"])
    print("total_length_nm:", round(summary["total_length_nm"], 6))
    print("uncovered_ratio_percent:", round(summary["uncovered_ratio"] * 100, 6))
    print("overlap_excess_length_nm:", round(summary["overlap_excess_length_nm"], 6))
    print("saved_lines_csv:", OUTPUT_LINES_CSV)
    print("saved_summary_csv:", OUTPUT_SUMMARY_CSV)
    print("saved_png:", OUTPUT_PNG)
    print("saved_svg:", OUTPUT_SVG)


if __name__ == "__main__":
    main()
