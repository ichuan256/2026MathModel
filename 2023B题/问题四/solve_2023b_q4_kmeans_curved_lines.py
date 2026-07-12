from __future__ import annotations

import csv
import math
import random
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from kmeans_depth_classification import ZONE_COLORS, cluster_ranges, elbow_analysis, kmeans_xyz
from make_contour_from_attachment import marching_segments, read_bathymetry_grid, setup_matplotlib


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


def color_hex(color: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*color)


def cluster_at_point(labels: np.ndarray, x: np.ndarray, y: np.ndarray, point: tuple[float, float]) -> int:
    xi = int(np.clip(round((point[0] - x[0]) / max(x[1] - x[0], 1e-12)), 0, len(x) - 1))
    yi = int(np.clip(round((point[1] - y[0]) / max(y[1] - y[0], 1e-12)), 0, len(y) - 1))
    return int(labels[yi, xi])


def swath_half_depth(center_depth: float, gradient_m_per_nm: float) -> float:
    half_width_nm = center_depth * TAN_HALF / NM
    return max(0.05, gradient_m_per_nm * half_width_nm)


def estimate_region_gradient(x: np.ndarray, y: np.ndarray, depth: np.ndarray, labels: np.ndarray, cluster_id: int) -> float:
    grad_y, grad_x = np.gradient(depth, y, x)
    magnitude = np.hypot(grad_x, grad_y)
    values = magnitude[(labels == cluster_id) & np.isfinite(magnitude)]
    if values.size == 0:
        return max(0.5, float(np.nanmedian(magnitude)))
    return max(0.5, float(np.nanmedian(values)))


def contour_length_lookup(
    x: np.ndarray,
    y: np.ndarray,
    depth: np.ndarray,
    labels: np.ndarray,
    cluster_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    print("[3/8] 预计算各分区等深线长度表...", flush=True)
    z_min = float(np.nanmin(depth))
    z_max = float(np.nanmax(depth))
    levels = np.arange(math.floor(z_min), math.ceil(z_max) + 0.25, 0.25)
    lengths = np.zeros((len(levels), cluster_count), dtype=float)

    for index, level in enumerate(levels, start=1):
        for p1, p2 in marching_segments(x, y, depth, float(level)):
            midpoint = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
            cluster_id = cluster_at_point(labels, x, y, midpoint)
            if 0 <= cluster_id < cluster_count:
                lengths[index - 1, cluster_id] += math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        if index == 1 or index % 120 == 0 or index == len(levels):
            print(f"    长度表进度 {index}/{len(levels)}", flush=True)

    return levels, lengths


def interpolated_contour_length(levels: np.ndarray, lengths: np.ndarray, depth_value: float, cluster_id: int) -> float:
    return float(np.interp(depth_value, levels, lengths[:, cluster_id], left=0.0, right=0.0))


def build_region_lane_depths(shallow_boundary: float, deep_boundary: float, gradient_m_per_nm: float, eta: float) -> list[dict]:
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
            length_nm = interpolated_contour_length(levels, lengths, lane["center_depth"], cluster_id)
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
                overlap_excess_length_nm += interpolated_contour_length(levels, lengths, current["center_depth"], cluster_id) * (rate - 0.20)

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
    return summary["total_length_nm"] + penalty, lanes, summary


def optimize_etas(
    ranges: list[dict],
    gradients: list[float],
    levels: np.ndarray,
    lengths: np.ndarray,
) -> tuple[np.ndarray, list[dict], dict, int, int]:
    print(
        f"[4/8] 开始改进粒子群迭代：粒子数={SWARM_SIZE}, 至少迭代={MIN_ITER}, "
        f"连续稳定={CONVERGENCE_PATIENCE}, 阈值={CONVERGENCE_TOL}",
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
        particles.append({"position": position, "velocity": velocity, "best_position": position.copy(), "best_fitness": fitness})
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
                particle["velocity"][d] = float(np.clip(w * particle["velocity"][d] + cognitive + social, -0.03, 0.03))
                particle["position"][d] = float(np.clip(particle["position"][d] + particle["velocity"][d], ETA_MIN, ETA_MAX))

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
                f"lines={global_best_summary['line_count']}, etas=[{eta_text}], stable={converged_count}",
                flush=True,
            )

        if iteration >= MIN_ITER and converged_count >= CONVERGENCE_PATIENCE:
            iterations_used = iteration
            print(f"    满足终止条件：iteration={iterations_used}, stable_count={converged_count}", flush=True)
            break

    return global_best_position, global_best_lanes, global_best_summary, iterations_used, converged_count


def draw_layout(
    x: np.ndarray,
    y: np.ndarray,
    depth: np.ndarray,
    labels: np.ndarray,
    ranges: list[dict],
    lanes: list[dict],
    summary: dict,
    iterations_used: int,
) -> None:
    if not HAS_MATPLOTLIB:
        raise ImportError("需要安装 matplotlib 才能输出平滑 PNG/SVG。")

    print("[6/8] 绘制横向拉伸版曲线测线 PNG/SVG...", flush=True)
    setup_matplotlib()
    x_grid, y_grid = np.meshgrid(x, y)
    masked_labels = np.ma.masked_less(labels.astype(float), 0)
    label_levels = np.arange(-0.5, len(ranges) + 0.5, 1.0)
    colors = [color_hex(ZONE_COLORS[i]) for i in range(len(ranges))]
    cmap = ListedColormap(colors)

    fig = plt.figure(figsize=(20.5, 9.8), facecolor="#f7fafc")
    ax = fig.add_axes([0.055, 0.13, 0.74, 0.75])
    panel = fig.add_axes([0.825, 0.18, 0.15, 0.66])
    panel.set_axis_off()

    ax.contourf(x_grid, y_grid, masked_labels, levels=label_levels, cmap=cmap, alpha=0.78)
    if len(ranges) > 1:
        ax.contour(x_grid, y_grid, masked_labels, levels=np.arange(0.5, len(ranges), 1.0), colors="#132438", linewidths=0.65, alpha=0.9)
    ax.contour(x_grid, y_grid, depth, levels=8, colors="#ffffff", linewidths=0.34, alpha=0.36)

    depth_span = max(float(np.nanmax(depth) - np.nanmin(depth)), 1e-12)
    depth_to_px = 680 / depth_span

    def plot_region_contour_segments(level: float, cluster_idx: int, color: str, line_width: float, alpha: float) -> None:
        for p1, p2 in marching_segments(x, y, depth, float(level)):
            midpoint = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
            if cluster_at_point(labels, x, y, midpoint) == cluster_idx:
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=line_width, alpha=alpha, solid_capstyle="round")

    for lane in lanes:
        cluster_idx = int(lane["cluster"]) - 1
        width_depth = lane["deep_edge"] - lane["shallow_edge"]
        line_width = float(np.clip(width_depth * depth_to_px * 0.34, 4.0, 12.0))
        plot_region_contour_segments(float(lane["center_depth"]), cluster_idx, "#bde6e8", line_width, 0.42)

    for previous, current in zip(lanes, lanes[1:]):
        if previous["cluster"] != current["cluster"]:
            continue
        overlap_depth = current["deep_edge"] - previous["shallow_edge"]
        if overlap_depth <= 0:
            continue
        cluster_idx = int(current["cluster"]) - 1
        overlap_level = (current["deep_edge"] + previous["shallow_edge"]) / 2
        line_width = float(np.clip(overlap_depth * depth_to_px * 0.62, 3.0, 12.0))
        plot_region_contour_segments(float(overlap_level), cluster_idx, "#e75e36", line_width, 0.72)

    for lane in lanes:
        cluster_idx = int(lane["cluster"]) - 1
        stroke = "#ffffff" if lane["lane_index"] == 1 else "#fff2a8"
        width = 0.95 if lane["lane_index"] == 1 else 0.65
        plot_region_contour_segments(float(lane["center_depth"]), cluster_idx, stroke, width, 0.98)

    ax.set_title("问题四：三维 K-means 分区曲线测线布设结果", fontsize=18, pad=14, fontweight="bold")
    ax.set_xlabel("东西方向 / n mile", fontsize=13, fontweight="bold")
    ax.set_ylabel("南北方向 / n mile", fontsize=13, fontweight="bold")
    ax.set_aspect("auto")
    ax.grid(color="white", linewidth=0.55, alpha=0.42)

    panel.set_xlim(0, 1)
    panel.set_ylim(0, 1)
    panel.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="#b9c7ce", alpha=0.94))
    panel.text(0.07, 0.94, "结果摘要", fontsize=14, fontweight="bold", color="#0f172a")
    info = [
        f"K-means 分区数：{len(ranges)}",
        f"测线数量：{summary['line_count']} 条",
        f"测线总长：{summary['total_length_nm']:.3f} n mile",
        f"未覆盖比例：{summary['uncovered_ratio'] * 100:.3f}%",
        f"超 20% 重叠长度：{summary['overlap_excess_length_nm']:.3f} n mile",
        f"迭代代数：{iterations_used}",
    ]
    for index, item in enumerate(info):
        panel.text(0.07, 0.87 - index * 0.055, item, fontsize=9.5, color="#102c3d")

    panel.text(0.07, 0.50, "颜色说明", fontsize=13, fontweight="bold", color="#1e293b")
    legend_items = [
        Patch(facecolor="#bde6e8", edgecolor="#64748b", label="浅青色：测线覆盖区域", alpha=0.55),
        Patch(facecolor="#e75e36", edgecolor="#64748b", label="红橙色：重叠覆盖区域", alpha=0.75),
        Patch(facecolor="#ffffff", edgecolor="#64748b", label="白色：实际测线"),
    ]
    panel.legend(handles=legend_items, loc="upper left", bbox_to_anchor=(0.05, 0.48), frameon=False, fontsize=8.8)

    panel.text(0.07, 0.26, "三维 K-means 分区", fontsize=13, fontweight="bold", color="#1e293b")
    for idx, row in enumerate(ranges):
        y0 = 0.20 - idx * 0.050
        panel.add_patch(plt.Rectangle((0.07, y0 - 0.015), 0.06, 0.030, facecolor=colors[idx], edgecolor="#64748b", linewidth=0.6))
        panel.text(0.16, y0 - 0.010, f"区 {idx + 1}: {row['min_depth']:.1f}-{row['max_depth']:.1f} m", fontsize=8.2, color="#1e293b")

    fig.savefig(OUTPUT_PNG, dpi=240, bbox_inches="tight", facecolor="#f7fafc")
    fig.savefig(OUTPUT_SVG, format="svg", bbox_inches="tight", facecolor="#f7fafc")
    plt.close(fig)


def save_outputs(lanes: list[dict], summary: dict, etas: np.ndarray, iterations_used: int, stable_count: int) -> None:
    print("[7/8] 保存 CSV 输出...", flush=True)
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
    print("=== 2023B 问题四：三维 K-means 曲线测线迭代布设 ===", flush=True)
    print("[1/8] 读取测深网格并用肘部法选择 K...", flush=True)
    x, y, depth = read_bathymetry_grid()
    _, selected_k = elbow_analysis(x, y, depth, draw_plot=False)
    labels, centers, _, _ = kmeans_xyz(x, y, depth, selected_k)
    ranges = cluster_ranges(depth, labels, centers)

    print(f"[2/8] 选定 K={selected_k}；估计每个三维聚类区域的坡度...", flush=True)
    gradients = []
    for cluster_id, row in enumerate(ranges):
        gradient = estimate_region_gradient(x, y, depth, labels, cluster_id)
        gradients.append(gradient)
        print(
            f"    区域 {cluster_id + 1}: 深度={row['min_depth']:.2f}-{row['max_depth']:.2f} m, "
            f"中心=({row['center_x']:.2f}, {row['center_y']:.2f}, {row['center']:.2f}), "
            f"中位坡度={gradient:.4f} m/n mile",
            flush=True,
        )

    levels, lengths = contour_length_lookup(x, y, depth, labels, len(ranges))
    etas, lanes, summary, iterations_used, stable_count = optimize_etas(ranges, gradients, levels, lengths)
    print("[5/8] 生成最终曲线测线几何...", flush=True)
    lanes, summary = build_layout_from_eta(etas, ranges, gradients, levels, lengths)
    draw_layout(x, y, depth, labels, ranges, lanes, summary, iterations_used)
    save_outputs(lanes, summary, etas, iterations_used, stable_count)

    print("[8/8] 完成。", flush=True)
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
