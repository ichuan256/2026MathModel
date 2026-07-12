from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from make_contour_from_attachment import CONTOUR_PNG_PATH, read_bathymetry_grid, setup_matplotlib


BASE_DIR = Path(__file__).resolve().parent
CLUSTER_CSV_PATH = BASE_DIR / "attachment_kmeans_depth_clusters.csv"
CLUSTER_PNG_PATH = BASE_DIR / "attachment_kmeans_depth_zones.png"
CLUSTER_SVG_PATH = BASE_DIR / "attachment_kmeans_depth_zones.svg"
ELBOW_CSV_PATH = BASE_DIR / "attachment_kmeans_elbow.csv"

K_MIN = 2
K_MAX = 10
MANUAL_K = None
MAX_ITER = 200
TOL = 1e-8

# 三维特征全部参与聚类；水深略加权，避免分区只按空间切块。
FEATURE_WEIGHTS = np.asarray([1.0, 1.0, 1.6], dtype=float)

ZONE_NAMES = ["极浅水区", "浅水区", "中等水深区", "深水区", "极深水区"]
ZONE_COLORS = [
    (232, 244, 198),
    (164, 215, 192),
    (93, 180, 190),
    (53, 121, 170),
    (30, 61, 119),
    (89, 55, 127),
    (140, 69, 112),
    (184, 92, 87),
    (207, 134, 82),
    (221, 181, 96),
]


def color_hex(color: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*color)


def build_feature_matrix(x: np.ndarray, y: np.ndarray, depth: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_grid, y_grid = np.meshgrid(x, y)
    valid_mask = np.isfinite(depth)
    raw_features = np.column_stack([x_grid[valid_mask], y_grid[valid_mask], depth[valid_mask]]).astype(float)
    means = raw_features.mean(axis=0)
    scales = raw_features.std(axis=0)
    scales[scales < 1e-12] = 1.0
    features = (raw_features - means) / scales * FEATURE_WEIGHTS
    return features, raw_features, valid_mask


def kmeans_xyz(x: np.ndarray, y: np.ndarray, depth: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, int, float]:
    features, raw_features, valid_mask = build_feature_matrix(x, y, depth)
    if features.shape[0] < k:
        raise ValueError("有效测深点数量不足，无法进行 K-means 聚类。")

    depth_order = np.argsort(raw_features[:, 2])
    seed_positions = np.linspace(0, len(depth_order) - 1, k + 2, dtype=int)[1:-1]
    centers = features[depth_order[seed_positions]].copy()
    labels = np.zeros(features.shape[0], dtype=int)

    iterations = 0
    for iterations in range(1, MAX_ITER + 1):
        distances = np.linalg.norm(features[:, None, :] - centers[None, :, :], axis=2)
        new_labels = np.argmin(distances, axis=1)
        new_centers = centers.copy()
        for cluster_id in range(k):
            members = features[new_labels == cluster_id]
            if members.size:
                new_centers[cluster_id] = members.mean(axis=0)
        shift = float(np.max(np.linalg.norm(new_centers - centers, axis=1)))
        centers = new_centers
        labels = new_labels
        if shift < TOL:
            break

    raw_centers = np.vstack([raw_features[labels == cluster_id].mean(axis=0) for cluster_id in range(k)])
    order = np.argsort(raw_centers[:, 2])
    remap = np.empty_like(order)
    remap[order] = np.arange(k)
    sorted_labels = remap[labels]
    sorted_raw_centers = raw_centers[order]

    full_labels = np.full(depth.shape, -1, dtype=int)
    full_labels[valid_mask] = sorted_labels
    sorted_feature_centers = np.vstack([features[sorted_labels == cluster_id].mean(axis=0) for cluster_id in range(k)])
    residuals = features - sorted_feature_centers[sorted_labels]
    sse = float(np.sum(residuals * residuals))
    return full_labels, sorted_raw_centers, iterations, sse


def select_k_by_elbow(rows: list[dict]) -> int:
    xs = np.asarray([row["k"] for row in rows], dtype=float)
    ys = np.asarray([row["sse"] for row in rows], dtype=float)
    x1, y1 = xs[0], ys[0]
    x2, y2 = xs[-1], ys[-1]
    denominator = max(float(np.hypot(y2 - y1, x2 - x1)), 1e-12)
    distances = np.abs((y2 - y1) * xs - (x2 - x1) * ys + x2 * y1 - y2 * x1) / denominator
    return int(xs[int(np.argmax(distances))])


def elbow_analysis(x: np.ndarray, y: np.ndarray, depth: np.ndarray, draw_plot: bool = False) -> tuple[list[dict], int]:
    print(f"[2/6] 肘部法分析三维 K-means：K={K_MIN}..{K_MAX}...", flush=True)
    rows = []
    for k in range(K_MIN, K_MAX + 1):
        _, _, iterations, sse = kmeans_xyz(x, y, depth, k)
        rows.append({"k": k, "sse": sse, "iterations": iterations})
        print(f"    K={k}, 三维标准化 SSE={sse:.4f}, iterations={iterations}", flush=True)

    selected_k = int(MANUAL_K) if MANUAL_K is not None else select_k_by_elbow(rows)
    with ELBOW_CSV_PATH.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=["k", "sse", "iterations", "selected", "feature_model"])
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "selected": 1 if row["k"] == selected_k else 0, "feature_model": "standardized_x_y_depth"})
    return rows, selected_k


def cluster_ranges(depth: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> list[dict]:
    ranges = []
    valid_count = max(int(np.isfinite(depth).sum()), 1)
    for cluster_id, center in enumerate(centers):
        members = depth[labels == cluster_id]
        ranges.append(
            {
                "cluster": cluster_id + 1,
                "name": ZONE_NAMES[cluster_id] if cluster_id < len(ZONE_NAMES) else f"区域 {cluster_id + 1}",
                "center_x": float(center[0]),
                "center_y": float(center[1]),
                "center": float(center[2]),
                "min_depth": float(np.nanmin(members)),
                "max_depth": float(np.nanmax(members)),
                "count": int(members.size),
                "ratio": float(members.size / valid_count),
            }
        )
    return ranges


def export_cluster_points(x: np.ndarray, y: np.ndarray, depth: np.ndarray, labels: np.ndarray, ranges: list[dict]) -> None:
    print("[3/6] 导出三维 K-means 分区点表...", flush=True)
    zone_names = {row["cluster"] - 1: row["name"] for row in ranges}
    with CLUSTER_CSV_PATH.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["x_nm", "y_nm", "depth_m", "cluster", "zone_name"])
        for i, y_value in enumerate(y):
            for j, x_value in enumerate(x):
                if labels[i, j] >= 0:
                    cluster = int(labels[i, j]) + 1
                    writer.writerow([round(float(x_value), 6), round(float(y_value), 6), round(float(depth[i, j]), 4), cluster, zone_names[int(labels[i, j])]])


def draw_kmeans_figures(x: np.ndarray, y: np.ndarray, depth: np.ndarray, labels: np.ndarray, ranges: list[dict]) -> None:
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for smooth K-means PNG/SVG output.")

    print("[4/6] 绘制横向拉伸版三维 K-means 分区 PNG/SVG...", flush=True)
    setup_matplotlib()
    x_grid, y_grid = np.meshgrid(x, y)
    masked_labels = np.ma.masked_less(labels.astype(float), 0)
    levels = np.arange(-0.5, len(ranges) + 0.5, 1.0)
    colors = [color_hex(ZONE_COLORS[i]) for i in range(len(ranges))]
    cmap = ListedColormap(colors)

    fig = plt.figure(figsize=(20.5, 9.5), facecolor="#f7fafc")
    ax = fig.add_axes([0.055, 0.13, 0.74, 0.76])
    panel = fig.add_axes([0.825, 0.20, 0.15, 0.60])
    panel.set_axis_off()

    ax.contourf(x_grid, y_grid, masked_labels, levels=levels, cmap=cmap, alpha=0.94)
    if len(ranges) > 1:
        ax.contour(x_grid, y_grid, masked_labels, levels=np.arange(0.5, len(ranges), 1.0), colors="#152336", linewidths=0.65, alpha=0.9)
    ax.contour(x_grid, y_grid, depth, levels=8, colors="#ffffff", linewidths=0.35, alpha=0.45)

    ax.set_title("K-means 三维特征分区图（x、y、水深）", fontsize=18, pad=14, fontweight="bold")
    ax.set_xlabel("东西方向 / n mile", fontsize=13, fontweight="bold")
    ax.set_ylabel("南北方向 / n mile", fontsize=13, fontweight="bold")
    ax.set_aspect("auto")
    ax.grid(color="white", linewidth=0.55, alpha=0.42)

    panel.set_xlim(0, 1)
    panel.set_ylim(0, 1)
    panel.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="#b9c7ce", alpha=0.94))
    panel.text(0.08, 0.93, "结果摘要", fontsize=15, fontweight="bold", color="#0f172a")
    panel.text(0.08, 0.83, f"K = {len(ranges)}", fontsize=11, color="#102c3d")
    panel.text(0.08, 0.75, f"有效测深点：{np.isfinite(depth).sum()}", fontsize=11, color="#102c3d")
    panel.text(0.08, 0.67, "聚类特征：x、y、水深", fontsize=11, color="#102c3d")
    panel.text(0.08, 0.57, "分区说明", fontsize=13, fontweight="bold", color="#1e293b")
    for idx, row in enumerate(ranges):
        y0 = 0.49 - idx * 0.071
        panel.add_patch(plt.Rectangle((0.08, y0 - 0.02), 0.07, 0.035, facecolor=colors[idx], edgecolor="#64748b", linewidth=0.7))
        panel.text(0.18, y0 - 0.01, f"{row['cluster']}. {row['min_depth']:.1f}-{row['max_depth']:.1f} m ({row['ratio'] * 100:.1f}%)", fontsize=9.0, color="#1e293b")

    fig.savefig(CLUSTER_PNG_PATH, dpi=240, bbox_inches="tight", facecolor="#f7fafc")
    fig.savefig(CLUSTER_SVG_PATH, format="svg", bbox_inches="tight", facecolor="#f7fafc")
    plt.close(fig)


def main() -> None:
    print("=== 2023B 问题四：三维 K-means 分区 ===", flush=True)
    print("[1/6] 读取测深网格...", flush=True)
    x, y, depth = read_bathymetry_grid()
    _, selected_k = elbow_analysis(x, y, depth)
    print(f"[4/6] 使用 K={selected_k} 进行最终三维 K-means 聚类...", flush=True)
    labels, centers, iterations, _ = kmeans_xyz(x, y, depth, selected_k)
    ranges = cluster_ranges(depth, labels, centers)
    export_cluster_points(x, y, depth, labels, ranges)
    draw_kmeans_figures(x, y, depth, labels, ranges)

    print("[6/6] 完成。", flush=True)
    print("selected_k:", selected_k)
    print("iterations:", iterations)
    for row in ranges:
        print(
            f"cluster_{row['cluster']}: center=({row['center_x']:.3f}, {row['center_y']:.3f}, {row['center']:.3f}), "
            f"depth_range={row['min_depth']:.2f}-{row['max_depth']:.2f} m, ratio={row['ratio'] * 100:.2f}%"
        )
    print("source_contour_png:", CONTOUR_PNG_PATH)
    print("saved_elbow_csv:", ELBOW_CSV_PATH)
    print("saved_csv:", CLUSTER_CSV_PATH)
    print("saved_png:", CLUSTER_PNG_PATH)
    print("saved_svg:", CLUSTER_SVG_PATH)


if __name__ == "__main__":
    main()
