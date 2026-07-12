import csv
import html
from pathlib import Path

import numpy as np

from make_contour_from_attachment import (
    CONTOUR_PNG_PATH,
    HAS_PIL,
    load_font,
    marching_segments,
    read_bathymetry_grid,
)


if not HAS_PIL:
    raise SystemExit("Pillow is required for drawing the classified map.")

from PIL import Image, ImageDraw


BASE_DIR = Path(__file__).resolve().parent
CLUSTER_CSV_PATH = BASE_DIR / "attachment_kmeans_depth_clusters.csv"
CLUSTER_PNG_PATH = BASE_DIR / "attachment_kmeans_depth_zones.png"
CLUSTER_SVG_PATH = BASE_DIR / "attachment_kmeans_depth_zones.svg"
ELBOW_CSV_PATH = BASE_DIR / "attachment_kmeans_elbow.csv"
ELBOW_PNG_PATH = BASE_DIR / "attachment_kmeans_elbow.png"

K_MIN = 2
K_MAX = 10
MANUAL_K = None
MAX_ITER = 200
TOL = 1e-8

ZONE_NAMES = [
    "极浅水区",
    "浅水区",
    "中等水深区",
    "深水区",
    "极深水区",
]
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


def kmeans_1d(values: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Cluster one-dimensional depth values using deterministic K-means."""
    valid_values = values[np.isfinite(values)].reshape(-1)
    if valid_values.size < k:
        raise ValueError("Not enough valid depth points for K-means clustering.")

    percentiles = np.linspace(0, 100, k + 2)[1:-1]
    centers = np.percentile(valid_values, percentiles)
    labels = np.zeros(valid_values.shape[0], dtype=int)

    iterations = 0
    for iterations in range(1, MAX_ITER + 1):
        distances = np.abs(valid_values[:, None] - centers[None, :])
        new_labels = np.argmin(distances, axis=1)
        new_centers = centers.copy()

        for cluster_id in range(k):
            members = valid_values[new_labels == cluster_id]
            if members.size:
                new_centers[cluster_id] = members.mean()

        shift = float(np.max(np.abs(new_centers - centers)))
        centers = new_centers
        labels = new_labels
        if shift < TOL:
            break

    order = np.argsort(centers)
    remap = np.empty_like(order)
    remap[order] = np.arange(k)
    sorted_centers = centers[order]
    sorted_labels = remap[labels]

    full_labels = np.full(values.shape, -1, dtype=int)
    full_labels[np.isfinite(values)] = sorted_labels
    return full_labels, sorted_centers, iterations


def compute_sse(values: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> float:
    valid_values = values[np.isfinite(values)]
    valid_labels = labels[np.isfinite(values)]
    residuals = valid_values - centers[valid_labels]
    return float(np.sum(residuals * residuals))


def elbow_analysis(depth: np.ndarray, draw_plot: bool = False) -> tuple[list[dict], int]:
    print(f"[2/6] Running elbow analysis for K={K_MIN}..{K_MAX}...", flush=True)
    rows = []
    for k in range(K_MIN, K_MAX + 1):
        labels, centers, iterations = kmeans_1d(depth, k)
        sse = compute_sse(depth, labels, centers)
        rows.append({"k": k, "sse": sse, "iterations": iterations})
        print(f"    K={k}, SSE={sse:.4f}, iterations={iterations}", flush=True)

    if MANUAL_K is not None:
        selected_k = int(MANUAL_K)
    else:
        selected_k = select_k_by_elbow(rows)

    with ELBOW_CSV_PATH.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=["k", "sse", "iterations", "selected"])
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "selected": 1 if row["k"] == selected_k else 0})

    if draw_plot:
        draw_elbow_plot(rows, selected_k)
    return rows, selected_k


def select_k_by_elbow(rows: list[dict]) -> int:
    """Select the elbow by the maximum distance to the line from first to last SSE point."""
    xs = np.asarray([row["k"] for row in rows], dtype=float)
    ys = np.asarray([row["sse"] for row in rows], dtype=float)
    x1, y1 = xs[0], ys[0]
    x2, y2 = xs[-1], ys[-1]
    denominator = max(float(np.hypot(y2 - y1, x2 - x1)), 1e-12)
    distances = np.abs((y2 - y1) * xs - (x2 - x1) * ys + x2 * y1 - y2 * x1) / denominator
    return int(xs[int(np.argmax(distances))])


def draw_elbow_plot(rows: list[dict], selected_k: int) -> None:
    print("[3/6] Drawing elbow curve...", flush=True)
    width, height = 1250, 720
    left, top, plot_w, plot_h = 110, 105, 780, 470
    right, bottom = left + plot_w, top + plot_h
    k_values = [row["k"] for row in rows]
    sse_values = [row["sse"] for row in rows]
    sse_min, sse_max = min(sse_values), max(sse_values)

    def to_pixel(k: float, sse: float) -> tuple[float, float]:
        px = left + (k - min(k_values)) / max(max(k_values) - min(k_values), 1e-12) * plot_w
        py = bottom - (sse - sse_min) / max(sse_max - sse_min, 1e-12) * plot_h
        return px, py

    image = Image.new("RGB", (width, height), (248, 250, 252))
    draw = ImageDraw.Draw(image, "RGBA")
    title_font = load_font(28, bold=True)
    label_font = load_font(18, bold=True)
    text_font = load_font(15)
    small_font = load_font(13)

    draw.text((55, 36), "K-means 水深分类肘部法选 K", font=title_font, fill=(15, 23, 42))

    for frac in np.linspace(0, 1, 6):
        py = top + frac * plot_h
        sse_tick = sse_max - frac * (sse_max - sse_min)
        draw.line((left, py, right, py), fill=(203, 213, 225, 160), width=1)
        draw.text((30, py - 8), f"{sse_tick/1_000_000:.2f}M", font=small_font, fill=(71, 85, 105))

    for k in k_values:
        px, _ = to_pixel(k, sse_min)
        draw.line((px, top, px, bottom), fill=(226, 232, 240, 130), width=1)
        draw.text((px - 7, bottom + 12), str(k), font=small_font, fill=(71, 85, 105))

    points = [to_pixel(row["k"], row["sse"]) for row in rows]
    for p1, p2 in zip(points, points[1:]):
        draw.line((*p1, *p2), fill=(37, 99, 235, 235), width=4)

    for row, point in zip(rows, points):
        fill = (220, 38, 38, 255) if row["k"] == selected_k else (37, 99, 235, 255)
        radius = 8 if row["k"] == selected_k else 6
        draw.ellipse((point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius), fill=fill)
        if row["k"] == selected_k:
            draw.text((point[0] + 12, point[1] - 18), f"K={selected_k}", font=text_font, fill=(127, 29, 29))

    draw.rectangle((left, top, right, bottom), outline=(15, 23, 42, 255), width=2)
    draw.text((left + plot_w / 2 - 90, bottom + 48), "聚类数量 K", font=label_font, fill=(30, 41, 59))
    draw.text((left, top - 32), "SSE", font=label_font, fill=(30, 41, 59))

    info = (
        f"候选 K：{K_MIN}-{K_MAX}\n"
        f"选取 K：{selected_k}\n"
        "判据：到首尾 SSE 连线距离最大"
    )
    panel_x, panel_y, panel_w, panel_h = 925, 105, 270, 180
    draw.rounded_rectangle((panel_x, panel_y, panel_x + panel_w, panel_y + panel_h), radius=10, fill=(255, 255, 255, 235), outline=(185, 199, 206, 255))
    draw.text((panel_x + 20, panel_y + 20), "结果摘要", font=label_font, fill=(15, 23, 42))
    draw.multiline_text((panel_x + 20, panel_y + 66), info, font=text_font, fill=(16, 44, 61), spacing=8)

    image.save(ELBOW_PNG_PATH)


def cluster_ranges(depth: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> list[dict]:
    ranges = []
    for cluster_id, center in enumerate(centers):
        members = depth[labels == cluster_id]
        ranges.append(
            {
                "cluster": cluster_id + 1,
                "name": ZONE_NAMES[cluster_id] if cluster_id < len(ZONE_NAMES) else f"区域 {cluster_id + 1}",
                "center": float(center),
                "min_depth": float(np.nanmin(members)),
                "max_depth": float(np.nanmax(members)),
                "count": int(members.size),
                "ratio": float(members.size / np.isfinite(depth).sum()),
            }
        )
    return ranges


def export_cluster_points(
    x: np.ndarray, y: np.ndarray, depth: np.ndarray, labels: np.ndarray, ranges: list[dict]
) -> None:
    print("[3/5] Exporting classified point table...", flush=True)
    zone_names = {row["cluster"] - 1: row["name"] for row in ranges}
    with CLUSTER_CSV_PATH.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["x_nm", "y_nm", "depth_m", "cluster", "zone_name"])
        for i, y_value in enumerate(y):
            for j, x_value in enumerate(x):
                if labels[i, j] >= 0:
                    cluster = int(labels[i, j]) + 1
                    writer.writerow(
                        [
                            round(float(x_value), 6),
                            round(float(y_value), 6),
                            round(float(depth[i, j]), 4),
                            cluster,
                            zone_names[int(labels[i, j])],
                        ]
                    )


def draw_kmeans_map(
    x: np.ndarray, y: np.ndarray, depth: np.ndarray, labels: np.ndarray, ranges: list[dict]
) -> None:
    print("[4/5] Drawing K-means depth-zone map...", flush=True)
    width, height = 1650, 950
    left, top, plot_w, plot_h = 105, 130, 1080, 650
    right, bottom = left + plot_w, top + plot_h

    def to_pixel(px: float, py: float) -> tuple[float, float]:
        xx = left + (px - float(x.min())) / max(float(x.max() - x.min()), 1e-12) * plot_w
        yy = bottom - (py - float(y.min())) / max(float(y.max() - y.min()), 1e-12) * plot_h
        return xx, yy

    image = Image.new("RGB", (width, height), (247, 250, 252))
    draw = ImageDraw.Draw(image, "RGBA")
    title_font = load_font(30, bold=True)
    label_font = load_font(19, bold=True)
    text_font = load_font(15)
    small_font = load_font(13)

    draw.text((60, 38), "K-means 水深分区图", font=title_font, fill=(15, 23, 42))

    for i in range(len(y) - 1):
        for j in range(len(x) - 1):
            cell_labels = [labels[i, j], labels[i, j + 1], labels[i + 1, j], labels[i + 1, j + 1]]
            valid = [int(v) for v in cell_labels if v >= 0]
            if not valid:
                continue
            cluster_id = max(set(valid), key=valid.count)
            color = ZONE_COLORS[min(cluster_id, len(ZONE_COLORS) - 1)]
            p1 = to_pixel(float(x[j]), float(y[i]))
            p2 = to_pixel(float(x[j + 1]), float(y[i + 1]))
            draw.rectangle((p1[0], p2[1], p2[0], p1[1]), fill=color + (238,))

    for value in np.linspace(float(x.min()), float(x.max()), 9):
        px, _ = to_pixel(value, float(y.min()))
        draw.line((px, top, px, bottom), fill=(255, 255, 255, 65), width=1)
        draw.text((px - 18, bottom + 10), f"{value:.1f}", font=small_font, fill=(71, 85, 105))

    for value in np.linspace(float(y.min()), float(y.max()), 6):
        _, py = to_pixel(float(x.min()), value)
        draw.line((left, py, right, py), fill=(255, 255, 255, 65), width=1)
        draw.text((left - 52, py - 8), f"{value:.1f}", font=small_font, fill=(71, 85, 105))

    boundaries = [(ranges[i]["max_depth"] + ranges[i + 1]["min_depth"]) / 2 for i in range(len(ranges) - 1)]
    for boundary in boundaries:
        for p1, p2 in marching_segments(x, y, depth, float(boundary)):
            draw.line((*to_pixel(*p1), *to_pixel(*p2)), fill=(21, 35, 54, 210), width=2)

    draw.rectangle((left, top, right, bottom), outline=(16, 44, 61, 255), width=3)
    draw.text((left + plot_w / 2 - 135, bottom + 48), "东西方向 / n mile", font=label_font, fill=(30, 41, 59))
    draw.text((left, top - 34), "南北方向 / n mile", font=label_font, fill=(30, 41, 59))

    info = (
        f"K = {len(ranges)}\n"
        f"有效测深点：{np.isfinite(depth).sum()}\n"
        f"水深范围：{np.nanmin(depth):.2f} - {np.nanmax(depth):.2f} m\n"
        "分界线：相邻聚类阈值"
    )
    panel_x, panel_y, panel_w, panel_h = 1235, 130, 345, 590
    draw.rounded_rectangle((panel_x, panel_y, panel_x + panel_w, panel_y + panel_h), radius=10, fill=(255, 255, 255, 235), outline=(185, 199, 206, 255))
    draw.text((panel_x + 22, panel_y + 22), "结果摘要", font=label_font, fill=(15, 23, 42))
    draw.multiline_text((panel_x + 22, panel_y + 68), info, font=text_font, fill=(16, 44, 61), spacing=8)

    legend_x, legend_y = panel_x + 22, panel_y + 240
    draw.text((legend_x, legend_y - 30), "K-means 水深分区", font=label_font, fill=(30, 41, 59))
    for idx, row in enumerate(ranges):
        y0 = legend_y + idx * 28
        color = ZONE_COLORS[min(idx, len(ZONE_COLORS) - 1)]
        draw.rounded_rectangle((legend_x, y0, legend_x + 24, y0 + 18), radius=3, fill=color + (255,))
        text = (
            f"{row['cluster']}. {row['min_depth']:.1f}-{row['max_depth']:.1f} m "
            f"({row['ratio'] * 100:.1f}%)"
        )
        draw.text((legend_x + 34, y0 - 1), text, font=small_font, fill=(30, 41, 59))

    image.save(CLUSTER_PNG_PATH)


def draw_kmeans_svg(
    x: np.ndarray, y: np.ndarray, depth: np.ndarray, labels: np.ndarray, ranges: list[dict]
) -> None:
    print("[5/6] Drawing K-means depth-zone SVG...", flush=True)
    width, height = 1650, 950
    left, top, plot_w, plot_h = 105, 130, 1080, 650
    right, bottom = left + plot_w, top + plot_h

    def to_pixel(px: float, py: float) -> tuple[float, float]:
        xx = left + (px - float(x.min())) / max(float(x.max() - x.min()), 1e-12) * plot_w
        yy = bottom - (py - float(y.min())) / max(float(y.max() - y.min()), 1e-12) * plot_h
        return xx, yy

    def rgb(color: tuple[int, int, int]) -> str:
        return f"rgb({color[0]},{color[1]},{color[2]})"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style><![CDATA[text{font-family:'Microsoft YaHei','SimHei',Arial,sans-serif}.thin{vector-effect:non-scaling-stroke}.fixed-label{font-size:13px}.title{font-size:30px;font-weight:700}.axis{font-size:19px;font-weight:700}.panel-title{font-size:19px;font-weight:700}.panel-text{font-size:15px}]]></style>",
        f'<rect width="{width}" height="{height}" fill="#f7fafc"/>',
        '<text x="60" y="70" class="title" fill="#0f172a">K-means 水深分区图</text>',
    ]
    for i in range(len(y) - 1):
        for j in range(len(x) - 1):
            cell_labels = [labels[i, j], labels[i, j + 1], labels[i + 1, j], labels[i + 1, j + 1]]
            valid = [int(v) for v in cell_labels if v >= 0]
            if not valid:
                continue
            cluster_id = max(set(valid), key=valid.count)
            color = ZONE_COLORS[min(cluster_id, len(ZONE_COLORS) - 1)]
            p1 = to_pixel(float(x[j]), float(y[i]))
            p2 = to_pixel(float(x[j + 1]), float(y[i + 1]))
            parts.append(f'<rect x="{p1[0]:.2f}" y="{p2[1]:.2f}" width="{p2[0] - p1[0]:.2f}" height="{p1[1] - p2[1]:.2f}" fill="{rgb(color)}" opacity="0.93"/>')

    for value in np.linspace(float(x.min()), float(x.max()), 9):
        px, _ = to_pixel(value, float(y.min()))
        parts.append(f'<line class="thin" x1="{px:.2f}" y1="{top}" x2="{px:.2f}" y2="{bottom}" stroke="#ffffff" stroke-width="1" opacity="0.45"/>')
        parts.append(f'<text x="{px:.2f}" y="{bottom + 24}" class="fixed-label" fill="#475569" text-anchor="middle">{value:.1f}</text>')
    for value in np.linspace(float(y.min()), float(y.max()), 6):
        _, py = to_pixel(float(x.min()), value)
        parts.append(f'<line class="thin" x1="{left}" y1="{py:.2f}" x2="{right}" y2="{py:.2f}" stroke="#ffffff" stroke-width="1" opacity="0.45"/>')
        parts.append(f'<text x="{left - 22}" y="{py + 5:.2f}" class="fixed-label" fill="#475569" text-anchor="middle">{value:.1f}</text>')

    boundaries = [(ranges[i]["max_depth"] + ranges[i + 1]["min_depth"]) / 2 for i in range(len(ranges) - 1)]
    for boundary in boundaries:
        for p1, p2 in marching_segments(x, y, depth, float(boundary)):
            a, b = to_pixel(*p1), to_pixel(*p2)
            parts.append(f'<line class="thin" x1="{a[0]:.2f}" y1="{a[1]:.2f}" x2="{b[0]:.2f}" y2="{b[1]:.2f}" stroke="#152336" stroke-width="1.6" opacity="0.85"/>')

    parts.append(f'<rect class="thin" x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#102c3d" stroke-width="3"/>')
    parts.append(f'<text x="{left + plot_w / 2}" y="{bottom + 54}" class="axis" fill="#1e293b" text-anchor="middle">东西方向 / n mile</text>')
    parts.append(f'<text x="{left}" y="{top - 18}" class="axis" fill="#1e293b">南北方向 / n mile</text>')

    panel_x, panel_y = 1235, 130
    parts.append(f'<rect class="thin" x="{panel_x}" y="{panel_y}" width="345" height="590" rx="10" fill="#ffffff" opacity="0.92" stroke="#b9c7ce"/>')
    parts.append(f'<text x="{panel_x + 22}" y="{panel_y + 47}" class="panel-title" fill="#0f172a">结果摘要</text>')
    for i, item in enumerate([
        f"K = {len(ranges)}",
        f"有效测深点：{np.isfinite(depth).sum()}",
        f"水深范围：{np.nanmin(depth):.2f} - {np.nanmax(depth):.2f} m",
        "分界线：相邻聚类阈值",
    ]):
        parts.append(f'<text x="{panel_x + 22}" y="{panel_y + 84 + i * 26}" class="panel-text" fill="#102c3d">{html.escape(item)}</text>')
    legend_x, legend_y = panel_x + 22, panel_y + 240
    parts.append(f'<text x="{legend_x}" y="{legend_y - 30}" class="panel-title" fill="#1e293b">K-means 水深分区</text>')
    for idx, row in enumerate(ranges):
        y0 = legend_y + idx * 28
        color = ZONE_COLORS[min(idx, len(ZONE_COLORS) - 1)]
        label = f"{row['cluster']}. {row['min_depth']:.1f}-{row['max_depth']:.1f} m ({row['ratio'] * 100:.1f}%)"
        parts.append(f'<rect class="thin" x="{legend_x}" y="{y0}" width="24" height="18" rx="3" fill="{rgb(color)}" stroke="#64748b" stroke-width="1"/>')
        parts.append(f'<text x="{legend_x + 34}" y="{y0 + 14}" class="fixed-label" fill="#1e293b">{html.escape(label)}</text>')
    parts.append("</svg>")
    CLUSTER_SVG_PATH.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    print("=== 2023B attachment K-means depth classification ===", flush=True)
    print("[1/6] Reading bathymetric grid...", flush=True)
    x, y, depth = read_bathymetry_grid()
    _, selected_k = elbow_analysis(depth)
    print(f"[4/6] Running final K-means on depth values, K={selected_k}...", flush=True)
    labels, centers, iterations = kmeans_1d(depth, selected_k)
    ranges = cluster_ranges(depth, labels, centers)
    export_cluster_points(x, y, depth, labels, ranges)
    draw_kmeans_map(x, y, depth, labels, ranges)
    draw_kmeans_svg(x, y, depth, labels, ranges)

    print("[6/6] Done.", flush=True)
    print("selected_k:", selected_k)
    print("iterations:", iterations)
    for row in ranges:
        print(
            f"cluster_{row['cluster']}: center={row['center']:.4f}, "
            f"range={row['min_depth']:.2f}-{row['max_depth']:.2f} m, "
            f"ratio={row['ratio'] * 100:.2f}%"
        )
    print("source_contour_png:", CONTOUR_PNG_PATH)
    print("saved_elbow_csv:", ELBOW_CSV_PATH)
    print("saved_csv:", CLUSTER_CSV_PATH)
    print("saved_png:", CLUSTER_PNG_PATH)
    print("saved_svg:", CLUSTER_SVG_PATH)


if __name__ == "__main__":
    main()
