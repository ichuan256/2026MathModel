import csv
import html
import math
from pathlib import Path

import numpy as np
import openpyxl


try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from PIL import Image, ImageDraw, ImageFont

    HAS_PIL = True
except ImportError:
    HAS_PIL = False


BASE_DIR = Path(__file__).resolve().parent
WORKBOOK_PATH = BASE_DIR / "\u9644\u4ef6.xlsx"
POINTS_CSV_PATH = BASE_DIR / "attachment_bathymetry_points.csv"
CONTOUR_PNG_PATH = BASE_DIR / "attachment_contour_map.png"
CONTOUR_SVG_PATH = BASE_DIR / "attachment_contour_map.svg"


def is_number(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def find_x_axis(rows: list[list]) -> tuple[int, list[int], list[float]]:
    """Find the row containing east-west coordinates."""
    best = None
    for row_index, row in enumerate(rows[:20]):
        numeric_cols = [(col_index, float(value)) for col_index, value in enumerate(row) if is_number(value)]
        if len(numeric_cols) < 10:
            continue
        values = [value for _, value in numeric_cols]
        value_range = max(values) - min(values)
        if value_range > 20:
            continue
        increasing_pairs = sum(b > a for a, b in zip(values, values[1:]))
        if increasing_pairs >= len(values) * 0.8:
            diffs = [b - a for a, b in zip(values, values[1:]) if b > a]
            mean_step = sum(diffs) / max(len(diffs), 1)
            step_error = sum(abs(d - mean_step) for d in diffs) / max(len(diffs), 1)
            score = (len(numeric_cols), -step_error, -value_range)
            if best is None or score > best[0]:
                best = (score, row_index, [col for col, _ in numeric_cols], values)

    if best is None:
        raise ValueError("Could not identify the east-west coordinate row.")
    _, row_index, x_cols, values = best
    return row_index, x_cols, values


def find_y_value(row: list, first_x_col: int) -> float | None:
    """Find the south-north coordinate before the depth matrix begins."""
    for value in row[:first_x_col]:
        if is_number(value):
            return float(value)
    return None


def read_bathymetry_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    print("[1/4] Reading workbook and detecting the bathymetry grid...", flush=True)
    if not WORKBOOK_PATH.exists():
        raise FileNotFoundError(f"Workbook not found: {WORKBOOK_PATH}")

    workbook = openpyxl.load_workbook(WORKBOOK_PATH, data_only=True, read_only=True)
    sheet = workbook[workbook.sheetnames[0]]
    rows = [list(row) for row in sheet.iter_rows(values_only=True)]

    x_row_index, x_cols, x_values = find_x_axis(rows)
    first_x_col = min(x_cols)

    y_values = []
    depth_rows = []
    for row in rows[x_row_index + 1 :]:
        y = find_y_value(row, first_x_col)
        if y is None:
            continue

        depths = []
        valid_count = 0
        for col in x_cols:
            value = row[col] if col < len(row) else None
            if is_number(value):
                depths.append(float(value))
                valid_count += 1
            else:
                depths.append(np.nan)

        if valid_count >= max(5, len(x_cols) * 0.5):
            y_values.append(y)
            depth_rows.append(depths)

    if not y_values or not depth_rows:
        raise ValueError("Could not identify the south-north coordinates and depth matrix.")

    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    depth = np.asarray(depth_rows, dtype=float)

    print(
        f"    x nodes: {len(x)}, y nodes: {len(y)}, valid depth points: {np.isfinite(depth).sum()}",
        flush=True,
    )
    return x, y, depth


def export_points(x: np.ndarray, y: np.ndarray, depth: np.ndarray) -> None:
    print("[2/4] Exporting discrete bathymetry points...", flush=True)
    with POINTS_CSV_PATH.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["x_nm", "y_nm", "depth_m"])
        for row_index, y_value in enumerate(y):
            for col_index, x_value in enumerate(x):
                z = depth[row_index, col_index]
                if np.isfinite(z):
                    writer.writerow([round(float(x_value), 6), round(float(y_value), 6), round(float(z), 4)])


def nice_levels(depth: np.ndarray, interval: float = 10.0) -> np.ndarray:
    z_min = float(np.nanmin(depth))
    z_max = float(np.nanmax(depth))
    start = math.floor(z_min / interval) * interval
    stop = math.ceil(z_max / interval) * interval
    return np.arange(start, stop + interval, interval)


def plot_with_matplotlib(x: np.ndarray, y: np.ndarray, depth: np.ndarray) -> None:
    x_grid, y_grid = np.meshgrid(x, y)
    levels = nice_levels(depth)

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(14, 8), constrained_layout=False)
    fig.subplots_adjust(right=0.76)
    filled = ax.contourf(x_grid, y_grid, depth, levels=levels, cmap="YlGnBu", extend="both")
    contour = ax.contour(x_grid, y_grid, depth, levels=levels, colors="#18384a", linewidths=0.65, alpha=0.82)
    ax.clabel(contour, inline=True, fmt="%.0f m", fontsize=8)
    ax.scatter(x_grid.ravel(), y_grid.ravel(), s=2, c="white", alpha=0.18, linewidths=0)

    colorbar = fig.colorbar(filled, ax=ax, pad=0.02, shrink=0.92)
    colorbar.set_label("海水深度 / m")

    ax.set_title("离散测深点生成等深线图", fontsize=16, pad=14)
    ax.set_xlabel("东西方向 / n mile")
    ax.set_ylabel("南北方向 / n mile")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="white", linewidth=0.6, alpha=0.45)

    info = (
        "结果摘要\n"
        f"数据来源：附件.xlsx\n"
        f"有效测深点：{np.isfinite(depth).sum()}\n"
        f"水深范围：{np.nanmin(depth):.2f} - {np.nanmax(depth):.2f} m\n"
        "等深线间隔：10 m"
    )
    fig.text(
        0.79,
        0.76,
        info,
        ha="left",
        va="top",
        fontsize=10,
        color="#102c3d",
        bbox={"boxstyle": "round,pad=0.45", "fc": "white", "ec": "#b9c7ce", "alpha": 0.9},
    )

    fig.savefig(CONTOUR_PNG_PATH, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/simhei.ttf"),
    ]
    for font_path in candidates:
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def blend(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    t = min(max(float(t), 0.0), 1.0)
    return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))


def depth_color(z: float, z_min: float, z_max: float) -> tuple[int, int, int]:
    t = (z - z_min) / max(z_max - z_min, 1e-12)
    if t < 0.5:
        return blend((225, 245, 238), (88, 176, 184), t * 2)
    return blend((88, 176, 184), (22, 72, 110), (t - 0.5) * 2)


def rgb(color: tuple[int, int, int]) -> str:
    return f"rgb({color[0]},{color[1]},{color[2]})"


def marching_segments(
    x: np.ndarray, y: np.ndarray, depth: np.ndarray, level: float
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    segments = []

    def interp(p1, z1, p2, z2):
        if abs(z2 - z1) < 1e-12:
            return p1
        t = (level - z1) / (z2 - z1)
        return (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))

    for i in range(len(y) - 1):
        for j in range(len(x) - 1):
            corners = [
                ((x[j], y[i]), depth[i, j]),
                ((x[j + 1], y[i]), depth[i, j + 1]),
                ((x[j + 1], y[i + 1]), depth[i + 1, j + 1]),
                ((x[j], y[i + 1]), depth[i + 1, j]),
            ]
            if any(not np.isfinite(z) for _, z in corners):
                continue

            crossings = []
            for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
                p1, z1 = corners[a]
                p2, z2 = corners[b]
                if (z1 - level) * (z2 - level) < 0:
                    crossings.append(interp(p1, z1, p2, z2))

            if len(crossings) == 2:
                segments.append((crossings[0], crossings[1]))
            elif len(crossings) == 4:
                segments.append((crossings[0], crossings[1]))
                segments.append((crossings[2], crossings[3]))
    return segments


def plot_with_pillow(x: np.ndarray, y: np.ndarray, depth: np.ndarray) -> None:
    if not HAS_PIL:
        raise ImportError("Neither matplotlib nor Pillow is available for plotting.")

    width, height = 1650, 950
    left, top, plot_w, plot_h = 105, 130, 1080, 650
    right, bottom = left + plot_w, top + plot_h
    z_min, z_max = float(np.nanmin(depth)), float(np.nanmax(depth))
    levels = nice_levels(depth)

    def to_pixel(px: float, py: float) -> tuple[float, float]:
        xx = left + (px - float(x.min())) / max(float(x.max() - x.min()), 1e-12) * plot_w
        yy = bottom - (py - float(y.min())) / max(float(y.max() - y.min()), 1e-12) * plot_h
        return xx, yy

    image = Image.new("RGB", (width, height), (247, 250, 252))
    draw = ImageDraw.Draw(image, "RGBA")
    title_font = load_font(30, bold=True)
    label_font = load_font(19, bold=True)
    text_font = load_font(16)
    small_font = load_font(13)

    draw.text((60, 38), "离散测深点生成等深线图", font=title_font, fill=(15, 23, 42))

    for i in range(len(y) - 1):
        for j in range(len(x) - 1):
            vals = [depth[i, j], depth[i, j + 1], depth[i + 1, j], depth[i + 1, j + 1]]
            if any(not np.isfinite(v) for v in vals):
                continue
            color = depth_color(float(np.mean(vals)), z_min, z_max)
            p1 = to_pixel(float(x[j]), float(y[i]))
            p2 = to_pixel(float(x[j + 1]), float(y[i + 1]))
            draw.rectangle((p1[0], p2[1], p2[0], p1[1]), fill=color + (235,))

    for value in np.linspace(float(x.min()), float(x.max()), 9):
        px, _ = to_pixel(value, float(y.min()))
        draw.line((px, top, px, bottom), fill=(255, 255, 255, 70), width=1)
        draw.text((px - 18, bottom + 10), f"{value:.1f}", font=small_font, fill=(71, 85, 105))

    for value in np.linspace(float(y.min()), float(y.max()), 6):
        _, py = to_pixel(float(x.min()), value)
        draw.line((left, py, right, py), fill=(255, 255, 255, 70), width=1)
        draw.text((left - 52, py - 8), f"{value:.1f}", font=small_font, fill=(71, 85, 105))

    for level in levels:
        for p1, p2 in marching_segments(x, y, depth, float(level)):
            draw.line((*to_pixel(*p1), *to_pixel(*p2)), fill=(24, 56, 74, 220), width=2)

    label_counter = 0
    for level in levels[::2]:
        segments = marching_segments(x, y, depth, float(level))
        for p1, p2 in segments[:: max(1, len(segments) // 6 or 1)]:
            if label_counter % 3 == 0:
                mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
                px, py = to_pixel(mx, my)
                text = f"{level:.0f}"
                box = draw.textbbox((0, 0), text, font=small_font)
                tw, th = box[2] - box[0], box[3] - box[1]
                draw.rounded_rectangle((px - tw / 2 - 4, py - th / 2 - 3, px + tw / 2 + 4, py + th / 2 + 3), 4, fill=(255, 255, 255, 210))
                draw.text((px - tw / 2, py - th / 2 - 1), text, font=small_font, fill=(24, 56, 74))
            label_counter += 1

    draw.rectangle((left, top, right, bottom), outline=(16, 44, 61, 255), width=3)
    draw.text((left + plot_w / 2 - 135, bottom + 48), "东西方向 / n mile", font=label_font, fill=(30, 41, 59))
    draw.text((left, top - 34), "南北方向 / n mile", font=label_font, fill=(30, 41, 59))

    info = (
        f"数据来源：附件.xlsx\n"
        f"有效测深点：{np.isfinite(depth).sum()}\n"
        f"水深范围：{z_min:.2f} - {z_max:.2f} m\n"
        "等深线间隔：10 m"
    )
    panel_x, panel_y, panel_w, panel_h = 1235, 130, 345, 430
    draw.rounded_rectangle((panel_x, panel_y, panel_x + panel_w, panel_y + panel_h), radius=10, fill=(255, 255, 255, 235), outline=(185, 199, 206, 255))
    draw.text((panel_x + 22, panel_y + 22), "结果摘要", font=label_font, fill=(15, 23, 42))
    draw.multiline_text((panel_x + 22, panel_y + 68), info, font=text_font, fill=(16, 44, 61), spacing=8)

    legend_x, legend_y = panel_x + 22, panel_y + 278
    draw.text((legend_x, legend_y - 28), "海水深度 / m", font=small_font, fill=(71, 85, 105))
    for i in range(220):
        z = z_min + (z_max - z_min) * i / 219
        draw.line((legend_x + i, legend_y, legend_x + i, legend_y + 18), fill=depth_color(z, z_min, z_max) + (255,))
    draw.text((legend_x, legend_y + 24), f"{z_min:.1f}", font=small_font, fill=(71, 85, 105))
    draw.text((legend_x + 170, legend_y + 24), f"{z_max:.1f}", font=small_font, fill=(71, 85, 105))

    image.save(CONTOUR_PNG_PATH)


def plot_svg(x: np.ndarray, y: np.ndarray, depth: np.ndarray) -> None:
    width, height = 1650, 950
    left, top, plot_w, plot_h = 105, 130, 1080, 650
    right, bottom = left + plot_w, top + plot_h
    z_min, z_max = float(np.nanmin(depth)), float(np.nanmax(depth))
    levels = nice_levels(depth)

    def to_pixel(px: float, py: float) -> tuple[float, float]:
        xx = left + (px - float(x.min())) / max(float(x.max() - x.min()), 1e-12) * plot_w
        yy = bottom - (py - float(y.min())) / max(float(y.max() - y.min()), 1e-12) * plot_h
        return xx, yy

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style><![CDATA[text{font-family:'Microsoft YaHei','SimHei',Arial,sans-serif}.thin{vector-effect:non-scaling-stroke}.fixed-label{font-size:13px}.title{font-size:30px;font-weight:700}.axis{font-size:19px;font-weight:700}.panel-title{font-size:19px;font-weight:700}.panel-text{font-size:16px}]]></style>",
        f'<rect width="{width}" height="{height}" fill="#f7fafc"/>',
        '<text x="60" y="70" class="title" fill="#0f172a">离散测深点生成等深线图</text>',
    ]
    for i in range(len(y) - 1):
        for j in range(len(x) - 1):
            vals = [depth[i, j], depth[i, j + 1], depth[i + 1, j], depth[i + 1, j + 1]]
            if any(not np.isfinite(v) for v in vals):
                continue
            p1 = to_pixel(float(x[j]), float(y[i]))
            p2 = to_pixel(float(x[j + 1]), float(y[i + 1]))
            parts.append(f'<rect x="{p1[0]:.2f}" y="{p2[1]:.2f}" width="{p2[0] - p1[0]:.2f}" height="{p1[1] - p2[1]:.2f}" fill="{rgb(depth_color(float(np.mean(vals)), z_min, z_max))}" opacity="0.92"/>')

    for level in levels:
        for p1, p2 in marching_segments(x, y, depth, float(level)):
            a, b = to_pixel(*p1), to_pixel(*p2)
            parts.append(f'<line class="thin" x1="{a[0]:.2f}" y1="{a[1]:.2f}" x2="{b[0]:.2f}" y2="{b[1]:.2f}" stroke="#18384a" stroke-width="1.2" opacity="0.82"/>')

    for value in np.linspace(float(x.min()), float(x.max()), 9):
        px, _ = to_pixel(value, float(y.min()))
        parts.append(f'<line class="thin" x1="{px:.2f}" y1="{top}" x2="{px:.2f}" y2="{bottom}" stroke="#ffffff" stroke-width="1" opacity="0.5"/>')
        parts.append(f'<text x="{px:.2f}" y="{bottom + 24}" class="fixed-label" fill="#475569" text-anchor="middle">{value:.1f}</text>')
    for value in np.linspace(float(y.min()), float(y.max()), 6):
        _, py = to_pixel(float(x.min()), value)
        parts.append(f'<line class="thin" x1="{left}" y1="{py:.2f}" x2="{right}" y2="{py:.2f}" stroke="#ffffff" stroke-width="1" opacity="0.5"/>')
        parts.append(f'<text x="{left - 22}" y="{py + 5:.2f}" class="fixed-label" fill="#475569" text-anchor="middle">{value:.1f}</text>')

    parts.append(f'<rect class="thin" x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#102c3d" stroke-width="3"/>')
    parts.append(f'<text x="{left + plot_w / 2}" y="{bottom + 54}" class="axis" fill="#1e293b" text-anchor="middle">东西方向 / n mile</text>')
    parts.append(f'<text x="{left}" y="{top - 18}" class="axis" fill="#1e293b">南北方向 / n mile</text>')

    panel_x, panel_y = 1235, 130
    parts.append(f'<rect class="thin" x="{panel_x}" y="{panel_y}" width="345" height="430" rx="10" fill="#ffffff" opacity="0.92" stroke="#b9c7ce"/>')
    parts.append(f'<text x="{panel_x + 22}" y="{panel_y + 47}" class="panel-title" fill="#0f172a">结果摘要</text>')
    for i, item in enumerate([
        "数据来源：附件.xlsx",
        f"有效测深点：{np.isfinite(depth).sum()}",
        f"水深范围：{z_min:.2f} - {z_max:.2f} m",
        "等深线间隔：10 m",
    ]):
        parts.append(f'<text x="{panel_x + 22}" y="{panel_y + 84 + i * 28}" class="panel-text" fill="#102c3d">{html.escape(item)}</text>')
    legend_x, legend_y = panel_x + 22, panel_y + 250
    parts.append(f'<text x="{legend_x}" y="{legend_y}" class="fixed-label" fill="#475569">海水深度 / m</text>')
    for i in range(220):
        z = z_min + (z_max - z_min) * i / 219
        parts.append(f'<line class="thin" x1="{legend_x + i}" y1="{legend_y + 30}" x2="{legend_x + i}" y2="{legend_y + 50}" stroke="{rgb(depth_color(z, z_min, z_max))}" stroke-width="1"/>')
    parts.append(f'<text x="{legend_x}" y="{legend_y + 76}" class="fixed-label" fill="#475569">{z_min:.1f}</text>')
    parts.append(f'<text x="{legend_x + 170}" y="{legend_y + 76}" class="fixed-label" fill="#475569">{z_max:.1f}</text>')
    parts.append("</svg>")
    CONTOUR_SVG_PATH.write_text("\n".join(parts), encoding="utf-8")


def plot_contour(x: np.ndarray, y: np.ndarray, depth: np.ndarray) -> None:
    print("[3/4] Drawing contour map...", flush=True)
    if HAS_MATPLOTLIB:
        plot_with_matplotlib(x, y, depth)
    else:
        print("    matplotlib not found; using Pillow fallback renderer.", flush=True)
        plot_with_pillow(x, y, depth)
    plot_svg(x, y, depth)


def main() -> None:
    print("=== 2023B attachment bathymetric contour program ===", flush=True)
    x, y, depth = read_bathymetry_grid()
    export_points(x, y, depth)
    plot_contour(x, y, depth)
    print("[4/4] Done.", flush=True)
    print("saved_points:", POINTS_CSV_PATH)
    print("saved_png:", CONTOUR_PNG_PATH)
    print("saved_svg:", CONTOUR_SVG_PATH)


if __name__ == "__main__":
    main()
