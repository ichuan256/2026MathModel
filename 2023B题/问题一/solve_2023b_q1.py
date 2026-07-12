from math import cos, pi, sin, tan
from pathlib import Path

from openpyxl import load_workbook
from PIL import Image, ImageDraw, ImageFont


THETA = 120 * pi / 180
GAMMA = THETA / 2
ALPHA = 1.5 * pi / 180
CENTER_DEPTH = 70.0
LINE_SPACING = 200.0
DISTANCES = [-800, -600, -400, -200, 0, 200, 400, 600, 800]

BASE_DIR = Path(__file__).resolve().parent
PROBLEM_DIR = BASE_DIR
WORKBOOK_PATH = PROBLEM_DIR / "result1.xlsx"
PNG_PATH = PROBLEM_DIR / "q1_multibeam_profile.png"
SVG_PATH = PROBLEM_DIR / "q1_multibeam_profile.svg"


def depth_at(distance: float) -> float:
    return CENTER_DEPTH - distance * tan(ALPHA)


def side_widths(depth: float) -> tuple[float, float]:
    common = depth * sin(GAMMA) * cos(ALPHA)
    left_width = common / cos(GAMMA + ALPHA)
    right_width = common / cos(GAMMA - ALPHA)
    return left_width, right_width


def total_width(depth: float) -> float:
    left_width, right_width = side_widths(depth)
    return left_width + right_width


def round2(value: float) -> float:
    return round(value + 1e-12, 2)


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


def solve() -> tuple[list[float], list[float], list[float | str]]:
    depths = [depth_at(distance) for distance in DISTANCES]
    widths = [total_width(depth) for depth in depths]
    side_pairs = [side_widths(depth) for depth in depths]

    overlaps: list[float | str] = ["\u2014\u2014"]
    for index in range(1, len(DISTANCES)):
        previous_right = side_pairs[index - 1][1]
        current_left = side_pairs[index][0]
        overlap_length = previous_right + current_left - LINE_SPACING
        overlap_rate = overlap_length / widths[index] * 100
        overlaps.append(round2(overlap_rate))

    return [round2(x) for x in depths], [round2(x) for x in widths], overlaps


def draw_centered(
    draw: ImageDraw.ImageDraw,
    point: tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    draw.text((point[0] - (bbox[2] - bbox[0]) / 2, point[1] - (bbox[3] - bbox[1]) / 2), text, font=font, fill=fill)


def map_point(distance: float, value: float, value_min: float, value_max: float) -> tuple[float, float]:
    left, top, plot_w, plot_h = 95, 130, 760, 390
    x = left + (distance - min(DISTANCES)) / (max(DISTANCES) - min(DISTANCES)) * plot_w
    y = top + (value_max - value) / max(value_max - value_min, 1e-12) * plot_h
    return x, y


def draw_png(depths: list[float], widths: list[float], overlaps: list[float | str]) -> None:
    image = Image.new("RGB", (1120, 690), (248, 250, 252))
    draw = ImageDraw.Draw(image, "RGBA")
    title_font = load_font(28, True)
    label_font = load_font(17, True)
    text_font = load_font(15)
    small_font = load_font(13)

    left, top, plot_w, plot_h = 95, 130, 760, 390
    right, bottom = left + plot_w, top + plot_h
    value_min = min(depths + widths) * 0.90
    value_max = max(depths + widths) * 1.06

    draw.text((54, 36), "问题一：多波束覆盖宽度与重叠率计算结果", font=title_font, fill=(15, 23, 42))
    for i in range(6):
        y = top + i / 5 * plot_h
        value = value_max - i / 5 * (value_max - value_min)
        draw.line((left, y, right, y), fill=(203, 213, 225, 180), width=1)
        draw.text((30, y - 8), f"{value:.0f}", font=small_font, fill=(71, 85, 105))
    for distance in DISTANCES:
        x, _ = map_point(distance, value_min, value_min, value_max)
        draw.line((x, top, x, bottom), fill=(226, 232, 240, 150), width=1)
        draw_centered(draw, (x, bottom + 18), str(distance), small_font, (71, 85, 105))

    depth_points = [map_point(d, z, value_min, value_max) for d, z in zip(DISTANCES, depths)]
    width_points = [map_point(d, w, value_min, value_max) for d, w in zip(DISTANCES, widths)]
    draw.line(depth_points, fill=(37, 99, 235, 255), width=3)
    draw.line(width_points, fill=(15, 118, 110, 255), width=3)
    for point, value in zip(depth_points, depths):
        draw.ellipse((point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4), fill=(37, 99, 235, 255))
        draw_centered(draw, (point[0], point[1] - 18), f"{value:.1f}", small_font, (30, 64, 175))
    for point, value in zip(width_points, widths):
        draw.ellipse((point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4), fill=(15, 118, 110, 255))
        draw_centered(draw, (point[0], point[1] + 18), f"{value:.1f}", small_font, (17, 94, 89))

    draw.rectangle((left, top, right, bottom), outline=(15, 23, 42, 255), width=2)
    draw_centered(draw, (left + plot_w / 2, bottom + 54), "测线距中心距离 / m", label_font, (30, 41, 59))
    draw.text((left, top - 28), "水深与覆盖宽度 / m", font=label_font, fill=(30, 41, 59))

    panel_x, panel_y = 890, 130
    draw.rounded_rectangle((panel_x, panel_y, panel_x + 190, panel_y + 260), radius=10, fill=(255, 255, 255, 235), outline=(185, 199, 206, 255))
    draw.text((panel_x + 20, panel_y + 22), "结果摘要", font=label_font, fill=(15, 23, 42))
    summary = [
        f"中心水深：{CENTER_DEPTH:.1f} m",
        f"测线间距：{LINE_SPACING:.0f} m",
        f"开角：120°",
        f"坡度：1.5°",
        f"最大覆盖宽度：{max(widths):.2f} m",
    ]
    for i, item in enumerate(summary):
        draw.text((panel_x + 20, panel_y + 66 + i * 26), item, font=text_font, fill=(16, 44, 61))
    draw.line((panel_x + 22, panel_y + 210, panel_x + 62, panel_y + 210), fill=(37, 99, 235, 255), width=3)
    draw.text((panel_x + 72, panel_y + 200), "水深", font=text_font, fill=(30, 41, 59))
    draw.line((panel_x + 22, panel_y + 238, panel_x + 62, panel_y + 238), fill=(15, 118, 110, 255), width=3)
    draw.text((panel_x + 72, panel_y + 228), "覆盖宽度", font=text_font, fill=(30, 41, 59))

    image.save(PNG_PATH)


def draw_svg(depths: list[float], widths: list[float], overlaps: list[float | str]) -> None:
    left, top, plot_w, plot_h = 95, 130, 760, 390
    right, bottom = left + plot_w, top + plot_h
    value_min = min(depths + widths) * 0.90
    value_max = max(depths + widths) * 1.06

    def pt(distance: float, value: float) -> tuple[float, float]:
        return map_point(distance, value, value_min, value_max)

    def points(values: list[float]) -> str:
        return " ".join(f"{pt(d, v)[0]:.2f},{pt(d, v)[1]:.2f}" for d, v in zip(DISTANCES, values))

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="690" viewBox="0 0 1120 690">',
        "<style><![CDATA[text{font-family:'Microsoft YaHei','SimHei',Arial,sans-serif}.thin{vector-effect:non-scaling-stroke}.fixed-label{font-size:13px}.title{font-size:28px;font-weight:700}.axis{font-size:17px;font-weight:700}.panel{font-size:15px}]]></style>",
        '<rect width="1120" height="690" fill="#f8fafc"/>',
        '<text x="54" y="66" class="title" fill="#0f172a">问题一：多波束覆盖宽度与重叠率计算结果</text>',
    ]
    for i in range(6):
        y = top + i / 5 * plot_h
        value = value_max - i / 5 * (value_max - value_min)
        parts.append(f'<line class="thin" x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" stroke="#cbd5e1" stroke-width="1" opacity="0.75"/>')
        parts.append(f'<text x="30" y="{y + 5:.2f}" class="fixed-label" fill="#475569">{value:.0f}</text>')
    for distance in DISTANCES:
        x, _ = pt(distance, value_min)
        parts.append(f'<line class="thin" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{bottom}" stroke="#e2e8f0" stroke-width="1" opacity="0.7"/>')
        parts.append(f'<text x="{x:.2f}" y="{bottom + 22}" class="fixed-label" fill="#475569" text-anchor="middle">{distance}</text>')
    parts.append(f'<polyline class="thin" points="{points(depths)}" fill="none" stroke="#2563eb" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>')
    parts.append(f'<polyline class="thin" points="{points(widths)}" fill="none" stroke="#0f766e" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>')
    for distance, value in zip(DISTANCES, depths):
        x, y = pt(distance, value)
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="#2563eb"/>')
        parts.append(f'<text x="{x:.2f}" y="{y - 14:.2f}" class="fixed-label" fill="#1e40af" text-anchor="middle">{value:.1f}</text>')
    for distance, value in zip(DISTANCES, widths):
        x, y = pt(distance, value)
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="#0f766e"/>')
        parts.append(f'<text x="{x:.2f}" y="{y + 22:.2f}" class="fixed-label" fill="#115e59" text-anchor="middle">{value:.1f}</text>')
    parts.append(f'<rect class="thin" x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#0f172a" stroke-width="2"/>')
    parts.append(f'<text x="{left + plot_w / 2}" y="{bottom + 58}" class="axis" fill="#1e293b" text-anchor="middle">测线距中心距离 / m</text>')
    parts.append(f'<text x="{left}" y="{top - 16}" class="axis" fill="#1e293b">水深与覆盖宽度 / m</text>')
    panel_x, panel_y = 890, 130
    parts.append(f'<rect class="thin" x="{panel_x}" y="{panel_y}" width="190" height="260" rx="10" fill="#ffffff" opacity="0.92" stroke="#b9c7ce"/>')
    parts.append(f'<text x="{panel_x + 20}" y="{panel_y + 46}" class="axis" fill="#0f172a">结果摘要</text>')
    for i, item in enumerate([f"中心水深：{CENTER_DEPTH:.1f} m", f"测线间距：{LINE_SPACING:.0f} m", "开角：120°", "坡度：1.5°", f"最大覆盖宽度：{max(widths):.2f} m"]):
        parts.append(f'<text x="{panel_x + 20}" y="{panel_y + 82 + i * 26}" class="panel" fill="#102c3d">{item}</text>')
    parts.extend([
        f'<line class="thin" x1="{panel_x + 22}" y1="{panel_y + 210}" x2="{panel_x + 62}" y2="{panel_y + 210}" stroke="#2563eb" stroke-width="3"/>',
        f'<text x="{panel_x + 72}" y="{panel_y + 215}" class="panel" fill="#1e293b">水深</text>',
        f'<line class="thin" x1="{panel_x + 22}" y1="{panel_y + 238}" x2="{panel_x + 62}" y2="{panel_y + 238}" stroke="#0f766e" stroke-width="3"/>',
        f'<text x="{panel_x + 72}" y="{panel_y + 243}" class="panel" fill="#1e293b">覆盖宽度</text>',
        "</svg>",
    ])
    SVG_PATH.write_text("\n".join(parts), encoding="utf-8")


def write_result() -> None:
    depths, widths, overlaps = solve()

    workbook = load_workbook(WORKBOOK_PATH)
    sheet = workbook.active

    for col, value in enumerate(depths, start=2):
        cell = sheet.cell(row=2, column=col)
        cell.value = value
        cell.number_format = "0.00"

    for col, value in enumerate(widths, start=2):
        cell = sheet.cell(row=3, column=col)
        cell.value = value
        cell.number_format = "0.00"

    for col, value in enumerate(overlaps, start=2):
        cell = sheet.cell(row=4, column=col)
        cell.value = value
        if isinstance(value, float):
            cell.number_format = "0.00"

    workbook.save(WORKBOOK_PATH)
    draw_png(depths, widths, overlaps)
    draw_svg(depths, widths, overlaps)

    print("workbook:", WORKBOOK_PATH)
    print("png:", PNG_PATH)
    print("svg:", SVG_PATH)
    print("depths:", depths)
    print("widths:", widths)
    print("overlap_rates_percent:", overlaps)


if __name__ == "__main__":
    write_result()
