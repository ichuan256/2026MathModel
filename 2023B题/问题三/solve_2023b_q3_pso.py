import csv
import html
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


NM = 1852.0
THETA = math.radians(120)
GAMMA = THETA / 2
ALPHA = math.radians(1.5)
CENTER_DEPTH = 110.0

X_MIN, X_MAX = -2 * NM, 2 * NM
Y_MIN, Y_MAX = -1 * NM, 1 * NM

PHI_BOUNDS = (80.0, 100.0)
PHI_SPAN_BOUNDS = (-3.0, 3.0)
ETA_BOUNDS = (0.10, 0.20)

SWARM_SIZE = 50
MAX_ITER = 3000
MIN_ITER = 500
CONVERGENCE_TOL = 0.0000001
CONVERGENCE_PATIENCE = 300
SEED = 2023

BASE_DIR = Path(__file__).resolve().parent
PROBLEM_DIR = BASE_DIR
OUTPUT_PATH = PROBLEM_DIR / "q3_pso_lines.csv"
PNG_PATH = PROBLEM_DIR / "q3_pso_layout.png"
SVG_PATH = PROBLEM_DIR / "q3_pso_layout.svg"
SAVE_CSV = False


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
    u = (math.cos(phi), math.sin(phi))
    n = (-math.sin(phi), math.cos(phi))
    return u, n


def projection_bounds(n: tuple[float, float]) -> tuple[float, float]:
    values = [x * n[0] + y * n[1] for x, y in rectangle_vertices()]
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
    u: tuple[float, float],
    n: tuple[float, float],
) -> tuple[float, float, float, float, float] | None:
    polygon = covered_polygon(left, right, n)
    if not polygon:
        return None
    t_values = [x * u[0] + y * u[1] for x, y in polygon]
    t_low, t_high = min(t_values), max(t_values)
    x1 = r * n[0] + t_low * u[0]
    y1 = r * n[1] + t_low * u[1]
    x2 = r * n[0] + t_high * u[0]
    y2 = r * n[1] + t_high * u[1]
    return x1, y1, x2, y2, t_high - t_low


def line_rectangle_segment(
    r: float, u: tuple[float, float], n: tuple[float, float]
) -> tuple[float, float, float, float] | None:
    # Points on the line satisfy p = r*n + t*u. Clip the t interval to the rectangle.
    t_low = -1e30
    t_high = 1e30

    for coord, lo, hi in ((0, X_MIN, X_MAX), (1, Y_MIN, Y_MAX)):
        base = r * n[coord]
        direction = u[coord]
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
    x1 = r * n[0] + t_low * u[0]
    y1 = r * n[1] + t_low * u[1]
    x2 = r * n[0] + t_high * u[0]
    y2 = r * n[1] + t_high * u[1]
    return x1, y1, x2, y2


def min_depth_on_line(r: float, u: tuple[float, float], n: tuple[float, float]) -> float:
    segment = line_rectangle_segment(r, u, n)
    if segment is None:
        # Outside the rectangle; use the depth of the projected point for numerical bracketing.
        return max(1.0, depth_at_x(r * n[0]))
    x1, _, x2, _ = segment
    return min(depth_at_x(x1), depth_at_x(x2))


def interval_at(
    r: float, phi_deg: float, u: tuple[float, float], n: tuple[float, float]
) -> tuple[float, float, float, float, float]:
    alpha_perp = math.atan(math.tan(ALPHA) * abs(n[0]))
    depth = min_depth_on_line(r, u, n)
    common = depth * math.sin(GAMMA) * math.cos(alpha_perp)
    deep_half = common / math.cos(GAMMA + alpha_perp)
    shallow_half = common / math.cos(GAMMA - alpha_perp)
    left = r - shallow_half
    right = r + deep_half
    return left, right, shallow_half + deep_half, depth, alpha_perp


def bisect(func, lo: float, hi: float, iterations: int = 80) -> float:
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
            return bisect(func, lo, hi)
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


def line_phi_at(base_phi_deg: float, phi_span_deg: float, progress: float) -> float:
    return base_phi_deg + phi_span_deg * (progress - 0.5)


def build_lines(base_phi_deg: float, phi_span_deg: float, eta_target: float) -> list[dict]:
    u, n = unit_vectors(base_phi_deg)
    r_min, r_max = projection_bounds(n)
    pad = 2200.0

    def first_objective(r: float) -> float:
        _, right, _, _, _ = interval_at(r, base_phi_deg, u, n)
        return right - r_max

    try:
        first_r = bracketed_root(
            first_objective,
            r_max - pad,
            r_max + pad,
            r_max - 5 * pad,
            r_max + 5 * pad,
        )
    except ValueError:
        return []

    lines = []
    r = first_r
    for _ in range(300):
        progress = (r - r_min) / max(r_max - r_min, 1e-12)
        line_phi_deg = line_phi_at(base_phi_deg, phi_span_deg, progress)
        line_u, line_n = unit_vectors(line_phi_deg)
        left, right, width, depth, alpha_perp = interval_at(r, base_phi_deg, u, n)
        segment = required_survey_segment(r, left, right, line_u, n)
        if segment is None:
            return []
        x1, y1, x2, y2, length = segment
        lines.append(
            {
                "r": r,
                "line_phi_deg": line_phi_deg,
                "left": left,
                "right": right,
                "width": width,
                "depth": depth,
                "alpha_perp_deg": math.degrees(alpha_perp),
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
            _, cand_right, cand_width, _, _ = interval_at(candidate_r, base_phi_deg, u, n)
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


def evaluate(position: list[float]) -> tuple[float, list[dict]]:
    phi_deg, phi_span_deg, eta_target = position
    lines = build_lines(phi_deg, phi_span_deg, eta_target)
    if not lines:
        return 1e12, []

    total_length = sum(line["length"] for line in lines)
    penalty = 0.0
    u, n = unit_vectors(phi_deg)
    r_min, r_max = projection_bounds(n)

    miss_deep = max(0.0, r_max - lines[0]["right"])
    miss_shallow = max(0.0, lines[-1]["left"] - r_min)
    penalty += 1e6 * (miss_deep + miss_shallow) / NM

    for a, b in zip(lines, lines[1:]):
        overlap = b["right"] - a["left"]
        rate = overlap / min(a["width"], b["width"])
        if rate < 0.10:
            penalty += 1e5 * (0.10 - rate) ** 2
        if rate > 0.20:
            penalty += 1e5 * (rate - 0.20) ** 2

    # Length is measured in nautical miles to keep the scale near the penalties.
    fitness = total_length / NM + penalty
    return fitness, lines


def clamp(value: float, bounds: tuple[float, float]) -> float:
    return min(max(value, bounds[0]), bounds[1])


def improved_pso() -> tuple[list[float], float, list[dict], int, int]:
    random.seed(SEED)
    bounds = [PHI_BOUNDS, PHI_SPAN_BOUNDS, ETA_BOUNDS]
    particles = []
    global_best_position = None
    global_best_fitness = float("inf")
    global_best_lines = []

    print(
        f"[1/4] 初始化粒子群: 粒子数={SWARM_SIZE}, 最大迭代={MAX_ITER}, "
        f"最少迭代={MIN_ITER}",
        flush=True,
    )
    for _ in range(SWARM_SIZE):
        position = [
            random.uniform(*PHI_BOUNDS),
            random.uniform(*PHI_SPAN_BOUNDS),
            random.uniform(*ETA_BOUNDS),
        ]
        velocity = [random.uniform(-1, 1), random.uniform(-0.4, 0.4), random.uniform(-0.01, 0.01)]
        fitness, lines = evaluate(position)
        particle = {
            "position": position,
            "velocity": velocity,
            "best_position": position[:],
            "best_fitness": fitness,
        }
        particles.append(particle)
        if fitness < global_best_fitness:
            global_best_position = position[:]
            global_best_fitness = fitness
            global_best_lines = lines

    print(
        "[2/4] 初始化完成: "
        f"best_phi={global_best_position[0]:.6f}°, "
        f"best_span={global_best_position[1]:.6f}°, "
        f"best_eta={global_best_position[2]:.6f}, "
        f"best_fitness={global_best_fitness:.9f}, "
        f"lines={len(global_best_lines)}",
        flush=True,
    )

    stagnant_iters = 0
    converged_iters = 0
    iterations_used = MAX_ITER
    for iteration in range(MAX_ITER):
        old_best = global_best_fitness
        w = 0.9 - (0.9 - 0.4) * iteration / max(1, MAX_ITER - 1)
        c1 = 2.5 - 2.0 * iteration / max(1, MAX_ITER - 1)
        c2 = 0.5 + 2.0 * iteration / max(1, MAX_ITER - 1)

        for particle in particles:
            for dim in range(3):
                r1 = random.random()
                r2 = random.random()
                cognitive = c1 * r1 * (particle["best_position"][dim] - particle["position"][dim])
                social = c2 * r2 * (global_best_position[dim] - particle["position"][dim])
                particle["velocity"][dim] = w * particle["velocity"][dim] + cognitive + social

                v_limit = (bounds[dim][1] - bounds[dim][0]) * 0.2
                particle["velocity"][dim] = clamp(particle["velocity"][dim], (-v_limit, v_limit))
                particle["position"][dim] = clamp(
                    particle["position"][dim] + particle["velocity"][dim], bounds[dim]
                )

            fitness, lines = evaluate(particle["position"])
            if fitness < particle["best_fitness"]:
                particle["best_fitness"] = fitness
                particle["best_position"] = particle["position"][:]
            if fitness < global_best_fitness:
                global_best_position = particle["position"][:]
                global_best_fitness = fitness
                global_best_lines = lines

        relative_error = abs(old_best - global_best_fitness) / max(abs(old_best), 1e-12)
        if iteration + 1 <= MIN_ITER:
            converged_iters = 0
        elif relative_error < CONVERGENCE_TOL:
            converged_iters += 1
        else:
            converged_iters = 0

        if abs(old_best - global_best_fitness) < 1e-7:
            stagnant_iters += 1
        else:
            stagnant_iters = 0

        # Stagnation perturbation: keep the elite particle, perturb the worst 20%.
        if stagnant_iters >= 12:
            print(
                f"  第 {iteration + 1} 代: 检测到停滞，扰动最差粒子以跳出局部区域",
                flush=True,
            )
            ranked = sorted(particles, key=lambda p: p["best_fitness"], reverse=True)
            for particle in ranked[: max(1, SWARM_SIZE // 5)]:
                particle["position"] = [
                    clamp(global_best_position[0] + random.gauss(0, 1.0), PHI_BOUNDS),
                    clamp(global_best_position[1] + random.gauss(0, 0.4), PHI_SPAN_BOUNDS),
                    clamp(global_best_position[2] + random.gauss(0, 0.01), ETA_BOUNDS),
                ]
                particle["velocity"] = [
                    random.uniform(-0.5, 0.5),
                    random.uniform(-0.2, 0.2),
                    random.uniform(-0.005, 0.005),
                ]
            stagnant_iters = 0

        if (iteration + 1) == 1 or (iteration + 1) % 100 == 0:
            print(
                f"  迭代进度 {iteration + 1:>4}/{MAX_ITER}: "
                f"best_phi={global_best_position[0]:.6f}°, "
                f"span={global_best_position[1]:.6f}°, "
                f"eta={global_best_position[2]:.6f}, "
                f"fitness={global_best_fitness:.9f}, "
                f"lines={len(global_best_lines)}, "
                f"收敛计数={converged_iters}",
                flush=True,
            )

        if iteration + 1 >= MIN_ITER and converged_iters >= CONVERGENCE_PATIENCE:
            iterations_used = iteration + 1
            print(
                f"[3/4] 达到终止条件: 第 {iterations_used} 代, "
                f"连续收敛计数={converged_iters}",
                flush=True,
            )
            break

    return (
        global_best_position,
        global_best_fitness,
        global_best_lines,
        iterations_used,
        converged_iters,
    )


def save_lines(phi_deg: float, phi_span_deg: float, eta_target: float, lines: list[dict]) -> None:
    fieldnames = [
        "index",
        "phi_deg",
        "phi_span_deg",
        "line_phi_deg",
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
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for index, line in enumerate(lines, start=1):
            writer.writerow(
                {
                    "index": index,
                    "phi_deg": round(phi_deg, 6),
                    "phi_span_deg": round(phi_span_deg, 6),
                    "line_phi_deg": round(line["line_phi_deg"], 6),
                    "eta_target": round(eta_target, 6),
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
    t = min(max(t, 0.0), 1.0)
    return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))


def depth_color(depth: float, min_depth: float, max_depth: float) -> tuple[int, int, int]:
    t = (depth - min_depth) / max(max_depth - min_depth, 1e-12)
    if t < 0.5:
        return blend((220, 244, 242), (82, 170, 184), t * 2)
    return blend((82, 170, 184), (20, 72, 106), (t - 0.5) * 2)


def rgb(color: tuple[int, int, int]) -> str:
    return f"rgb({color[0]},{color[1]},{color[2]})"


def world_to_pixel(x_nm: float, y_nm: float) -> tuple[float, float]:
    plot_left, plot_top = 84, 126
    plot_w, plot_h = 940, 470
    px = plot_left + (x_nm + 2.08) / 4.16 * plot_w
    py = plot_top + (1.08 - y_nm) / 2.16 * plot_h
    return px, py


def clip_half_plane_nm(
    polygon: list[tuple[float, float]], normal: tuple[float, float], limit: float, keep_less: bool
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
            output.append((start[0] + ratio * (end[0] - start[0]), start[1] + ratio * (end[1] - start[1])))
    return output


def stripe_polygon_nm(normal: tuple[float, float], left_m: float, right_m: float) -> list[tuple[float, float]]:
    if right_m <= left_m:
        return []
    polygon = [(-2.0, -1.0), (2.0, -1.0), (2.0, 1.0), (-2.0, 1.0)]
    polygon = clip_half_plane_nm(polygon, normal, right_m / NM, keep_less=True)
    return clip_half_plane_nm(polygon, normal, left_m / NM, keep_less=False)


def draw_centered(
    draw: ImageDraw.ImageDraw, center: tuple[float, float], text: str, font: ImageFont.ImageFont, fill
) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    draw.text((center[0] - (bbox[2] - bbox[0]) / 2, center[1] - (bbox[3] - bbox[1]) / 2 - 1), text, font=font, fill=fill)


def place_label(
    occupied: list[tuple[float, float, float, float]],
    center: tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
    draw: ImageDraw.ImageDraw,
) -> tuple[float, float, float, float] | None:
    bbox = draw.textbbox((0, 0), text, font=font)
    w, h = bbox[2] - bbox[0] + 16, bbox[3] - bbox[1] + 12
    for dx, dy in [(0, 0), (0, -30), (0, 30), (-36, 0), (36, 0), (-36, -28), (36, 28)]:
        box = (center[0] + dx - w / 2, center[1] + dy - h / 2, center[0] + dx + w / 2, center[1] + dy + h / 2)
        if not any(not (box[2] < old[0] or box[0] > old[2] or box[3] < old[1] or box[1] > old[3]) for old in occupied):
            occupied.append(box)
            return box
    return None


def pso_summary(phi_deg: float, phi_span_deg: float, eta_target: float, lines: list[dict]) -> str:
    line_phis = [line["line_phi_deg"] for line in lines]
    total_length_nm = sum(line["length"] for line in lines) / NM
    return (
        f"基准方向角：{phi_deg:.3f}°\n"
        f"单线角度范围：{min(line_phis):.3f}° - {max(line_phis):.3f}°\n"
        f"角度跨度：{phi_span_deg:.3f}°\n"
        f"测线数量：{len(lines)} 条\n"
        f"测线总长：{total_length_nm:.3f} n mile\n"
        f"目标重叠率：{eta_target * 100:.1f}%"
    )


def draw_png(phi_deg: float, phi_span_deg: float, eta_target: float, lines: list[dict]) -> None:
    print("[5/6] 绘制 PSO 布设 PNG 图片...", flush=True)
    image = Image.new("RGB", (1400, 760), (248, 250, 252))
    draw = ImageDraw.Draw(image, "RGBA")
    title_font = load_font(28, bold=True)
    panel_title_font = load_font(20, bold=True)
    label_font = load_font(17, bold=True)
    small_font = load_font(14)
    text_font = load_font(16)
    plot_left, plot_top, plot_w, plot_h = 84, 126, 940, 470
    min_depth, max_depth = depth_at_x(X_MAX), depth_at_x(X_MIN)
    normal = unit_vectors(phi_deg)[1]

    draw.text((56, 32), "2023B问题三：改进粒子群算法测线布设结果", font=title_font, fill=(15, 23, 42))
    for i in range(plot_w):
        x_nm = -2.08 + i / plot_w * 4.16
        draw.line((plot_left + i, plot_top, plot_left + i, plot_top + plot_h), fill=depth_color(depth_at_x(x_nm * NM), min_depth, max_depth) + (225,))
    for tick in [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]:
        px, _ = world_to_pixel(tick, 0)
        draw.line((px, plot_top, px, plot_top + plot_h), fill=(255, 255, 255, 75), width=1)
        draw_centered(draw, (px, plot_top + plot_h + 17), f"{tick:g}", small_font, (71, 85, 105))
    for tick in [-1, -0.5, 0, 0.5, 1]:
        _, py = world_to_pixel(0, tick)
        draw.line((plot_left, py, plot_left + plot_w, py), fill=(255, 255, 255, 75), width=1)
        draw.text((plot_left - 48, py - 8), f"{tick:g}", font=small_font, fill=(71, 85, 105))

    for line in lines:
        polygon = stripe_polygon_nm(normal, line["left"], line["right"])
        if polygon:
            draw.polygon([world_to_pixel(x, y) for x, y in polygon], fill=(189, 230, 232, 78))
    for previous, current in zip(lines, lines[1:]):
        polygon = stripe_polygon_nm(normal, previous["left"], current["right"])
        if polygon:
            draw.polygon([world_to_pixel(x, y) for x, y in polygon], fill=(231, 94, 54, 145))

    occupied = []
    for index, line in enumerate(lines, start=1):
        p1 = world_to_pixel(line["x1"] / NM, line["y1"] / NM)
        p2 = world_to_pixel(line["x2"] / NM, line["y2"] / NM)
        draw.line((*p1, *p2), fill=(255, 255, 255, 245), width=2)
        if index == 1 or index == len(lines) or index % 5 == 0:
            center = world_to_pixel((line["x1"] + line["x2"]) / (2 * NM), (line["y1"] + line["y2"]) / (2 * NM))
            box = place_label(occupied, center, str(index), small_font, draw)
            if box:
                draw.rounded_rectangle(box, radius=4, fill=(255, 255, 255, 225), outline=(148, 163, 184, 210))
                draw_centered(draw, ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), str(index), small_font, (18, 42, 58))

    border = [world_to_pixel(-2, -1), world_to_pixel(2, -1), world_to_pixel(2, 1), world_to_pixel(-2, 1)]
    draw.line(border + [border[0]], fill=(16, 44, 61, 255), width=3)
    draw_centered(draw, (plot_left + plot_w / 2, plot_top + plot_h + 50), "东西方向 / n mile（西深，东浅）", label_font, (30, 41, 59))
    draw.text((plot_left, plot_top - 30), "南北方向 / n mile", font=label_font, fill=(30, 41, 59))

    panel_x, panel_y, panel_w = 1060, 126, 292
    draw.rounded_rectangle((panel_x, panel_y, panel_x + panel_w, panel_y + 500), radius=10, fill=(255, 255, 255, 235), outline=(185, 199, 206, 255))
    draw.text((panel_x + 22, panel_y + 22), "结果摘要", font=panel_title_font, fill=(15, 23, 42))
    draw.multiline_text((panel_x + 22, panel_y + 66), pso_summary(phi_deg, phi_span_deg, eta_target, lines), font=text_font, fill=(16, 44, 61), spacing=8)
    draw.text((panel_x + 22, panel_y + 300), "图例说明", font=panel_title_font, fill=(15, 23, 42))
    for item_index, (color, label) in enumerate([((82, 170, 184), "蓝绿渐变：水深底图"), ((189, 230, 232), "浅青色：测线覆盖区域"), ((231, 94, 54), "红橙色：重叠覆盖区域"), ((255, 255, 255), "白色：实际测线")]):
        y0 = panel_y + 342 + item_index * 34
        draw.rounded_rectangle((panel_x + 24, y0, panel_x + 52, y0 + 18), radius=3, fill=color + (255,), outline=(100, 116, 139, 180))
        draw.text((panel_x + 64, y0 - 1), label, font=text_font, fill=(30, 41, 59))
    image.save(PNG_PATH)


def svg_points(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def draw_svg(phi_deg: float, phi_span_deg: float, eta_target: float, lines: list[dict]) -> None:
    print("[6/6] 绘制 PSO 布设 SVG 矢量图...", flush=True)
    min_depth, max_depth = depth_at_x(X_MAX), depth_at_x(X_MIN)
    normal = unit_vectors(phi_deg)[1]
    plot_left, plot_top, plot_w, plot_h = 84, 126, 940, 470
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="760" viewBox="0 0 1400 760">',
        "<style><![CDATA[text{font-family:'Microsoft YaHei','SimHei',Arial,sans-serif}.thin{vector-effect:non-scaling-stroke}.label{font-size:14px}.panel-text{font-size:16px}.title{font-size:28px;font-weight:700}.panel-title{font-size:20px;font-weight:700}]]></style>",
        '<rect width="1400" height="760" fill="#f8fafc"/>',
        '<text x="56" y="63" class="title" fill="#0f172a">2023B问题三：改进粒子群算法测线布设结果</text>',
    ]
    for i in range(120):
        x_nm = -2.08 + i / 119 * 4.16
        parts.append(f'<rect x="{plot_left + i / 120 * plot_w:.2f}" y="{plot_top}" width="{plot_w / 120 + 0.8:.2f}" height="{plot_h}" fill="{rgb(depth_color(depth_at_x(x_nm * NM), min_depth, max_depth))}" opacity="0.88"/>')
    for line in lines:
        polygon = stripe_polygon_nm(normal, line["left"], line["right"])
        if polygon:
            parts.append(f'<polygon points="{svg_points([world_to_pixel(x, y) for x, y in polygon])}" fill="#bde6e8" opacity="0.38"/>')
    for previous, current in zip(lines, lines[1:]):
        polygon = stripe_polygon_nm(normal, previous["left"], current["right"])
        if polygon:
            parts.append(f'<polygon points="{svg_points([world_to_pixel(x, y) for x, y in polygon])}" fill="#e75e36" opacity="0.62"/>')
    fake_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    small_font = load_font(14)
    occupied = []
    for index, line in enumerate(lines, start=1):
        p1 = world_to_pixel(line["x1"] / NM, line["y1"] / NM)
        p2 = world_to_pixel(line["x2"] / NM, line["y2"] / NM)
        parts.append(f'<line class="thin" x1="{p1[0]:.2f}" y1="{p1[1]:.2f}" x2="{p2[0]:.2f}" y2="{p2[1]:.2f}" stroke="#ffffff" stroke-width="2" stroke-linecap="round"/>')
        if index == 1 or index == len(lines) or index % 5 == 0:
            center = world_to_pixel((line["x1"] + line["x2"]) / (2 * NM), (line["y1"] + line["y2"]) / (2 * NM))
            box = place_label(occupied, center, str(index), small_font, fake_draw)
            if box:
                parts.append(f'<rect class="thin" x="{box[0]:.2f}" y="{box[1]:.2f}" width="{box[2] - box[0]:.2f}" height="{box[3] - box[1]:.2f}" rx="4" fill="#ffffff" opacity="0.88" stroke="#94a3b8" stroke-width="1"/>')
                parts.append(f'<text x="{(box[0] + box[2]) / 2:.2f}" y="{(box[1] + box[3]) / 2 + 5:.2f}" class="label" fill="#122a3a" text-anchor="middle">{index}</text>')
    border = [world_to_pixel(-2, -1), world_to_pixel(2, -1), world_to_pixel(2, 1), world_to_pixel(-2, 1)]
    parts.append(f'<polygon class="thin" points="{svg_points(border)}" fill="none" stroke="#102c3d" stroke-width="3"/>')
    parts.append(f'<text x="{plot_left + plot_w / 2}" y="{plot_top + plot_h + 55}" font-size="17" font-weight="700" fill="#1e293b" text-anchor="middle">东西方向 / n mile（西深，东浅）</text>')
    parts.append(f'<text x="{plot_left}" y="{plot_top - 12}" font-size="17" font-weight="700" fill="#1e293b">南北方向 / n mile</text>')
    panel_x, panel_y = 1060, 126
    parts.append(f'<rect class="thin" x="{panel_x}" y="{panel_y}" width="292" height="500" rx="10" fill="#ffffff" opacity="0.92" stroke="#b9c7ce"/>')
    parts.append(f'<text x="{panel_x + 22}" y="{panel_y + 47}" class="panel-title" fill="#0f172a">结果摘要</text>')
    for row_index, line in enumerate(pso_summary(phi_deg, phi_span_deg, eta_target, lines).splitlines()):
        parts.append(f'<text x="{panel_x + 22}" y="{panel_y + 82 + row_index * 26}" class="panel-text" fill="#102c3d">{html.escape(line)}</text>')
    parts.append(f'<text x="{panel_x + 22}" y="{panel_y + 325}" class="panel-title" fill="#0f172a">图例说明</text>')
    for item_index, (color, label) in enumerate([("#52aab8", "蓝绿渐变：水深底图"), ("#bde6e8", "浅青色：测线覆盖区域"), ("#e75e36", "红橙色：重叠覆盖区域"), ("#ffffff", "白色：实际测线")]):
        y0 = panel_y + 350 + item_index * 34
        parts.append(f'<rect class="thin" x="{panel_x + 24}" y="{y0}" width="28" height="18" rx="3" fill="{color}" stroke="#64748b"/>')
        parts.append(f'<text x="{panel_x + 64}" y="{y0 + 14}" class="panel-text" fill="#1e293b">{label}</text>')
    parts.append("</svg>")
    SVG_PATH.write_text("\n".join(parts), encoding="utf-8")


def save_plot(phi_deg: float, phi_span_deg: float, eta_target: float, lines: list[dict]) -> None:
    draw_png(phi_deg, phi_span_deg, eta_target, lines)
    draw_svg(phi_deg, phi_span_deg, eta_target, lines)
    print("saved_png:", PNG_PATH, flush=True)
    print("saved_svg:", SVG_PATH, flush=True)


def main() -> None:
    print("=== 2023B \u95ee\u9898\u4e09\uff1a\u975e\u5b8c\u5168\u5e73\u884c PSO \u7a0b\u5e8f\u542f\u52a8 ===", flush=True)
    best_position, best_fitness, best_lines, iterations_used, converged_iters = improved_pso()
    phi_deg, phi_span_deg, eta_target = best_position
    total_length_nm = sum(line["length"] for line in best_lines) / NM
    overlap_rates = [
        (b["right"] - a["left"]) / min(a["width"], b["width"]) * 100
        for a, b in zip(best_lines, best_lines[1:])
    ]

    print("[4/6] 迭代完成，开始输出图片", flush=True)
    if SAVE_CSV:
        save_lines(phi_deg, phi_span_deg, eta_target, best_lines)
    save_plot(phi_deg, phi_span_deg, eta_target, best_lines)
    print("文件保存完成，输出最终结果", flush=True)

    print("best_phi_deg:", round(phi_deg, 6))
    print("best_phi_span_deg:", round(phi_span_deg, 6))
    print("best_eta_target:", round(eta_target, 6))
    print("iterations_used:", iterations_used)
    print("converged_iterations:", converged_iters)
    print("line_count:", len(best_lines))
    print("total_length_nm:", round(total_length_nm, 6))
    print("fitness:", round(best_fitness, 6))
    if overlap_rates:
        print("overlap_min_percent:", round(min(overlap_rates), 6))
        print("overlap_max_percent:", round(max(overlap_rates), 6))
    if SAVE_CSV:
        print("saved:", OUTPUT_PATH)
    print("saved_png:", PNG_PATH)
    print("saved_svg:", SVG_PATH)

if __name__ == "__main__":
    main()
