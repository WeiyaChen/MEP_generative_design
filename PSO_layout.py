"""
test3_pso.py — Particle Swarm Optimization for 2D Rectangle Packing.

Angles are constrained to integer multiples of 10 degrees.
Follows the same geometry / fitness / repair logic as the GA version.
"""
from __future__ import annotations
import argparse
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

# ============================================================================
# 基础类型 & 几何工具（与 test3.py 完全一致）
# ============================================================================
from layout_common_def import (
    Point,
    EPS,
    VALID_ANGLES_DEG,
    RectangleSpec,
    SquareObstacle,
    PlacedRect,
    clamp,
    convex_overlap_strict,
    rect_inside_polygon,
    axis_alignment_deviation,
    snap_angle_to_10deg,
)


# ---------------------------------------------------------------------------
# 单个粒子的位置 + 速度
# ---------------------------------------------------------------------------
@dataclass
class Particle:
    """One particle = one candidate layout."""
    # position  — 每个矩形 4 个值: [cx, cy, angle_deg, active_prob]
    position: List[float] = field(default_factory=list)
    # velocity  — 同样维度
    velocity: List[float] = field(default_factory=list)
    # personal best
    pbest_position: List[float] = field(default_factory=list)
    pbest_fitness: float = -math.inf


# ============================================================================
# PSO 求解器
# ============================================================================
class PSORectanglePacker:
    """Particle Swarm Optimization for rectangle packing.

    Each rectangle contributes 4 decision variables:
        cx, cy  : centre coordinates (continuous)
        angle   : rotation in degrees → snapped to multiples of 10°
        active  : probability of inclusion → thresholded at 0.5
    """

    def __init__(
        self,
        polygon: Sequence[Point],
        obstacles: Sequence[SquareObstacle],
        rectangles: Sequence[RectangleSpec],
        swarm_size: int = 40,
        iterations: int = 300,
        w_start: float = 0.9,          # inertia start (linear decay)
        w_end: float = 0.4,            # inertia end
        c1: float = 1.8,               # cognitive coefficient
        c2: float = 1.8,               # social coefficient
        vmax_cxy_ratio: float = 0.15,  # v_max for cx,cy = ratio * range
        vmax_angle: float = 60.0,      # v_max for angle (degrees)
        vmax_active: float = 0.35,     # v_max for active probability
        angle_preference_weight: float = 50.0,
        random_seed: int | None = 7,
    ) -> None:
        if len(polygon) < 3:
            raise ValueError("polygon must have at least 3 points")

        self.polygon = list(polygon)
        self.obstacles = list(obstacles)
        self.rectangles = list(rectangles)

        self.swarm_size = swarm_size
        self.iterations = iterations
        self.w_start = w_start
        self.w_end = w_end
        self.c1 = c1
        self.c2 = c2
        self.angle_preference_weight = angle_preference_weight

        self.rng = random.Random(random_seed)

        # 包围盒
        xs = [p[0] for p in self.polygon]
        ys = [p[1] for p in self.polygon]
        self.min_x, self.max_x = min(xs), max(xs)
        self.min_y, self.max_y = min(ys), max(ys)

        # 速度上限
        span_x = self.max_x - self.min_x
        span_y = self.max_y - self.min_y
        self.vmax_cx = vmax_cxy_ratio * span_x
        self.vmax_cy = vmax_cxy_ratio * span_y
        self.vmax_angle = vmax_angle
        self.vmax_active = vmax_active

        self._obstacle_polys = [o.as_polygon() for o in self.obstacles]
        self._poly_area = self._polygon_area(self.polygon)

        # 每个矩形 4 维: cx, cy, angle_deg, active
        self.dim = len(self.rectangles) * 4

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------
    @staticmethod
    def _polygon_area(polygon: Sequence[Point]) -> float:
        area = 0.0
        n = len(polygon)
        for i in range(n):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % n]
            area += x1 * y2 - x2 * y1
        return abs(area) / 2.0

    def _inertia(self, t: int) -> float:
        """Linear decay of inertia weight."""
        return self.w_start - (self.w_start - self.w_end) * (t / max(self.iterations - 1, 1))

    # ------------------------------------------------------------------
    # 粒子初始化
    # ------------------------------------------------------------------
    def _random_position(self) -> List[float]:
        """Generate one random position vector."""
        pos: List[float] = []
        for spec in self.rectangles:
            # cx, cy
            pos.append(self.rng.uniform(self.min_x, self.max_x))
            pos.append(self.rng.uniform(self.min_y, self.max_y))
            # angle (degree) — only multiples of 10
            if spec.rotatable:
                r = self.rng.random()
                if r < 0.9:
                    pos.append(float(self.rng.choice([0, 90, 180, 270])))
                else:
                    pos.append(self.rng.choice(VALID_ANGLES_DEG))
            else:
                pos.append(0.0)
            # active probability
            if spec.mandatory:
                pos.append(1.0)
            else:
                pos.append(0.5 + self.rng.uniform(-0.3, 0.5))  # bias towards active
        return pos

    def _random_velocity(self) -> List[float]:
        """Small random initial velocity."""
        vel: List[float] = []
        for i in range(0, self.dim, 4):
            vel.append(self.rng.uniform(-self.vmax_cx * 0.5, self.vmax_cx * 0.5))
            vel.append(self.rng.uniform(-self.vmax_cy * 0.5, self.vmax_cy * 0.5))
            vel.append(self.rng.uniform(-self.vmax_angle * 0.5, self.vmax_angle * 0.5))
            vel.append(self.rng.uniform(-self.vmax_active * 0.5, self.vmax_active * 0.5))
        return vel

    def _init_swarm(self) -> List[Particle]:
        swarm: List[Particle] = []
        for _ in range(self.swarm_size):
            pos = self._random_position()
            vel = self._random_velocity()
            p = Particle(
                position=pos,
                velocity=vel,
                pbest_position=list(pos),
            )
            swarm.append(p)
        return swarm

    # ------------------------------------------------------------------
    # 解码：position → List[PlacedRect]
    # ------------------------------------------------------------------
    def _decode(self, position: List[float]) -> List[PlacedRect]:
        out: List[PlacedRect] = []
        for j, spec in enumerate(self.rectangles):
            base = j * 4
            cx = position[base]
            cy = position[base + 1]
            angle_deg = position[base + 2]
            active_prob = position[base + 3]

            active = spec.mandatory or (active_prob >= 0.5)
            if not active:
                continue

            angle_rad = math.radians(angle_deg)
            angle_rad = snap_angle_to_10deg(angle_rad)

            out.append(
                PlacedRect(
                    id=spec.id,
                    cx=clamp(cx, self.min_x, self.max_x),
                    cy=clamp(cy, self.min_y, self.max_y),
                    width=spec.width,
                    height=spec.height,
                    angle=angle_rad,
                    mandatory=spec.mandatory,
                )
            )
        return out

    # ------------------------------------------------------------------
    # 可行性检查
    # ------------------------------------------------------------------
    def _is_rect_valid(
        self, rect: PlacedRect, already_kept: Sequence[PlacedRect]
    ) -> bool:
        if not rect_inside_polygon(rect, self.polygon):
            return False
        rp = rect.corners()
        for op in self._obstacle_polys:
            if convex_overlap_strict(rp, op):
                return False
        for other in already_kept:
            if convex_overlap_strict(rp, other.corners()):
                return False
        return True

    # ------------------------------------------------------------------
    # 适应度（与 GA 一致）
    # ------------------------------------------------------------------
    def _fitness(self, position: List[float]) -> float:
        placed = self._decode(position)
        area_reward = sum(r.area for r in placed)
        count_reward = 8.0 * len(placed)
        penalty = 0.0

        for rect in placed:
            rect_poly = rect.corners()
            if not rect_inside_polygon(rect, self.polygon):
                penalty += 520.0 + 6.0 * rect.area
            for op in self._obstacle_polys:
                if convex_overlap_strict(rect_poly, op):
                    penalty += 760.0
            penalty += self.angle_preference_weight * axis_alignment_deviation(rect.angle)

        for i in range(len(placed)):
            pi = placed[i].corners()
            for j in range(i + 1, len(placed)):
                pj = placed[j].corners()
                if convex_overlap_strict(pi, pj):
                    penalty += 900.0

        return area_reward + count_reward - penalty

    # ------------------------------------------------------------------
    # 速度更新
    # ------------------------------------------------------------------
    def _update_velocity(
        self,
        particle: Particle,
        gbest_position: List[float],
        w: float,
    ) -> None:
        for d in range(self.dim):
            r1 = self.rng.random()
            r2 = self.rng.random()
            cognitive = self.c1 * r1 * (particle.pbest_position[d] - particle.position[d])
            social = self.c2 * r2 * (gbest_position[d] - particle.position[d])
            particle.velocity[d] = w * particle.velocity[d] + cognitive + social

    # ------------------------------------------------------------------
    # 位置更新 + 边界处理 + snap
    # ------------------------------------------------------------------
    def _update_position(self, particle: Particle) -> None:
        for j, spec in enumerate(self.rectangles):
            base = j * 4

            # cx
            particle.position[base] += particle.velocity[base]
            particle.position[base] = clamp(
                particle.position[base], self.min_x, self.max_x
            )
            particle.velocity[base] = clamp(
                particle.velocity[base], -self.vmax_cx, self.vmax_cx
            )

            # cy
            particle.position[base + 1] += particle.velocity[base + 1]
            particle.position[base + 1] = clamp(
                particle.position[base + 1], self.min_y, self.max_y
            )
            particle.velocity[base + 1] = clamp(
                particle.velocity[base + 1], -self.vmax_cy, self.vmax_cy
            )

            # angle (degrees)
            particle.position[base + 2] += particle.velocity[base + 2]
            # snap to 10° multiple
            particle.position[base + 2] = (
                round(particle.position[base + 2] / 10.0) * 10.0
            )
            # wrap to [0, 360)
            particle.position[base + 2] %= 360.0
            if not spec.rotatable:
                particle.position[base + 2] = 0.0
            particle.velocity[base + 2] = clamp(
                particle.velocity[base + 2], -self.vmax_angle, self.vmax_angle
            )

            # active probability
            particle.position[base + 3] += particle.velocity[base + 3]
            if spec.mandatory:
                particle.position[base + 3] = 1.0
                particle.velocity[base + 3] = 0.0
            else:
                particle.position[base + 3] = clamp(particle.position[base + 3], 0.0, 1.0)
            particle.velocity[base + 3] = clamp(
                particle.velocity[base + 3], -self.vmax_active, self.vmax_active
            )

    # ------------------------------------------------------------------
    # 修复（与 GA 一致）
    # ------------------------------------------------------------------
    def _repair_feasible_layout(
        self, position: List[float]
    ) -> Tuple[List[PlacedRect], List[str]]:
        decoded = self._decode(position)
        mandatory = [r for r in decoded if r.mandatory]
        optional = [r for r in decoded if not r.mandatory]

        kept: List[PlacedRect] = []
        dropped_mandatory: List[str] = []

        for rect in mandatory:
            if self._is_rect_valid(rect, kept):
                kept.append(rect)
            else:
                dropped_mandatory.append(rect.id)

        optional_sorted = sorted(
            optional,
            key=lambda r: (r.area - 0.5 * axis_alignment_deviation(r.angle) * r.area),
            reverse=True,
        )
        for rect in optional_sorted:
            if self._is_rect_valid(rect, kept):
                kept.append(rect)

        return kept, dropped_mandatory

    # ------------------------------------------------------------------
    # 求解主循环
    # ------------------------------------------------------------------
    def solve(self) -> Dict[str, object]:
        swarm = self._init_swarm()

        # 初始化 personal best
        for p in swarm:
            p.pbest_fitness = self._fitness(p.position)
            p.pbest_position = list(p.position)

        # 全局最优
        gbest_particle = max(swarm, key=lambda p: p.pbest_fitness)
        gbest_position = list(gbest_particle.pbest_position)
        gbest_fitness = gbest_particle.pbest_fitness

        fitness_history: List[float] = []

        for t in range(self.iterations):
            w = self._inertia(t)

            for p in swarm:
                self._update_velocity(p, gbest_position, w)
                self._update_position(p)

                fit = self._fitness(p.position)

                # 更新 personal best
                if fit > p.pbest_fitness:
                    p.pbest_fitness = fit
                    p.pbest_position = list(p.position)

                # 更新 global best
                if fit > gbest_fitness:
                    gbest_fitness = fit
                    gbest_position = list(p.position)

            fitness_history.append(gbest_fitness)

        final_layout, dropped_mandatory = self._repair_feasible_layout(gbest_position)
        used_area = sum(r.area for r in final_layout)
        utilization = used_area / self._poly_area if self._poly_area > EPS else 0.0

        return {
            "best_fitness": gbest_fitness,
            "polygon_area": self._poly_area,
            "placed_area": used_area,
            "utilization": utilization,
            "placed_rectangles": final_layout,
            "dropped_mandatory": dropped_mandatory,
            "fitness_history": fitness_history,
        }


# ============================================================================
# 可视化（与 GA 版本一致）
# ============================================================================
def visualize_layout(
    polygon: Sequence[Point],
    obstacles: Sequence[SquareObstacle],
    placed_rectangles: Sequence[PlacedRect],
    fitness_history: Sequence[float] | None = None,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    try:
        from matplotlib import patches
        import matplotlib.pyplot as plt
    except ImportError:
        target = save_path or "layout_result_pso.svg"
        if not target.lower().endswith(".svg"):
            target = f"{target}.svg"
        save_layout_svg(polygon, obstacles, placed_rectangles, target)
        print("matplotlib is not installed, fallback to SVG export.")
        print(f"figure_saved       : {target}")
        return

    if fitness_history:
        fig, (ax_layout, ax_fit) = plt.subplots(1, 2, figsize=(14, 6))
    else:
        fig, ax_layout = plt.subplots(1, 1, figsize=(8, 6))
        ax_fit = None

    ax_layout.add_patch(
        patches.Polygon(
            polygon, closed=True, fill=True,
            facecolor="#EEF7ED", edgecolor="#2D6A4F", linewidth=2.0,
        )
    )

    for obs in obstacles:
        ax_layout.add_patch(
            patches.Polygon(
                obs.as_polygon(), closed=True,
                facecolor="#6C757D", edgecolor="#212529",
                linewidth=1.5, hatch="//", alpha=0.85,
            )
        )

    cmap = plt.colormaps.get_cmap("tab20")
    for i, rect in enumerate(placed_rectangles):
        pts = rect.corners()
        color = cmap(i % 20)
        ax_layout.add_patch(
            patches.Polygon(
                pts, closed=True,
                facecolor=color, edgecolor="black",
                linewidth=1.2, alpha=0.66,
            )
        )
        angle_deg = math.degrees(rect.angle)
        ax_layout.text(
            rect.cx, rect.cy,
            f"{rect.id}\n{angle_deg:.0f}°",
            ha="center", va="center",
            fontsize=8, color="black", weight="bold",
        )

    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    ax_layout.set_xlim(min(xs) - 2, max(xs) + 2)
    ax_layout.set_ylim(min(ys) - 2, max(ys) + 2)
    ax_layout.set_aspect("equal", adjustable="box")
    ax_layout.set_title("Rectangle Layout — PSO (Angles: multiples of 10°)")
    ax_layout.set_xlabel("X")
    ax_layout.set_ylabel("Y")
    ax_layout.grid(True, alpha=0.2)

    if ax_fit is not None and fitness_history:
        ax_fit.plot(
            range(1, len(fitness_history) + 1),
            fitness_history, color="#E76F51", lw=2.0,
        )
        ax_fit.set_title("PSO Best Fitness Over Iterations")
        ax_fit.set_xlabel("Iteration")
        ax_fit.set_ylabel("Best Fitness")
        ax_fit.grid(True, alpha=0.25)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        print(f"figure_saved       : {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def save_layout_svg(
    polygon: Sequence[Point],
    obstacles: Sequence[SquareObstacle],
    placed_rectangles: Sequence[PlacedRect],
    output_path: str,
) -> None:
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    canvas_w = 1000.0
    canvas_h = 700.0
    margin = 40.0
    dx = max(max_x - min_x, 1e-6)
    dy = max(max_y - min_y, 1e-6)
    scale = min((canvas_w - 2 * margin) / dx, (canvas_h - 2 * margin) / dy)

    def to_svg_point(p: Point) -> Tuple[float, float]:
        sx = margin + (p[0] - min_x) * scale
        sy = canvas_h - (margin + (p[1] - min_y) * scale)
        return sx, sy

    def poly_to_points(poly: Sequence[Point]) -> str:
        return " ".join(
            f"{x:.2f},{y:.2f}" for x, y in (to_svg_point(p) for p in poly)
        )

    colors = [
        "#A8DADC", "#F4A261", "#2A9D8F", "#E76F51", "#8AB17D",
        "#F6BD60", "#90CAF9", "#BDB2FF", "#FFD6A5", "#84A59D",
    ]

    lines: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{int(canvas_w)}" '
        f'height="{int(canvas_h)}" viewBox="0 0 {int(canvas_w)} {int(canvas_h)}">',
        '<rect width="100%" height="100%" fill="#FAFAFA"/>',
        f'<polygon points="{poly_to_points(polygon)}" '
        f'fill="#EEF7ED" stroke="#2D6A4F" stroke-width="2"/>',
    ]
    for obs in obstacles:
        lines.append(
            f'<polygon points="{poly_to_points(obs.as_polygon())}" '
            f'fill="#6C757D" stroke="#212529" stroke-width="1.5"/>'
        )
    for i, rect in enumerate(placed_rectangles):
        c = colors[i % len(colors)]
        rect_pts = rect.corners()
        lines.append(
            f'<polygon points="{poly_to_points(rect_pts)}" '
            f'fill="{c}" fill-opacity="0.72" stroke="#111" stroke-width="1.2"/>'
        )
        tx, ty = to_svg_point((rect.cx, rect.cy))
        angle_deg = math.degrees(rect.angle)
        lines.append(
            f'<text x="{tx:.2f}" y="{ty:.2f}" font-size="12" '
            f'text-anchor="middle" dominant-baseline="central" '
            f'fill="#111" font-weight="700">{rect.id} {angle_deg:.0f}°</text>'
        )
    lines.append("</svg>")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ============================================================================
# demo & CLI
# ============================================================================
def demo(visualize: bool = True, save_path: str | None = None, show_plot: bool = True) -> None:
    polygon = [
        (0.0, 0.0),
        (42.0, 2.0),
        (38.0, 26.0),
        (24.0, 32.0),
        (8.0, 28.0),
        (-2.0, 14.0),
    ]

    obstacles = [
        SquareObstacle(x=9.0, y=8.0, size=4.0),
        SquareObstacle(x=23.0, y=12.0, size=5.0),
        SquareObstacle(x=30.0, y=20.0, size=3.0),
    ]

    rectangles = [
        RectangleSpec("R1", 7, 5, rotatable=True, mandatory=True),
        RectangleSpec("R2", 9, 4, rotatable=True, mandatory=True),
        RectangleSpec("R3", 6, 6, rotatable=True, mandatory=False),
        RectangleSpec("R4", 5, 4, rotatable=True, mandatory=False),
        RectangleSpec("R5", 8, 3, rotatable=True, mandatory=False),
        RectangleSpec("R6", 4, 4, rotatable=True, mandatory=False),
        RectangleSpec("R7", 3, 7, rotatable=True, mandatory=False),
        RectangleSpec("R8", 10, 2, rotatable=True, mandatory=False),
        RectangleSpec("R9", 4, 6, rotatable=True, mandatory=False),
    ]

    solver = PSORectanglePacker(
        polygon=polygon,
        obstacles=obstacles,
        rectangles=rectangles,
        swarm_size=40,
        iterations=300,
        w_start=0.9,
        w_end=0.4,
        c1=1.8,
        c2=1.8,
        angle_preference_weight=50.0,
        random_seed=42,
    )
    result = solver.solve()

    print("=== PSO Rectangle Layout Result (Angles: multiples of 10°) ===")
    print(f"best_fitness      : {result['best_fitness']:.3f}")
    print(f"polygon_area      : {result['polygon_area']:.3f}")
    print(f"placed_area       : {result['placed_area']:.3f}")
    print(f"utilization       : {result['utilization'] * 100:.2f}%")
    if result["dropped_mandatory"]:
        print(f"dropped_mandatory : {result['dropped_mandatory']}")

    print("placed_rectangles:")
    for r in result["placed_rectangles"]:
        print(
            f"  {r.id}: cx={r.cx:.2f}, cy={r.cy:.2f}, "
            f"w={r.width:.2f}, h={r.height:.2f}, angle_deg={math.degrees(r.angle):.1f}"
        )

    if visualize:
        visualize_layout(
            polygon=polygon,
            obstacles=obstacles,
            placed_rectangles=result["placed_rectangles"],
            fitness_history=result.get("fitness_history"),
            save_path=save_path,
            show=show_plot,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PSO rectangle layout — angles constrained to multiples of 10°."
    )
    parser.add_argument("--no-vis", action="store_true", help="Disable plotting.")
    parser.add_argument("--save-fig", type=str, default=None, help="Save figure to path.")
    parser.add_argument("--no-show", action="store_true", help="Do not open plot window.")
    args = parser.parse_args()

    demo(
        visualize=not args.no_vis,
        save_path=args.save_fig,
        show_plot=not args.no_show,
    )