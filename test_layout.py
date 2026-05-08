"""
test_compare.py — GA vs PSO 性能对比测试（二维矩形排布）

测试用例从易到难，包括：
  - 规则矩形边界
  - 不规则多边形边界
  - 带障碍物场景

对比指标：运行时间、适应度、空间利用率、放置数量、丢失必要矩形数

输出：指定文件夹内按编号存放结果文件
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence, Tuple

# ── 导入两个求解器 ─────────────────────────────────────────────
# 确保 test3.py 和 test3_pso.py 在当前目录
try:
    from GA_layout import (
        GeneticRectanglePacker,
        RectangleSpec,
        SquareObstacle,
        PlacedRect,
        Point,
        visualize_layout,
    )
except ImportError:
    print("错误: 找不到 test3.py，请确保该文件在当前目录下。")
    sys.exit(1)

try:
    from PSO_layout import PSORectanglePacker
except ImportError:
    print("错误: 找不到 test3_pso.py，请确保该文件在当前目录下。")
    sys.exit(1)


# ============================================================================
# 测试用例定义
# ============================================================================
@dataclass
class TestCase:
    """单个测试用例的完整描述"""
    id: int
    name: str
    difficulty: str            # "easy" / "medium" / "hard"
    polygon: List[Point]       # 边界多边形
    obstacles: List[SquareObstacle]
    rectangles: List[RectangleSpec]


# ── 辅助函数：生成轴对齐矩形边界 ──────────────────────────────
def rect_polygon(w: float, h: float) -> List[Point]:
    return [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]


# ── 测试用例库 ─────────────────────────────────────────────────
TEST_CASES: List[TestCase] = [
    # ──────── 简单 ────────
    TestCase(
        id=1,
        name="矩形20×15_3个矩形",
        difficulty="easy",
        polygon=rect_polygon(20.0, 15.0),
        obstacles=[],
        rectangles=[
            RectangleSpec("A1", 5, 4, rotatable=True, mandatory=True),
            RectangleSpec("A2", 6, 4, rotatable=True, mandatory=False),
            RectangleSpec("A3", 4, 3, rotatable=True, mandatory=False),
        ],
    ),
    TestCase(
        id=2,
        name="矩形30×20_6个矩形",
        difficulty="easy",
        polygon=rect_polygon(30.0, 20.0),
        obstacles=[],
        rectangles=[
            RectangleSpec("B1", 8, 5, rotatable=True, mandatory=True),
            RectangleSpec("B2", 7, 6, rotatable=True, mandatory=True),
            RectangleSpec("B3", 6, 4, rotatable=True, mandatory=False),
            RectangleSpec("B4", 5, 5, rotatable=True, mandatory=False),
            RectangleSpec("B5", 9, 3, rotatable=True, mandatory=False),
            RectangleSpec("B6", 4, 8, rotatable=True, mandatory=False),
        ],
    ),
    TestCase(
        id=3,
        name="矩形40×28_8个矩形",
        difficulty="easy",
        polygon=rect_polygon(40.0, 28.0),
        obstacles=[],
        rectangles=[
            RectangleSpec("C1", 9, 6, rotatable=True, mandatory=True),
            RectangleSpec("C2", 10, 5, rotatable=True, mandatory=True),
            RectangleSpec("C3", 7, 7, rotatable=True, mandatory=False),
            RectangleSpec("C4", 8, 4, rotatable=True, mandatory=False),
            RectangleSpec("C5", 6, 8, rotatable=True, mandatory=False),
            RectangleSpec("C6", 11, 3, rotatable=True, mandatory=False),
            RectangleSpec("C7", 5, 6, rotatable=True, mandatory=False),
            RectangleSpec("C8", 4, 9, rotatable=True, mandatory=False),
        ],
    ),

    # ──────── 中等 ────────
    TestCase(
        id=4,
        name="L形多边形_6个矩形",
        difficulty="medium",
        polygon=[
            (0, 0), (30, 0), (30, 8), (12, 8),
            (12, 24), (0, 24),
        ],
        obstacles=[],
        rectangles=[
            RectangleSpec("D1", 8, 5, rotatable=True, mandatory=True),
            RectangleSpec("D2", 7, 6, rotatable=True, mandatory=True),
            RectangleSpec("D3", 6, 4, rotatable=True, mandatory=False),
            RectangleSpec("D4", 5, 5, rotatable=True, mandatory=False),
            RectangleSpec("D5", 9, 3, rotatable=True, mandatory=False),
            RectangleSpec("D6", 4, 7, rotatable=True, mandatory=False),
        ],
    ),
    TestCase(
        id=5,
        name="五边形_7个矩形_1障碍物",
        difficulty="medium",
        polygon=[
            (0, 0), (35, 0), (38, 18), (18, 30), (-2, 16),
        ],
        obstacles=[
            SquareObstacle(x=14, y=10, size=4),
        ],
        rectangles=[
            RectangleSpec("E1", 8, 6, rotatable=True, mandatory=True),
            RectangleSpec("E2", 9, 5, rotatable=True, mandatory=True),
            RectangleSpec("E3", 7, 7, rotatable=True, mandatory=False),
            RectangleSpec("E4", 6, 4, rotatable=True, mandatory=False),
            RectangleSpec("E5", 10, 3, rotatable=True, mandatory=False),
            RectangleSpec("E6", 5, 8, rotatable=True, mandatory=False),
            RectangleSpec("E7", 4, 5, rotatable=True, mandatory=False),
        ],
    ),
    TestCase(
        id=6,
        name="六边形_9个矩形_2障碍物",
        difficulty="medium",
        polygon=[
            (0, 0), (40, 2), (42, 24), (22, 34), (6, 30), (-4, 12),
        ],
        obstacles=[
            SquareObstacle(x=10, y=8, size=4),
            SquareObstacle(x=28, y=16, size=3.5),
        ],
        rectangles=[
            RectangleSpec("F1", 8, 6, rotatable=True, mandatory=True),
            RectangleSpec("F2", 9, 5, rotatable=True, mandatory=True),
            RectangleSpec("F3", 7, 7, rotatable=True, mandatory=False),
            RectangleSpec("F4", 6, 4, rotatable=True, mandatory=False),
            RectangleSpec("F5", 10, 3, rotatable=True, mandatory=False),
            RectangleSpec("F6", 5, 8, rotatable=True, mandatory=False),
            RectangleSpec("F7", 4, 5, rotatable=True, mandatory=False),
            RectangleSpec("F8", 11, 2.5, rotatable=True, mandatory=False),
            RectangleSpec("F9", 3, 9, rotatable=True, mandatory=False),
        ],
    ),

    # ──────── 困难 ────────
    TestCase(
        id=7,
        name="原demo六边形_9个矩形_3障碍物",
        difficulty="hard",
        polygon=[
            (0.0, 0.0), (42.0, 2.0), (38.0, 26.0),
            (24.0, 32.0), (8.0, 28.0), (-2.0, 14.0),
        ],
        obstacles=[
            SquareObstacle(x=9.0, y=8.0, size=4.0),
            SquareObstacle(x=23.0, y=12.0, size=5.0),
            SquareObstacle(x=30.0, y=20.0, size=3.0),
        ],
        rectangles=[
            RectangleSpec("G1", 7, 5, rotatable=True, mandatory=True),
            RectangleSpec("G2", 9, 4, rotatable=True, mandatory=True),
            RectangleSpec("G3", 6, 6, rotatable=True, mandatory=False),
            RectangleSpec("G4", 5, 4, rotatable=True, mandatory=False),
            RectangleSpec("G5", 8, 3, rotatable=True, mandatory=False),
            RectangleSpec("G6", 4, 4, rotatable=True, mandatory=False),
            RectangleSpec("G7", 3, 7, rotatable=True, mandatory=False),
            RectangleSpec("G8", 10, 2, rotatable=True, mandatory=False),
            RectangleSpec("G9", 4, 6, rotatable=True, mandatory=False),
        ],
    ),
    TestCase(
        id=8,
        name="凹多边形_12个矩形_4障碍物",
        difficulty="hard",
        polygon=[
            (0, 0), (50, 0), (50, 10), (32, 10),
            (32, 20), (50, 20), (50, 38), (0, 38),
        ],
        obstacles=[
            SquareObstacle(x=8, y=6, size=3),
            SquareObstacle(x=38, y=24, size=4),
            SquareObstacle(x=22, y=15, size=3.5),
            SquareObstacle(x=6, y=28, size=3),
        ],
        rectangles=[
            RectangleSpec("H1", 7, 5, rotatable=True, mandatory=True),
            RectangleSpec("H2", 9, 4, rotatable=True, mandatory=True),
            RectangleSpec("H3", 6, 6, rotatable=True, mandatory=True),
            RectangleSpec("H4", 8, 5, rotatable=True, mandatory=False),
            RectangleSpec("H5", 5, 7, rotatable=True, mandatory=False),
            RectangleSpec("H6", 10, 3, rotatable=True, mandatory=False),
            RectangleSpec("H7", 4, 6, rotatable=True, mandatory=False),
            RectangleSpec("H8", 7, 3, rotatable=True, mandatory=False),
            RectangleSpec("H9", 6, 8, rotatable=True, mandatory=False),
            RectangleSpec("H10", 9, 2.5, rotatable=True, mandatory=False),
            RectangleSpec("H11", 3, 9, rotatable=True, mandatory=False),
            RectangleSpec("H12", 5, 4, rotatable=True, mandatory=False),
        ],
    ),
    TestCase(
        id=9,
        name="复杂十二边形_15个矩形_5障碍物",
        difficulty="hard",
        polygon=[
            (0, 0), (24, -1), (48, 3), (52, 16),
            (46, 30), (50, 42), (34, 48), (18, 44),
            (6, 46), (-4, 36), (-2, 22), (2, 8),
        ],
        obstacles=[
            SquareObstacle(x=8, y=6, size=3),
            SquareObstacle(x=30, y=10, size=4),
            SquareObstacle(x=38, y=28, size=3.5),
            SquareObstacle(x=16, y=30, size=3),
            SquareObstacle(x=42, y=38, size=2.5),
        ],
        rectangles=[
            RectangleSpec("I1", 7, 5, rotatable=True, mandatory=True),
            RectangleSpec("I2", 9, 4, rotatable=True, mandatory=True),
            RectangleSpec("I3", 6, 6, rotatable=True, mandatory=True),
            RectangleSpec("I4", 8, 5, rotatable=True, mandatory=False),
            RectangleSpec("I5", 5, 7, rotatable=True, mandatory=False),
            RectangleSpec("I6", 10, 3, rotatable=True, mandatory=False),
            RectangleSpec("I7", 4, 6, rotatable=True, mandatory=False),
            RectangleSpec("I8", 7, 3, rotatable=True, mandatory=False),
            RectangleSpec("I9", 6, 8, rotatable=True, mandatory=False),
            RectangleSpec("I10", 9, 2.5, rotatable=True, mandatory=False),
            RectangleSpec("I11", 3, 9, rotatable=True, mandatory=False),
            RectangleSpec("I12", 5, 4, rotatable=True, mandatory=False),
            RectangleSpec("I13", 8, 3.5, rotatable=True, mandatory=False),
            RectangleSpec("I14", 4, 7, rotatable=True, mandatory=False),
            RectangleSpec("I15", 6, 4, rotatable=True, mandatory=False),
        ],
    ),
]


# ============================================================================
# 运行器
# ============================================================================
@dataclass
class RunResult:
    """一次运行的结果汇总"""
    algorithm: str               # "GA" or "PSO"
    runtime_sec: float
    best_fitness: float
    polygon_area: float
    placed_area: float
    utilization: float           # 0..1
    placed_count: int
    dropped_mandatory: List[str]
    fitness_history: List[float]
    placed_rectangles: List[PlacedRect]


def run_ga(tc: TestCase, seed: int = 42) -> RunResult:
    """运行遗传算法并返回结果"""
    # 使用适中的参数（约 20,000 次适应度评估）
    solver = GeneticRectanglePacker(
        polygon=tc.polygon,
        obstacles=tc.obstacles,
        rectangles=tc.rectangles,
        population_size=80,
        generations=250,          # 80 × 250 = 20,000
        crossover_rate=0.85,
        mutation_rate=0.18,
        elite_count=5,
        tournament_size=3,
        angle_preference_weight=50.0,
        random_seed=seed,
    )

    t0 = time.perf_counter()
    result = solver.solve()
    t1 = time.perf_counter()

    return RunResult(
        algorithm="GA",
        runtime_sec=t1 - t0,
        best_fitness=float(result["best_fitness"]),
        polygon_area=float(result["polygon_area"]),
        placed_area=float(result["placed_area"]),
        utilization=float(result["utilization"]),
        placed_count=len(result["placed_rectangles"]),
        dropped_mandatory=list(result["dropped_mandatory"]),
        fitness_history=list(result["fitness_history"]),
        placed_rectangles=list(result["placed_rectangles"]),
    )


def run_pso(tc: TestCase, seed: int = 42) -> RunResult:
    """运行粒子群算法并返回结果"""
    solver = PSORectanglePacker(
        polygon=tc.polygon,
        obstacles=tc.obstacles,
        rectangles=tc.rectangles,
        swarm_size=50,
        iterations=400,           # 50 × 400 = 20,000
        w_start=0.9,
        w_end=0.4,
        c1=1.8,
        c2=1.8,
        angle_preference_weight=50.0,
        random_seed=seed,
    )

    t0 = time.perf_counter()
    result = solver.solve()
    t1 = time.perf_counter()

    return RunResult(
        algorithm="PSO",
        runtime_sec=t1 - t0,
        best_fitness=float(result["best_fitness"]),
        polygon_area=float(result["polygon_area"]),
        placed_area=float(result["placed_area"]),
        utilization=float(result["utilization"]),
        placed_count=len(result["placed_rectangles"]),
        dropped_mandatory=list(result["dropped_mandatory"]),
        fitness_history=list(result["fitness_history"]),
        placed_rectangles=list(result["placed_rectangles"]),
    )


# ============================================================================
# 报告生成
# ============================================================================
def format_table_row(
    case_id: int,
    name: str,
    diff: str,
    ga: RunResult,
    pso: RunResult,
) -> str:
    """格式化为对齐的文本行"""
    faster = "GA" if ga.runtime_sec < pso.runtime_sec else "PSO"
    better_util = "GA" if ga.utilization >= pso.utilization else "PSO"
    time_ratio = ga.runtime_sec / max(pso.runtime_sec, 1e-6)

    lines = [
        f"{'─'*90}",
        f" 案例 {case_id:02d} │ {name}  [{diff}]",
        f"{'─'*90}",
        f" 指标                  │ {'GA':>20s} │ {'PSO':>20s} │ 对比",
        f"{'─'*90}",
        f" 运行时间 (秒)         │ {ga.runtime_sec:>20.3f} │ {pso.runtime_sec:>20.3f} │ 更快: {faster}  ({time_ratio:.2f}x)",
        f" 最佳适应度            │ {ga.best_fitness:>20.2f} │ {pso.best_fitness:>20.2f} │ {'GA' if ga.best_fitness>=pso.best_fitness else 'PSO'} 更优",
        f" 空间利用率 (%)        │ {ga.utilization*100:>19.2f}% │ {pso.utilization*100:>19.2f}% │ {better_util} 更高",
        f" 已放置矩形数          │ {ga.placed_count:>20d} │ {pso.placed_count:>20d} │",
        f" 已放置面积            │ {ga.placed_area:>20.2f} │ {pso.placed_area:>20.2f} │",
        f" 多边形面积            │ {ga.polygon_area:>20.2f} │ {pso.polygon_area:>20.2f} │",
    ]

    if ga.dropped_mandatory or pso.dropped_mandatory:
        lines.append(f" ⚠ GA  丢弃的必要矩形: {ga.dropped_mandatory if ga.dropped_mandatory else '无'}")
        lines.append(f" ⚠ PSO 丢弃的必要矩形: {pso.dropped_mandatory if pso.dropped_mandatory else '无'}")

    lines.append("")
    return "\n".join(lines)


def generate_summary(results: List[Tuple[TestCase, RunResult, RunResult]]) -> str:
    """生成完整的汇总报告"""
    lines: List[str] = []
    lines.append("=" * 90)
    lines.append("   GA vs PSO — 二维矩形排布对比测试报告")
    lines.append("=" * 90)
    lines.append(f"  测试用例数: {len(results)}")
    lines.append(f"  评估预算:   GA  ≈ 20,000 次适应度评估  (80 pop × 250 gen)")
    lines.append(f"              PSO ≈ 20,000 次适应度评估  (50 swarm × 400 iter)")
    lines.append(f"  角度约束:   10° 整数倍")
    lines.append("=" * 90)
    lines.append("")

    # 逐个案例详情
    for tc, ga, pso in results:
        lines.append(format_table_row(tc.id, tc.name, tc.difficulty, ga, pso))

    # ── 汇总统计 ──
    lines.append("=" * 90)
    lines.append("  汇总统计")
    lines.append("=" * 90)

    # 平均运行时间
    avg_ga_time = sum(r[1].runtime_sec for r in results) / len(results)
    avg_pso_time = sum(r[2].runtime_sec for r in results) / len(results)
    lines.append(f"  平均运行时间:         GA  {avg_ga_time:.3f}s    PSO {avg_pso_time:.3f}s")

    # 平均利用率
    avg_ga_util = sum(r[1].utilization for r in results) / len(results)
    avg_pso_util = sum(r[2].utilization for r in results) / len(results)
    lines.append(f"  平均空间利用率:       GA  {avg_ga_util*100:.2f}%     PSO {avg_pso_util*100:.2f}%")

    # 按难度分组
    for diff in ["easy", "medium", "hard"]:
        subset = [(t, g, p) for t, g, p in results if t.difficulty == diff]
        if not subset:
            continue
        ga_t = sum(r[1].runtime_sec for r in subset) / len(subset)
        pso_t = sum(r[2].runtime_sec for r in subset) / len(subset)
        ga_u = sum(r[1].utilization for r in subset) / len(subset)
        pso_u = sum(r[2].utilization for r in subset) / len(subset)
        lines.append(f"  [{diff:>6s}] 平均时间: GA {ga_t:.3f}s  PSO {pso_t:.3f}s  │  利用率: GA {ga_u*100:.1f}%  PSO {pso_u*100:.1f}%")

    lines.append("")
    lines.append("=" * 90)
    lines.append("  结论")
    lines.append("=" * 90)

    ga_wins_time = sum(1 for _, g, p in results if g.runtime_sec < p.runtime_sec)
    pso_wins_time = len(results) - ga_wins_time
    ga_wins_util = sum(1 for _, g, p in results if g.utilization > p.utilization)
    pso_wins_util = len(results) - ga_wins_util

    lines.append(f"  速度: GA 更快 {ga_wins_time}/{len(results)} 次, PSO 更快 {pso_wins_time}/{len(results)} 次")
    lines.append(f"  质量: GA 更优 {ga_wins_util}/{len(results)} 次, PSO 更优 {pso_wins_util}/{len(results)} 次")
    lines.append("")

    return "\n".join(lines)


# ============================================================================
# 主流程
# ============================================================================
def main(output_dir: str, skip_plots: bool = False) -> None:
    """运行所有测试用例并保存结果。

    Args:
        output_dir: 输出文件夹路径
        skip_plots: 是否跳过图片生成（无 GUI 环境时使用）
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 输出目录: {os.path.abspath(output_dir)}")
    print(f"📋 测试用例数: {len(TEST_CASES)}")
    print()

    all_results: List[Tuple[TestCase, RunResult, RunResult]] = []

    for tc in TEST_CASES:
        case_dir = os.path.join(output_dir, f"case_{tc.id:02d}")
        os.makedirs(case_dir, exist_ok=True)

        print(f"▶ 案例 {tc.id:02d}: {tc.name} [{tc.difficulty}]", end="  ")

        # 运行 GA
        ga_result = run_ga(tc, seed=tc.id * 100 + 42)
        print(f"GA ✓ ({ga_result.runtime_sec:.1f}s)", end="  ")

        # 运行 PSO
        pso_result = run_pso(tc, seed=tc.id * 100 + 99)
        print(f"PSO ✓ ({pso_result.runtime_sec:.1f}s)")

        all_results.append((tc, ga_result, pso_result))

        # ── 保存 JSON 结果 ──
        def result_to_dict(r: RunResult) -> dict:
            return {
                "algorithm": r.algorithm,
                "runtime_sec": round(r.runtime_sec, 4),
                "best_fitness": round(r.best_fitness, 3),
                "polygon_area": round(r.polygon_area, 3),
                "placed_area": round(r.placed_area, 3),
                "utilization": round(r.utilization, 6),
                "placed_count": r.placed_count,
                "dropped_mandatory": r.dropped_mandatory,
                "fitness_history": [round(v, 3) for v in r.fitness_history],
                "placed_rectangles": [
                    {
                        "id": rect.id,
                        "cx": round(rect.cx, 3),
                        "cy": round(rect.cy, 3),
                        "width": rect.width,
                        "height": rect.height,
                        "angle_deg": round(math.degrees(rect.angle), 1),
                        "mandatory": rect.mandatory,
                    }
                    for rect in r.placed_rectangles
                ],
            }

        with open(os.path.join(case_dir, "ga_result.json"), "w", encoding="utf-8") as f:
            json.dump(result_to_dict(ga_result), f, ensure_ascii=False, indent=2)
        with open(os.path.join(case_dir, "pso_result.json"), "w", encoding="utf-8") as f:
            json.dump(result_to_dict(pso_result), f, ensure_ascii=False, indent=2)

        # ── 保存可视化图片 ──
        if not skip_plots:
            try:
                visualize_layout(
                    polygon=tc.polygon,
                    obstacles=tc.obstacles,
                    placed_rectangles=ga_result.placed_rectangles,
                    fitness_history=ga_result.fitness_history,
                    save_path=os.path.join(case_dir, "ga_layout.png"),
                    show=False,
                )
            except Exception as e:
                print(f"    ⚠ GA 图片保存失败: {e}")

            try:
                visualize_layout(
                    polygon=tc.polygon,
                    obstacles=tc.obstacles,
                    placed_rectangles=pso_result.placed_rectangles,
                    fitness_history=pso_result.fitness_history,
                    save_path=os.path.join(case_dir, "pso_layout.png"),
                    show=False,
                )
            except Exception as e:
                print(f"    ⚠ PSO 图片保存失败: {e}")

        # ── 保存案例对比文本 ──
        case_summary = format_table_row(tc.id, tc.name, tc.difficulty, ga_result, pso_result)
        with open(os.path.join(case_dir, "comparison.txt"), "w", encoding="utf-8") as f:
            f.write(case_summary)

    # ── 生成总汇总报告 ──
    summary = generate_summary(all_results)
    summary_path = os.path.join(output_dir, "00_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)

    print()
    print(summary)
    print(f"\n✅ 所有结果已保存到: {os.path.abspath(output_dir)}")
    print(f"   汇总报告: {os.path.join(output_dir, '00_summary.txt')}")


# ============================================================================
# CLI 入口
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GA vs PSO 二维矩形排布对比测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python test_compare.py
  python test_compare.py --output ./my_results
  python test_compare.py --output ./results --no-plots
        """,
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./compare_results",
        help="输出文件夹路径 (默认: ./compare_results)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="跳过图片生成（无头服务器环境推荐）",
    )
    args = parser.parse_args()

    main(output_dir=args.output, skip_plots=args.no_plots)