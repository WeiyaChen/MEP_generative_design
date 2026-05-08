"""
auto_boundary_test.py — GA vs PSO 自动边界排布对比测试。

与 test_layout.py 的区别：
  - 测试用例无需指定边界多边形，由算法自动生成矩形边界
  - 对比指标增加：边界面积、边界尺寸
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

try:
    from auto_boundary_GA import (
        AutoBoundaryGAPacker,
        RectangleSpec,
        SquareObstacle,
        PlacedRect,
        Point,
        visualize_layout,
    )
except ImportError:
    print("错误: 找不到 auto_boundary_GA.py，请确保该文件在当前目录下。")
    sys.exit(1)

try:
    from auto_boundary_PSO import AutoBoundaryPSOPacker
except ImportError:
    print("错误: 找不到 auto_boundary_PSO.py，请确保该文件在当前目录下。")
    sys.exit(1)


# ============================================================================
# 测试用例定义（无需指定边界多边形）
# ============================================================================
@dataclass
class TestCase:
    id: int
    name: str
    difficulty: str
    obstacles: List[SquareObstacle]
    rectangles: List[RectangleSpec]


TEST_CASES: List[TestCase] = [
    # ──────── 简单 ────────
    TestCase(
        id=1,
        name="3个矩形_无障碍",
        difficulty="easy",
        obstacles=[],
        rectangles=[
            RectangleSpec("A1", 5, 4, rotatable=True, mandatory=True),
            RectangleSpec("A2", 6, 4, rotatable=True, mandatory=False),
            RectangleSpec("A3", 4, 3, rotatable=True, mandatory=False),
        ],
    ),
    TestCase(
        id=2,
        name="6个矩形_无障碍",
        difficulty="easy",
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
        name="8个矩形_无障碍",
        difficulty="easy",
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
        name="9个矩形_1障碍物",
        difficulty="medium",
        obstacles=[
            SquareObstacle(x=12, y=10, size=4),
        ],
        rectangles=[
            RectangleSpec("D1", 8, 5, rotatable=True, mandatory=True),
            RectangleSpec("D2", 7, 6, rotatable=True, mandatory=True),
            RectangleSpec("D3", 6, 4, rotatable=True, mandatory=False),
            RectangleSpec("D4", 5, 5, rotatable=True, mandatory=False),
            RectangleSpec("D5", 9, 3, rotatable=True, mandatory=False),
            RectangleSpec("D6", 4, 7, rotatable=True, mandatory=False),
            RectangleSpec("D7", 8, 3, rotatable=True, mandatory=False),
            RectangleSpec("D8", 6, 5, rotatable=True, mandatory=False),
            RectangleSpec("D9", 4, 6, rotatable=True, mandatory=False),
        ],
    ),
    TestCase(
        id=5,
        name="9个矩形_2障碍物",
        difficulty="medium",
        obstacles=[
            SquareObstacle(x=14, y=10, size=4),
            SquareObstacle(x=8, y=16, size=3.5),
        ],
        rectangles=[
            RectangleSpec("E1", 8, 6, rotatable=True, mandatory=True),
            RectangleSpec("E2", 9, 5, rotatable=True, mandatory=True),
            RectangleSpec("E3", 7, 7, rotatable=True, mandatory=False),
            RectangleSpec("E4", 6, 4, rotatable=True, mandatory=False),
            RectangleSpec("E5", 10, 3, rotatable=True, mandatory=False),
            RectangleSpec("E6", 5, 8, rotatable=True, mandatory=False),
            RectangleSpec("E7", 4, 5, rotatable=True, mandatory=False),
            RectangleSpec("E8", 11, 2.5, rotatable=True, mandatory=False),
            RectangleSpec("E9", 3, 9, rotatable=True, mandatory=False),
        ],
    ),
    TestCase(
        id=6,
        name="12个矩形_3障碍物",
        difficulty="medium",
        obstacles=[
            SquareObstacle(x=6, y=8, size=3),
            SquareObstacle(x=22, y=14, size=4),
            SquareObstacle(x=14, y=22, size=3.5),
        ],
        rectangles=[
            RectangleSpec("F1", 7, 5, rotatable=True, mandatory=True),
            RectangleSpec("F2", 9, 4, rotatable=True, mandatory=True),
            RectangleSpec("F3", 6, 6, rotatable=True, mandatory=True),
            RectangleSpec("F4", 8, 5, rotatable=True, mandatory=False),
            RectangleSpec("F5", 5, 7, rotatable=True, mandatory=False),
            RectangleSpec("F6", 10, 3, rotatable=True, mandatory=False),
            RectangleSpec("F7", 4, 6, rotatable=True, mandatory=False),
            RectangleSpec("F8", 7, 3, rotatable=True, mandatory=False),
            RectangleSpec("F9", 6, 8, rotatable=True, mandatory=False),
            RectangleSpec("F10", 9, 2.5, rotatable=True, mandatory=False),
            RectangleSpec("F11", 3, 9, rotatable=True, mandatory=False),
            RectangleSpec("F12", 5, 4, rotatable=True, mandatory=False),
        ],
    ),

    # ──────── 困难 ────────
    TestCase(
        id=7,
        name="12个矩形_4障碍物",
        difficulty="hard",
        obstacles=[
            SquareObstacle(x=6, y=6, size=3),
            SquareObstacle(x=18, y=8, size=4),
            SquareObstacle(x=10, y=18, size=3.5),
            SquareObstacle(x=24, y=20, size=3),
        ],
        rectangles=[
            RectangleSpec("G1", 7, 5, rotatable=True, mandatory=True),
            RectangleSpec("G2", 9, 4, rotatable=True, mandatory=True),
            RectangleSpec("G3", 6, 6, rotatable=True, mandatory=True),
            RectangleSpec("G4", 8, 5, rotatable=True, mandatory=False),
            RectangleSpec("G5", 5, 7, rotatable=True, mandatory=False),
            RectangleSpec("G6", 10, 3, rotatable=True, mandatory=False),
            RectangleSpec("G7", 4, 6, rotatable=True, mandatory=False),
            RectangleSpec("G8", 7, 3, rotatable=True, mandatory=False),
            RectangleSpec("G9", 6, 8, rotatable=True, mandatory=False),
            RectangleSpec("G10", 9, 2.5, rotatable=True, mandatory=False),
            RectangleSpec("G11", 3, 9, rotatable=True, mandatory=False),
            RectangleSpec("G12", 5, 4, rotatable=True, mandatory=False),
        ],
    ),
    TestCase(
        id=8,
        name="15个矩形_3障碍物",
        difficulty="hard",
        obstacles=[
            SquareObstacle(x=8, y=8, size=3.5),
            SquareObstacle(x=20, y=12, size=4),
            SquareObstacle(x=12, y=22, size=3),
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
            RectangleSpec("H13", 8, 3.5, rotatable=True, mandatory=False),
            RectangleSpec("H14", 4, 7, rotatable=True, mandatory=False),
            RectangleSpec("H15", 6, 4, rotatable=True, mandatory=False),
        ],
    ),
    TestCase(
        id=9,
        name="15个矩形_5障碍物",
        difficulty="hard",
        obstacles=[
            SquareObstacle(x=6, y=8, size=3),
            SquareObstacle(x=18, y=6, size=3.5),
            SquareObstacle(x=26, y=16, size=3),
            SquareObstacle(x=8, y=20, size=4),
            SquareObstacle(x=22, y=24, size=2.5),
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
    algorithm: str
    runtime_sec: float
    best_fitness: float
    boundary_w: float
    boundary_h: float
    polygon_area: float
    placed_area: float
    utilization: float
    placed_count: int
    dropped_mandatory: List[str]
    fitness_history: List[float]
    placed_rectangles: List[PlacedRect]


def run_ga(tc: TestCase, seed: int = 42) -> RunResult:
    solver = AutoBoundaryGAPacker(
        obstacles=tc.obstacles,
        rectangles=tc.rectangles,
        population_size=80,
        generations=250,
        crossover_rate=0.85,
        mutation_rate=0.18,
        elite_count=5,
        tournament_size=3,
        angle_preference_weight=22.0,
        boundary_area_weight=0.5,
        random_seed=seed,
    )

    t0 = time.perf_counter()
    result = solver.solve()
    t1 = time.perf_counter()

    return RunResult(
        algorithm="GA",
        runtime_sec=t1 - t0,
        best_fitness=float(result["best_fitness"]),
        boundary_w=float(result["boundary_w"]),
        boundary_h=float(result["boundary_h"]),
        polygon_area=float(result["polygon_area"]),
        placed_area=float(result["placed_area"]),
        utilization=float(result["utilization"]),
        placed_count=len(result["placed_rectangles"]),
        dropped_mandatory=list(result["dropped_mandatory"]),
        fitness_history=list(result["fitness_history"]),
        placed_rectangles=list(result["placed_rectangles"]),
    )


def run_pso(tc: TestCase, seed: int = 42) -> RunResult:
    solver = AutoBoundaryPSOPacker(
        obstacles=tc.obstacles,
        rectangles=tc.rectangles,
        swarm_size=50,
        iterations=400,
        w_start=0.9,
        w_end=0.4,
        c1=1.8,
        c2=1.8,
        angle_preference_weight=22.0,
        boundary_area_weight=0.5,
        random_seed=seed,
    )

    t0 = time.perf_counter()
    result = solver.solve()
    t1 = time.perf_counter()

    return RunResult(
        algorithm="PSO",
        runtime_sec=t1 - t0,
        best_fitness=float(result["best_fitness"]),
        boundary_w=float(result["boundary_w"]),
        boundary_h=float(result["boundary_h"]),
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
    faster = "GA" if ga.runtime_sec < pso.runtime_sec else "PSO"
    better_util = "GA" if ga.utilization >= pso.utilization else "PSO"
    smaller_boundary = "GA" if ga.polygon_area <= pso.polygon_area else "PSO"
    time_ratio = ga.runtime_sec / max(pso.runtime_sec, 1e-6)

    lines = [
        f"{'─'*92}",
        f" 案例 {case_id:02d} │ {name}  [{diff}]",
        f"{'─'*92}",
        f" 指标                  │ {'GA':>20s} │ {'PSO':>20s} │ 对比",
        f"{'─'*92}",
        f" 运行时间 (秒)         │ {ga.runtime_sec:>20.3f} │ {pso.runtime_sec:>20.3f} │ 更快: {faster}  ({time_ratio:.2f}x)",
        f" 最佳适应度            │ {ga.best_fitness:>20.2f} │ {pso.best_fitness:>20.2f} │ {'GA' if ga.best_fitness>=pso.best_fitness else 'PSO'} 更优",
        f" 边界尺寸 (W×H)        │ {ga.boundary_w:>9.2f} ×{ga.boundary_h:<9.2f} │ {pso.boundary_w:>9.2f} ×{pso.boundary_h:<9.2f} │",
        f" 边界面积              │ {ga.polygon_area:>20.2f} │ {pso.polygon_area:>20.2f} │ {smaller_boundary} 更小",
        f" 空间利用率 (%)        │ {ga.utilization*100:>19.2f}% │ {pso.utilization*100:>19.2f}% │ {better_util} 更高",
        f" 已放置矩形数          │ {ga.placed_count:>20d} │ {pso.placed_count:>20d} │",
        f" 已放置面积            │ {ga.placed_area:>20.2f} │ {pso.placed_area:>20.2f} │",
    ]

    if ga.dropped_mandatory or pso.dropped_mandatory:
        lines.append(f" ⚠ GA  丢弃的必要矩形: {ga.dropped_mandatory if ga.dropped_mandatory else '无'}")
        lines.append(f" ⚠ PSO 丢弃的必要矩形: {pso.dropped_mandatory if pso.dropped_mandatory else '无'}")

    lines.append("")
    return "\n".join(lines)


def generate_summary(results: List[Tuple[TestCase, RunResult, RunResult]]) -> str:
    lines: List[str] = []
    lines.append("=" * 92)
    lines.append("   GA vs PSO — 自动边界二维矩形排布对比测试报告")
    lines.append("=" * 92)
    lines.append(f"  测试用例数: {len(results)}")
    lines.append(f"  评估预算:   GA  ≈ 20,000 次适应度评估  (80 pop × 250 gen)")
    lines.append(f"              PSO ≈ 20,000 次适应度评估  (50 swarm × 400 iter)")
    lines.append(f"  角度约束:   10° 整数倍")
    lines.append(f"  边界类型:   自动优化矩形边界 (W × H)")
    lines.append("=" * 92)
    lines.append("")

    for tc, ga, pso in results:
        lines.append(format_table_row(tc.id, tc.name, tc.difficulty, ga, pso))

    lines.append("=" * 92)
    lines.append("  汇总统计")
    lines.append("=" * 92)

    avg_ga_time = sum(r[1].runtime_sec for r in results) / len(results)
    avg_pso_time = sum(r[2].runtime_sec for r in results) / len(results)
    lines.append(f"  平均运行时间:         GA  {avg_ga_time:.3f}s    PSO {avg_pso_time:.3f}s")

    avg_ga_area = sum(r[1].polygon_area for r in results) / len(results)
    avg_pso_area = sum(r[2].polygon_area for r in results) / len(results)
    lines.append(f"  平均边界面积:         GA  {avg_ga_area:.2f}      PSO {avg_pso_area:.2f}")

    avg_ga_util = sum(r[1].utilization for r in results) / len(results)
    avg_pso_util = sum(r[2].utilization for r in results) / len(results)
    lines.append(f"  平均空间利用率:       GA  {avg_ga_util*100:.2f}%     PSO {avg_pso_util*100:.2f}%")

    for diff in ["easy", "medium", "hard"]:
        subset = [(t, g, p) for t, g, p in results if t.difficulty == diff]
        if not subset:
            continue
        ga_t = sum(r[1].runtime_sec for r in subset) / len(subset)
        pso_t = sum(r[2].runtime_sec for r in subset) / len(subset)
        ga_a = sum(r[1].polygon_area for r in subset) / len(subset)
        pso_a = sum(r[2].polygon_area for r in subset) / len(subset)
        ga_u = sum(r[1].utilization for r in subset) / len(subset)
        pso_u = sum(r[2].utilization for r in subset) / len(subset)
        lines.append(
            f"  [{diff:>6s}] 时间: GA {ga_t:.3f}s PSO {pso_t:.3f}s  │  "
            f"边界: GA {ga_a:.1f} PSO {pso_a:.1f}  │  "
            f"利用: GA {ga_u*100:.1f}% PSO {pso_u*100:.1f}%"
        )

    lines.append("")
    lines.append("=" * 92)
    lines.append("  结论")
    lines.append("=" * 92)

    ga_wins_time = sum(1 for _, g, p in results if g.runtime_sec < p.runtime_sec)
    pso_wins_time = len(results) - ga_wins_time
    ga_wins_util = sum(1 for _, g, p in results if g.utilization > p.utilization)
    pso_wins_util = len(results) - ga_wins_util
    ga_wins_area = sum(1 for _, g, p in results if g.polygon_area < p.polygon_area)
    pso_wins_area = len(results) - ga_wins_area

    lines.append(f"  速度:      GA 更快 {ga_wins_time}/{len(results)} 次, PSO 更快 {pso_wins_time}/{len(results)} 次")
    lines.append(f"  利用率:    GA 更优 {ga_wins_util}/{len(results)} 次, PSO 更优 {pso_wins_util}/{len(results)} 次")
    lines.append(f"  边界面积:  GA 更小 {ga_wins_area}/{len(results)} 次, PSO 更小 {pso_wins_area}/{len(results)} 次")
    lines.append("")

    return "\n".join(lines)


# ============================================================================
# 主流程
# ============================================================================
def main(output_dir: str, skip_plots: bool = False) -> None:
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {os.path.abspath(output_dir)}")
    print(f"测试用例数: {len(TEST_CASES)}")
    print()

    all_results: List[Tuple[TestCase, RunResult, RunResult]] = []

    for tc in TEST_CASES:
        case_dir = os.path.join(output_dir, f"case_{tc.id:02d}")
        os.makedirs(case_dir, exist_ok=True)

        print(f"> 案例 {tc.id:02d}: {tc.name} [{tc.difficulty}]", end="  ")

        ga_result = run_ga(tc, seed=tc.id * 100 + 42)
        print(f"GA ok ({ga_result.runtime_sec:.1f}s)", end="  ")

        pso_result = run_pso(tc, seed=tc.id * 100 + 99)
        print(f"PSO ok ({pso_result.runtime_sec:.1f}s)")

        all_results.append((tc, ga_result, pso_result))

        # ── 保存 JSON ──
        def result_to_dict(r: RunResult) -> dict:
            return {
                "algorithm": r.algorithm,
                "runtime_sec": round(r.runtime_sec, 4),
                "best_fitness": round(r.best_fitness, 3),
                "boundary_w": round(r.boundary_w, 3),
                "boundary_h": round(r.boundary_h, 3),
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

        # ── 可视化 ──
        if not skip_plots:
            for algo, r in [("GA", ga_result), ("PSO", pso_result)]:
                try:
                    bw, bh = r.boundary_w, r.boundary_h
                    polygon = [(0.0, 0.0), (bw, 0.0), (bw, bh), (0.0, bh)]
                    visualize_layout(
                        polygon=polygon,
                        obstacles=tc.obstacles,
                        placed_rectangles=r.placed_rectangles,
                        fitness_history=r.fitness_history,
                        save_path=os.path.join(case_dir, f"{algo.lower()}_layout.png"),
                        show=False,
                    )
                except Exception as e:
                    print(f"    ! {algo} 图片保存失败: {e}")

        # ── 案例对比文本 ──
        case_summary = format_table_row(tc.id, tc.name, tc.difficulty, ga_result, pso_result)
        with open(os.path.join(case_dir, "comparison.txt"), "w", encoding="utf-8") as f:
            f.write(case_summary)

    # ── 汇总报告 ──
    summary = generate_summary(all_results)
    summary_path = os.path.join(output_dir, "00_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)

    print()
    print(summary)
    print(f"\n所有结果已保存到: {os.path.abspath(output_dir)}")
    print(f"   汇总报告: {os.path.join(output_dir, '00_summary.txt')}")


# ============================================================================
# CLI
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GA vs PSO 自动边界矩形排布对比测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python auto_boundary_test.py
  python auto_boundary_test.py --output ./auto_results
  python auto_boundary_test.py --output ./auto_results --no-plots
        """,
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./auto_boundary_results",
        help="输出文件夹路径 (默认: ./auto_boundary_results)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="跳过图片生成（无头服务器环境推荐）",
    )
    args = parser.parse_args()

    main(output_dir=args.output, skip_plots=args.no_plots)
