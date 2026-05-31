"""
pipeline_2D_to_3D.py — 二维排布 → 三维连接 完整管道。

流程:
  1. 运行二维排布求解器（GA/PSO 自动边界），得到设备位置
  2. 输入额外 3D 信息：每个设备的高度、端口定义、连接关系
  3. 将 PlacedRect 提升为 Cuboid3D
  4. 运行正交斯坦纳树 GA 求解连接方案
  5. 输出结果 + 可视化

用法:
  python pipeline_2D_to_3D.py                           # 使用内置 demo 配置
  python pipeline_2D_to_3D.py --config config.json       # 从 JSON 加载 3D 配置
  python pipeline_2D_to_3D.py --output ./results         # 保存结果
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

from config import Equipment3DConfig, PipelineConfig

# ── 导入二维求解器 ──
from auto_boundary_GA import (
    AutoBoundaryGAPacker,
    RectangleSpec as Rect2D,
    SquareObstacle,
    PlacedRect,
    visualize_layout as vis_2d,
)

# ── 导入三维连接求解器 ──
from steiner_connect_3D import (
    Cuboid3D,
    Port,
    ConnectionGroup,
    solve_connections,
    visualize_3d,
    orthogonal_segments,
)


# ============================================================================
# 步骤 1: 二维排布
# ============================================================================
def run_2d_layout(
    rectangles: List[Rect2D],
    obstacles: List[SquareObstacle] | None = None,
    seed: int = 42,
    **ga_kwargs,
) -> Dict[str, object]:
    """运行二维自动边界排布，返回结果字典。"""
    solver = AutoBoundaryGAPacker(
        obstacles=obstacles or [],
        rectangles=rectangles,
        population_size=ga_kwargs.get("population_size", 80),
        generations=ga_kwargs.get("generations", 200),
        mutation_rate=ga_kwargs.get("mutation_rate", 0.18),
        angle_preference_weight=ga_kwargs.get("angle_preference_weight", 50.0),
        boundary_area_weight=ga_kwargs.get("boundary_area_weight", 0.5),
        random_seed=seed,
    )
    return solver.solve()


# ============================================================================
# 步骤 2: 二维 → 三维转换
# ============================================================================

def convert_to_cuboids(
    placed_rectangles: List[PlacedRect],
    equipment_configs: List[Equipment3DConfig],
) -> Tuple[List[Cuboid3D], List[Port]]:
    cuboids: List[Cuboid3D] = []
    ports: List[Port] = []
    config_map = {cfg.rect_id: cfg for cfg in equipment_configs}

    for pr in placed_rectangles:
        ec = config_map.get(pr.id)
        if ec is None:
            print(f"  [警告] 设备 {pr.id} 无 3D 配置，跳过")
            continue

        cb = Cuboid3D(
            id=pr.id,
            cx=pr.cx,
            cy=pr.cy,
            cz=ec.height / 2.0,   # 底面在 z=0，中心在 height/2
            width=pr.width,
            depth=pr.height,       # 2D 的 "height" → 3D 的 "depth"（Y方向）
            height=ec.height,
            angle=pr.angle,
        )
        cuboids.append(cb)

        for p in ec.ports:
            ports.append(Port(
                id=p["id"],
                cuboid_id=pr.id,
                face=p.get("face", "top"),
                u=p.get("u", 0.5),
                v=p.get("v", 0.5),
            ))

    return cuboids, ports


# ============================================================================
# 主流程
# ============================================================================
def run_pipeline(
    rectangles_2d: List[Rect2D],
    pipeline_config: PipelineConfig,
    obstacles: List[SquareObstacle] | None = None,
    output_dir: str | None = None,
    skip_plots: bool = False,
) -> Dict[str, object]:
    """执行完整 2D→3D 管道。

    Returns:
        {
            "layout_2d": {...},       # 二维排布结果
            "cuboids": [...],
            "ports": [...],
            "connections_3d": {...},  # 三维连接结果
        }
    """
    config = pipeline_config
    print("=" * 60)
    print("  管道: 二维排布 → 三维正交连接")
    print("=" * 60)

    # ─── 步骤 1: 二维排布 ───
    print("\n[步骤 1/4] 运行二维自动边界排布...")
    layout_2d = run_2d_layout(
        rectangles=rectangles_2d,
        obstacles=obstacles,
        seed=config.layout_seed,
        **config.ga_2d_kwargs,
    )
    placed = layout_2d["placed_rectangles"]
    print(f"  边界: {layout_2d['boundary_w']:.2f} × {layout_2d['boundary_h']:.2f}")
    print(f"  放置设备: {len(placed)} 个")
    for r in placed:
        print(f"    {r.id}: cx={r.cx:.2f}, cy={r.cy:.2f}, w={r.width:.1f}, h={r.height:.1f}, "
              f"angle={math.degrees(r.angle):.0f}°")

    if not skip_plots and output_dir:
        vis_2d(
            polygon=layout_2d["polygon"],
            obstacles=obstacles or [],
            placed_rectangles=placed,
            fitness_history=layout_2d.get("fitness_history"),
            save_path=os.path.join(output_dir, "01_2d_layout.png"),
            show=False,
        )

    # ─── 步骤 2: 2D → 3D ───
    print("\n[步骤 2/4] 提升为三维长方体...")
    cuboids, ports = convert_to_cuboids(placed, config.equipment)
    print(f"  长方体: {len(cuboids)} 个")
    for cb in cuboids:
        print(f"    {cb.id}: ({cb.cx:.2f}, {cb.cy:.2f}, {cb.cz:.2f}) "
              f"{cb.width:.1f}×{cb.depth:.1f}×{cb.height:.1f}")
    print(f"  端口: {len(ports)} 个")

    cuboid_map = {c.id: c for c in cuboids}
    for p in ports:
        c = cuboid_map[p.cuboid_id]
        wx, wy, wz = c.face_to_world(p.face, p.u, p.v)
        print(f"    {p.id} @ {p.cuboid_id}:{p.face} → ({wx:.2f}, {wy:.2f}, {wz:.2f})")

    # ─── 步骤 3: 构建连接组（过滤未放置设备的端口） ───
    print("\n[步骤 3/4] 构建连接组...")
    valid_port_ids = {p.id for p in ports}
    connection_groups = []
    for conn in config.connections:
        valid_ports = [pid for pid in conn if pid in valid_port_ids]
        missing = [pid for pid in conn if pid not in valid_port_ids]
        if missing:
            print(f"  [警告] 以下端口对应设备未放置，已跳过: {missing}")
        if len(valid_ports) >= 2:
            connection_groups.append(ConnectionGroup(port_ids=valid_ports))
        elif valid_ports:
            print(f"  [警告] 连接组只剩 {len(valid_ports)} 个端口，不足 2 个，跳过")
    print(f"  有效连接组: {len(connection_groups)} 组")
    for cg in connection_groups:
        print(f"    端口: {cg.port_ids}")

    # ─── 步骤 4: 三维连接求解 ───
    print("\n[步骤 4/4] 正交斯坦纳树连接求解...")
    connections_3d = solve_connections(
        cuboids=cuboids,
        ports=ports,
        connection_groups=connection_groups,
        boundary_w=layout_2d["boundary_w"],
        boundary_h=layout_2d["boundary_h"],
        **config.ga_3d_kwargs,
    )
    print(f"  总连接长度 (曼哈顿): {connections_3d['total_length']:.3f}")
    for gi, gr in enumerate(connections_3d["groups"]):
        print(f"  组 {gi+1}: 树长={gr['total_length']:.3f}, "
              f"碰撞={gr['collision_count']}, Steiner点={len(gr['steiner_points'])}")

    # ─── 可视化 ───
    if not skip_plots:
        if output_dir:
            vis_3d_path = os.path.join(output_dir, "02_3d_connections.png")
            visualize_3d(cuboids, ports, connections_3d, save_path=vis_3d_path, show=False)
            print(f"\n图片已保存到: {output_dir}")
        else:
            visualize_3d(cuboids, ports, connections_3d)

    return {
        "layout_2d": layout_2d,
        "cuboids": cuboids,
        "ports": ports,
        "connections_3d": connections_3d,
    }


# ============================================================================
# 内置 Demo 配置
# ============================================================================
def build_demo_config() -> Tuple[List[Rect2D], PipelineConfig]:
    """构建 demo 用的二维矩形列表和管道配置。"""
    rectangles = [
        Rect2D("R1", 5, 4, rotatable=True, mandatory=True),
        Rect2D("R2", 6, 4, rotatable=True, mandatory=True),
        Rect2D("R3", 4, 3, rotatable=True, mandatory=False),
        Rect2D("R4", 5, 5, rotatable=True, mandatory=False),
        Rect2D("R5", 9, 3, rotatable=True, mandatory=False),
    ]

    config = PipelineConfig(
        rectangles=[
            {"id": "R1", "width": 5, "height": 4, "rotatable": True, "mandatory": True},
            {"id": "R2", "width": 6, "height": 4, "rotatable": True, "mandatory": True},
            {"id": "R3", "width": 4, "height": 3, "rotatable": True, "mandatory": False},
            {"id": "R4", "width": 5, "height": 5, "rotatable": True, "mandatory": False},
            {"id": "R5", "width": 9, "height": 3, "rotatable": True, "mandatory": False},
        ],
        equipment=[
            Equipment3DConfig(
                rect_id="R1",
                height=3.0,
                ports=[
                    {"id": "P1_in",  "face": "top",   "u": 0.3, "v": 0.5},
                    {"id": "P1_out", "face": "right",  "u": 0.5, "v": 0.5},
                ],
            ),
            Equipment3DConfig(
                rect_id="R2",
                height=3.0,
                ports=[
                    {"id": "P2_in",  "face": "top",    "u": 0.5, "v": 0.7},
                    {"id": "P2_out", "face": "left",   "u": 0.5, "v": 0.5},
                ],
            ),
            Equipment3DConfig(
                rect_id="R3",
                height=2.5,
                ports=[
                    {"id": "P3_in",  "face": "top",    "u": 0.5, "v": 0.5},
                    {"id": "P3_out", "face": "back",   "u": 0.5, "v": 0.5},
                ],
            ),
            Equipment3DConfig(
                rect_id="R4",
                height=3.5,
                ports=[
                    {"id": "P4_in",  "face": "top",    "u": 0.5, "v": 0.5},
                    {"id": "P4_out", "face": "front",  "u": 0.5, "v": 0.5},
                ],
            ),
            Equipment3DConfig(
                rect_id="R5",
                height=2.8,
                ports=[
                    {"id": "P5_in",  "face": "top",    "u": 0.5, "v": 0.5},
                    {"id": "P5_out", "face": "left",   "u": 0.5, "v": 0.5},
                ],
            ),
        ],
        connections=[
            ["P1_in",  "P2_in",  "P3_in",  "P4_in",  "P5_in"],
            ["P1_out", "P2_out", "P3_out", "P4_out", "P5_out"],
        ],
        layout_seed=42,
        ga_2d_kwargs={
            "population_size": 100,
            "generations": 250,
        },
        ga_3d_kwargs={
            "max_height": 8.0,
            "population_size": 150,
            "generations": 300,
            "collision_penalty": 800.0,
            "random_seed": 142,
        },
    )
    return rectangles, config


# ============================================================================
# JSON 配置加载
# ============================================================================
def load_config_from_json(json_path: str) -> Tuple[List[Rect2D], PipelineConfig]:
    """从 JSON 文件加载管道配置。

    格式见 example_pipeline_config.json。
    """
    config = PipelineConfig.from_json(json_path)
    rectangles = [
        Rect2D(
            id=r["id"],
            width=r["width"],
            height=r["height"],
            rotatable=r.get("rotatable", True),
            mandatory=r.get("mandatory", True),
        )
        for r in config.rectangles
    ]
    return rectangles, config


# ============================================================================
# CLI
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="二维排布 → 三维正交连接 完整管道",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python pipeline_2D_to_3D.py                              # 使用内置 demo
  python pipeline_2D_to_3D.py --output ./my_results         # 保存结果
  python pipeline_2D_to_3D.py --no-plots                    # 不生成图片
  python pipeline_2D_to_3D.py --config my_config.json       # 从 JSON 加载
        """,
    )
    parser.add_argument("--config", type=str, default=None,
                        help="JSON 配置文件路径（不指定则使用内置 demo）")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="输出目录")
    parser.add_argument("--no-plots", action="store_true",
                        help="跳过图片生成")
    args = parser.parse_args()

    # 加载配置
    if args.config:
        print(f"从 JSON 加载配置: {args.config}")
        rectangles, pipeline_config = load_config_from_json(args.config)
    else:
        print("使用内置 demo 配置")
        rectangles, pipeline_config = build_demo_config()

    # 输出目录
    output_dir = args.output
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 运行管道
    result = run_pipeline(
        rectangles_2d=rectangles,
        pipeline_config=pipeline_config,
        obstacles=None,
        output_dir=output_dir,
        skip_plots=args.no_plots,
    )

    # 保存 JSON 结果
    if output_dir:
        def to_serializable(obj):
            if hasattr(obj, "__dataclass_fields__"):
                return {k: to_serializable(v) for k, v in obj.__dict__.items()}
            if isinstance(obj, list):
                return [to_serializable(v) for v in obj]
            if isinstance(obj, tuple):
                return [to_serializable(v) for v in obj]
            if isinstance(obj, dict):
                return {k: to_serializable(v) for k, v in obj.items()}
            return obj

        pipeline_result = {
            "layout_2d": {
                "boundary_w": result["layout_2d"]["boundary_w"],
                "boundary_h": result["layout_2d"]["boundary_h"],
                "polygon_area": result["layout_2d"]["polygon_area"],
                "placed_area": result["layout_2d"]["placed_area"],
                "utilization": result["layout_2d"]["utilization"],
                "placed_rectangles": [
                    {
                        "id": r.id, "cx": round(r.cx, 3), "cy": round(r.cy, 3),
                        "width": r.width, "height": r.height,
                        "angle_deg": round(math.degrees(r.angle), 1),
                    }
                    for r in result["layout_2d"]["placed_rectangles"]
                ],
            },
            "connections_3d": {
                "total_length": round(result["connections_3d"]["total_length"], 3),
                "groups": [
                    {
                        "total_length": round(g["total_length"], 3),
                        "collision_count": g["collision_count"],
                        "steiner_points": [
                            {"x": round(sp[0], 3), "y": round(sp[1], 3), "z": round(sp[2], 3)}
                            for sp in g["steiner_points"]
                        ],
                    }
                    for g in result["connections_3d"]["groups"]
                ],
            },
        }
        json_path = os.path.join(output_dir, "pipeline_result.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(pipeline_result, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {json_path}")

    print("\n管道完成。")
