"""
steiner_connect_3D.py — 基于遗传算法的三维斯坦纳树连接优化。

将二维排布结果提升为三维长方体，定义设备端口（入口/出口）及连接关系，
采用斯坦纳树建模，在三维碰撞约束下优化连接线总长度最小。

输入：二维排布结果 + 长方体高度 + 端口定义 + 连接组
输出：斯坦纳树连接方案（Steiner点位置 + 树边集合）
"""
from __future__ import annotations
import argparse
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

EPS = 1e-9


# ============================================================================
# 数据类型
# ============================================================================
@dataclass(frozen=True)
class Cuboid3D:
    """三维长方体（由二维 PlacedRect 提升得到）。

    angle 为绕 Z 轴旋转角（弧度），Z 方向无旋转。
    """
    id: str
    cx: float
    cy: float
    cz: float          # 底面中心 Z 坐标
    width: float       # X 方向尺寸
    depth: float       # Y 方向尺寸
    height: float      # Z 方向尺寸
    angle: float       # 绕 Z 轴旋转

    def face_center(self, face: str) -> Tuple[float, float, float]:
        """返回指定面中心的世界坐标。"""
        hw, hd, hh = self.width / 2, self.depth / 2, self.height / 2
        ca, sa = math.cos(self.angle), math.sin(self.angle)
        # 局部坐标系中的偏移
        local = {
            "front":  (0.0,  hd, 0.0),   # +Y face
            "back":   (0.0, -hd, 0.0),   # -Y face
            "right":  ( hw, 0.0, 0.0),   # +X face
            "left":   (-hw, 0.0, 0.0),   # -X face
            "top":    (0.0, 0.0,  hh),   # +Z face
            "bottom": (0.0, 0.0, -hh),   # -Z face
        }
        lx, ly, lz = local[face]
        # 绕 Z 旋转
        wx = self.cx + lx * ca - ly * sa
        wy = self.cy + lx * sa + ly * ca
        wz = self.cz + lz
        return (wx, wy, wz)

    def face_to_world(self, face: str, u: float, v: float) -> Tuple[float, float, float]:
        """将面上的局部坐标 (u, v) ∈ [0,1]×[0,1] 转为世界坐标。"""
        hw, hd, hh = self.width / 2, self.depth / 2, self.height / 2
        ca, sa = math.cos(self.angle), math.sin(self.angle)

        PORT_OFFSET = 0.12  # 端口外偏移，避免碰撞误判

        # local (u,v) → local 3D offset（含外偏移）
        if face == "front":    # +Y, u→X, v→Z
            lx = (u - 0.5) * self.width
            ly = hd + PORT_OFFSET
            lz = (v - 0.5) * self.height
        elif face == "back":   # -Y, u→X, v→Z
            lx = (u - 0.5) * self.width
            ly = -hd - PORT_OFFSET
            lz = (v - 0.5) * self.height
        elif face == "right":  # +X, u→Y, v→Z
            lx = hw + PORT_OFFSET
            ly = (u - 0.5) * self.depth
            lz = (v - 0.5) * self.height
        elif face == "left":   # -X, u→Y, v→Z
            lx = -hw - PORT_OFFSET
            ly = (u - 0.5) * self.depth
            lz = (v - 0.5) * self.height
        elif face == "top":    # +Z, u→X, v→Y
            lx = (u - 0.5) * self.width
            ly = (v - 0.5) * self.depth
            lz = hh + PORT_OFFSET
        elif face == "bottom": # -Z, u→X, v→Y
            lx = (u - 0.5) * self.width
            ly = (v - 0.5) * self.depth
            lz = -hh - PORT_OFFSET
        else:
            raise ValueError(f"Unknown face: {face}")

        wx = self.cx + lx * ca - ly * sa
        wy = self.cy + lx * sa + ly * ca
        wz = self.cz + lz
        return (wx, wy, wz)


@dataclass(frozen=True)
class Port:
    """设备上的连接端口。"""
    id: str
    cuboid_id: str
    face: str      # "front"|"back"|"left"|"right"|"top"|"bottom"
    u: float = 0.5  # 面上的水平位置 [0,1]
    v: float = 0.5  # 面上的垂直位置 [0,1]


@dataclass
class ConnectionGroup:
    """一组需要互联的端口集合（形成一个斯坦纳树网络）。"""
    port_ids: List[str]


# ============================================================================
# 3D 几何工具
# ============================================================================
Point3D = Tuple[float, float, float]


def point_distance(a: Point3D, b: Point3D) -> float:
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)


def segment_obb_intersect(
    p0: Point3D,
    p1: Point3D,
    cuboid: Cuboid3D,
) -> bool:
    """检测线段 p0-p1 是否与长方体（OBB，仅绕 Z 旋转）相交。

    方法：将线段变换到长方体局部坐标系，然后做 AABB 射线检测。
    """
    ca = math.cos(-cuboid.angle)
    sa = math.sin(-cuboid.angle)

    # 平移
    tx0 = p0[0] - cuboid.cx
    ty0 = p0[1] - cuboid.cy
    tz0 = p0[2] - cuboid.cz
    tx1 = p1[0] - cuboid.cx
    ty1 = p1[1] - cuboid.cy
    tz1 = p1[2] - cuboid.cz

    # 绕 Z 旋转到局部坐标系
    lx0 = tx0 * ca - ty0 * sa
    ly0 = tx0 * sa + ty0 * ca
    lz0 = tz0
    lx1 = tx1 * ca - ty1 * sa
    ly1 = tx1 * sa + ty1 * ca
    lz1 = tz1

    hw = cuboid.width / 2 + 0.05
    hd = cuboid.depth / 2 + 0.05
    hh = cuboid.height / 2 + 0.05

    return _segment_aabb_intersect(
        (lx0, ly0, lz0), (lx1, ly1, lz1),
        (-hw, -hd, -hh), (hw, hd, hh),
    )


def _segment_aabb_intersect(
    p0: Tuple[float, float, float],
    p1: Tuple[float, float, float],
    box_min: Tuple[float, float, float],
    box_max: Tuple[float, float, float],
) -> bool:
    """线段与轴对齐包围盒相交检测（slab 方法）。"""
    d = (p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])

    tmin, tmax = 0.0, 1.0

    for i in range(3):
        if abs(d[i]) < EPS:
            if p0[i] < box_min[i] or p0[i] > box_max[i]:
                return False
        else:
            inv_d = 1.0 / d[i]
            t1 = (box_min[i] - p0[i]) * inv_d
            t2 = (box_max[i] - p0[i]) * inv_d
            if t1 > t2:
                t1, t2 = t2, t1
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
            if tmin > tmax:
                return False

    return True


# ============================================================================
# Prim 算法求最小生成树
# ============================================================================
def compute_mst(nodes: List[Point3D]) -> List[Tuple[int, int, float]]:
    """计算欧几里得距离下的 MST，返回边列表 (i, j, distance)。"""
    n = len(nodes)
    if n <= 1:
        return []

    visited = [False] * n
    min_dist = [float('inf')] * n
    parent = [-1] * n
    min_dist[0] = 0.0

    for _ in range(n):
        u = -1
        best_d = float('inf')
        for i in range(n):
            if not visited[i] and min_dist[i] < best_d:
                best_d = min_dist[i]
                u = i
        if u == -1:
            break

        visited[u] = True

        for v in range(n):
            if not visited[v]:
                d = point_distance(nodes[u], nodes[v])
                if d < min_dist[v]:
                    min_dist[v] = d
                    parent[v] = u

    edges: List[Tuple[int, int, float]] = []
    for v in range(1, n):
        if parent[v] >= 0:
            edges.append((parent[v], v, point_distance(nodes[parent[v]], nodes[v])))

    return edges


# ============================================================================
# 斯坦纳树 GA 求解器
# ============================================================================
@dataclass
class SteinerGene:
    x: float
    y: float
    z: float
    active: bool


class SteinerTreeGA:
    """遗传算法求解带碰撞约束的三维斯坦纳树。

    决策变量：最多 max_steiner 个 Steiner 点，每个点 (x,y,z,active)。
    适应度 = -(总树长 + 碰撞惩罚)
    """

    def __init__(
        self,
        terminals: List[Point3D],
        cuboids: Sequence[Cuboid3D],
        boundary_w: float,
        boundary_h: float,
        max_height: float = 10.0,
        max_steiner: int | None = None,
        population_size: int = 150,
        generations: int = 300,
        crossover_rate: float = 0.80,
        mutation_rate: float = 0.18,
        elite_count: int = 5,
        tournament_size: int = 3,
        collision_penalty: float = 500.0,
        random_seed: int | None = 7,
    ) -> None:
        if len(terminals) < 2:
            raise ValueError("At least 2 terminals required")

        self.terminals = list(terminals)
        self.cuboids = list(cuboids)
        self.boundary_w = boundary_w
        self.boundary_h = boundary_h
        self.max_height = max_height
        self.max_steiner = max_steiner if max_steiner is not None else max(0, len(terminals))

        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_count = elite_count
        self.tournament_size = tournament_size
        self.collision_penalty = collision_penalty

        self.rng = random.Random(random_seed)

        # terminal 包围盒扩大一些作为 Steiner 点的搜索范围
        xs = [t[0] for t in terminals]
        ys = [t[1] for t in terminals]
        zs = [t[2] for t in terminals]
        self._min_x = max(min(xs) - 3.0, 0.0)
        self._max_x = min(max(xs) + 3.0, boundary_w)
        self._min_y = max(min(ys) - 3.0, 0.0)
        self._max_y = min(max(ys) + 3.0, boundary_h)
        self._min_z = 0.0
        self._max_z = max(zs) + 3.0

    def _random_gene(self) -> SteinerGene:
        return SteinerGene(
            x=self.rng.uniform(self._min_x, self._max_x),
            y=self.rng.uniform(self._min_y, self._max_y),
            z=self.rng.uniform(self._min_z, self._max_z),
            active=self.rng.random() < 0.5,
        )

    def _init_population(self) -> List[List[SteinerGene]]:
        pop = []
        for _ in range(self.population_size):
            chromosome = [self._random_gene() for _ in range(self.max_steiner)]
            pop.append(chromosome)
        return pop

    def _get_all_nodes(self, chromosome: Sequence[SteinerGene]) -> List[Point3D]:
        """返回 terminals + 活跃 Steiner 点的全部节点列表。"""
        nodes = list(self.terminals)
        for gene in chromosome:
            if gene.active:
                nodes.append((gene.x, gene.y, gene.z))
        return nodes

    def _edge_collides(self, a: Point3D, b: Point3D) -> bool:
        """检查线段 ab 是否与任何长方体碰撞。"""
        for cuboid in self.cuboids:
            if segment_obb_intersect(a, b, cuboid):
                return True
        return False

    def _fitness(self, chromosome: Sequence[SteinerGene]) -> float:
        nodes = self._get_all_nodes(chromosome)
        if len(nodes) < 2:
            return -1e9

        edges = compute_mst(nodes)
        total_length = sum(e[2] for e in edges)
        penalty = 0.0

        for i, j, _ in edges:
            if self._edge_collides(nodes[i], nodes[j]):
                penalty += self.collision_penalty

        return -(total_length + penalty)

    def _tournament_select(
        self, population: Sequence[Sequence[SteinerGene]], fitnesses: Sequence[float]
    ) -> List[SteinerGene]:
        idxs = [self.rng.randrange(len(population)) for _ in range(self.tournament_size)]
        best_idx = max(idxs, key=lambda i: fitnesses[i])
        return [SteinerGene(g.x, g.y, g.z, g.active) for g in population[best_idx]]

    def _crossover(
        self, p1: Sequence[SteinerGene], p2: Sequence[SteinerGene]
    ) -> Tuple[List[SteinerGene], List[SteinerGene]]:
        if self.rng.random() > self.crossover_rate:
            return (
                [SteinerGene(g.x, g.y, g.z, g.active) for g in p1],
                [SteinerGene(g.x, g.y, g.z, g.active) for g in p2],
            )
        c1, c2 = [], []
        for g1, g2 in zip(p1, p2):
            if self.rng.random() < 0.5:
                c1.append(SteinerGene(g1.x, g1.y, g1.z, g1.active))
                c2.append(SteinerGene(g2.x, g2.y, g2.z, g2.active))
            else:
                c1.append(SteinerGene(g2.x, g2.y, g2.z, g2.active))
                c2.append(SteinerGene(g1.x, g1.y, g1.z, g1.active))
        return c1, c2

    def _mutate(self, chromosome: Sequence[SteinerGene]) -> List[SteinerGene]:
        mutated = []
        span_x = (self._max_x - self._min_x) * 0.12
        span_y = (self._max_y - self._min_y) * 0.12
        span_z = (self._max_z - self._min_z) * 0.12

        for gene in chromosome:
            x, y, z, active = gene.x, gene.y, gene.z, gene.active

            if self.rng.random() < self.mutation_rate:
                x = x + self.rng.gauss(0.0, span_x)
                x = max(self._min_x, min(self._max_x, x))
            if self.rng.random() < self.mutation_rate:
                y = y + self.rng.gauss(0.0, span_y)
                y = max(self._min_y, min(self._max_y, y))
            if self.rng.random() < self.mutation_rate:
                z = z + self.rng.gauss(0.0, span_z)
                z = max(self._min_z, min(self._max_z, z))

            if self.rng.random() < self.mutation_rate * 0.5:
                active = not active

            mutated.append(SteinerGene(x=x, y=y, z=z, active=active))

        return mutated

    def solve(self) -> Dict[str, object]:
        population = self._init_population()

        best_chromosome = population[0]
        best_fitness = -math.inf
        fitness_history: List[float] = []

        for _ in range(self.generations):
            fitnesses = [self._fitness(ch) for ch in population]
            idx = max(range(len(population)), key=lambda i: fitnesses[i])
            if fitnesses[idx] > best_fitness:
                best_fitness = fitnesses[idx]
                best_chromosome = [
                    SteinerGene(g.x, g.y, g.z, g.active) for g in population[idx]
                ]
            fitness_history.append(max(fitnesses))

            ranked = sorted(
                range(len(population)), key=lambda i: fitnesses[i], reverse=True
            )
            next_pop = [
                [SteinerGene(g.x, g.y, g.z, g.active) for g in population[i]]
                for i in ranked[: self.elite_count]
            ]

            while len(next_pop) < self.population_size:
                p1 = self._tournament_select(population, fitnesses)
                p2 = self._tournament_select(population, fitnesses)
                c1, c2 = self._crossover(p1, p2)
                c1 = self._mutate(c1)
                c2 = self._mutate(c2)
                next_pop.append(c1)
                if len(next_pop) < self.population_size:
                    next_pop.append(c2)

            population = next_pop

        # 构建最终结果
        nodes = self._get_all_nodes(best_chromosome)
        edges = compute_mst(nodes)
        total_length = sum(e[2] for e in edges)
        collision_count = 0
        for i, j, _ in edges:
            if self._edge_collides(nodes[i], nodes[j]):
                collision_count += 1

        steiner_points = [
            (g.x, g.y, g.z)
            for g in best_chromosome if g.active
        ]

        edge_list = [(nodes[i], nodes[j], d) for i, j, d in edges]

        return {
            "best_fitness": best_fitness,
            "total_length": total_length,
            "collision_count": collision_count,
            "nodes": nodes,
            "terminals": self.terminals,
            "steiner_points": steiner_points,
            "edges": edge_list,
            "fitness_history": fitness_history,
        }


# ============================================================================
# 完整求解流程
# ============================================================================
def solve_connections(
    cuboids: List[Cuboid3D],
    ports: List[Port],
    connection_groups: List[ConnectionGroup],
    boundary_w: float,
    boundary_h: float,
    max_height: float = 10.0,
    max_steiner_per_group: int | None = None,
    **ga_kwargs,
) -> Dict[str, object]:
    """求解多组连接的斯坦纳树。

    Returns:
        groups: 每组的结果列表
        total_length: 全部连接总长
    """
    port_map = {p.id: p for p in ports}
    cuboid_map = {c.id: c for c in cuboids}

    def port_to_world(port_id: str) -> Point3D:
        p = port_map[port_id]
        c = cuboid_map[p.cuboid_id]
        return c.face_to_world(p.face, p.u, p.v)

    all_results = []
    grand_total = 0.0

    for cg in connection_groups:
        terminals = [port_to_world(pid) for pid in cg.port_ids]

        solver = SteinerTreeGA(
            terminals=terminals,
            cuboids=cuboids,
            boundary_w=boundary_w,
            boundary_h=boundary_h,
            max_height=max_height,
            max_steiner=max_steiner_per_group,
            **ga_kwargs,
        )
        result = solver.solve()
        all_results.append(result)
        grand_total += result["total_length"]

    return {
        "groups": all_results,
        "total_length": grand_total,
    }


# ============================================================================
# 3D 可视化
# ============================================================================
def visualize_3d(
    cuboids: List[Cuboid3D],
    ports: List[Port],
    result: Dict[str, object],
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """绘制三维场景：长方体 + 端口 + 斯坦纳树连接线。"""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError:
        print("需要安装 matplotlib 进行 3D 可视化。")
        return

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制长方体
    face_colors = [
        "#A8DADC", "#F4A261", "#2A9D8F", "#E76F51", "#8AB17D",
        "#F6BD60", "#90CAF9", "#BDB2FF", "#FFD6A5", "#84A59D",
    ]
    for idx, cb in enumerate(cuboids):
        color = face_colors[idx % len(face_colors)]
        _draw_cuboid(ax, cb, color=color, alpha=0.4)

    # 绘制端口
    port_map = {p.id: p for p in ports}
    cuboid_map = {c.id: c for c in cuboids}

    for p in ports:
        c = cuboid_map[p.cuboid_id]
        wx, wy, wz = c.face_to_world(p.face, p.u, p.v)
        ax.scatter(wx, wy, wz, c='red', s=60, marker='o', edgecolors='darkred', linewidth=1.5)
        ax.text(wx, wy, wz + 0.3, p.id, fontsize=9, ha='center', va='bottom', color='darkred')

    # 绘制连接线（斯坦纳树）
    edge_colors = [
        "#1D3557", "#E63946", "#457B9D", "#6D6875",
        "#2A9D8F", "#D62828", "#3A5A40",
    ]
    for gi, group_result in enumerate(result["groups"]):
        ec = edge_colors[gi % len(edge_colors)]
        edges = group_result["edges"]
        nodes = group_result["nodes"]
        terminals = group_result["terminals"]
        steiner = group_result.get("steiner_points", [])

        for a, b, _ in edges:
            ax.plot(
                [a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                color=ec, linewidth=2.5, alpha=0.85,
            )

        # 绘制 Steiner 点
        for sp in steiner:
            ax.scatter(sp[0], sp[1], sp[2], c='blue', s=50, marker='^',
                       edgecolors='darkblue', linewidth=1.0, alpha=0.9)

    # 标注长方体标签
    for cb in cuboids:
        ax.text(cb.cx, cb.cy, cb.cz + cb.height / 2 + 0.5,
                cb.id, fontsize=10, ha='center', va='bottom', weight='bold')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Steiner Tree Connection (GA)')
    ax.set_box_aspect([1, 1, 0.6])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"3D figure saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def _draw_cuboid(ax, cb: Cuboid3D, color: str = "#A8DADC", alpha: float = 0.4):
    """在 3D 坐标轴中绘制一个绕 Z 轴旋转的长方体。"""
    hw = cb.width / 2
    hd = cb.depth / 2
    hh = cb.height / 2
    ca, sa = math.cos(cb.angle), math.sin(cb.angle)

    # 8 个角点（局部坐标）
    corners_local = []
    for dx in (-hw, hw):
        for dy in (-hd, hd):
            for dz in (-hh, hh):
                corners_local.append((dx, dy, dz))

    # 转换到世界坐标
    corners_world = []
    for lx, ly, lz in corners_local:
        wx = cb.cx + lx * ca - ly * sa
        wy = cb.cy + lx * sa + ly * ca
        wz = cb.cz + lz
        corners_world.append((wx, wy, wz))

    # 6 个面
    faces = [
        [0, 1, 3, 2],  # -Z
        [4, 5, 7, 6],  # +Z
        [0, 1, 5, 4],  # -Y
        [2, 3, 7, 6],  # +Y
        [0, 2, 6, 4],  # -X
        [1, 3, 7, 5],  # +X
    ]
    verts = [[corners_world[i] for i in face] for face in faces]
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    poly = Poly3DCollection(verts, alpha=alpha, facecolor=color,
                            edgecolor='black', linewidth=0.8)
    ax.add_collection3d(poly)


# ============================================================================
# Demo
# ============================================================================
def demo(visualize: bool = True, save_path: str | None = None, show_plot: bool = True) -> None:
    """基于 auto_boundary GA case1 排布结果的 3D 连接演示。"""
    print("=== 3D Steiner Tree Connection (GA) ===\n")

    # ── 2D 排布结果提升为 3D ──
    # 模拟 auto_boundary GA 的 case1 输出
    cuboids = [
        Cuboid3D(id="A1", cx=3.50, cy=14.50, cz=1.50, width=5, depth=4, height=3.0, angle=0.0),
        Cuboid3D(id="A2", cx=3.50, cy=4.50,  cz=1.50, width=6, depth=4, height=3.0, angle=0.0),
        Cuboid3D(id="A3", cx=12.50, cy=9.00, cz=1.50, width=4, depth=3, height=3.0, angle=0.0),
    ]
    boundary_w = 18.0
    boundary_h = 20.0

    # ── 定义端口 ──
    ports = [
        # A1 设备
        Port(id="P1_in",  cuboid_id="A1", face="top",    u=0.5, v=0.5),  # 顶部中心
        Port(id="P1_out", cuboid_id="A1", face="right",  u=0.5, v=0.5),  # 右侧中心
        # A2 设备
        Port(id="P2_in",  cuboid_id="A2", face="top",    u=0.5, v=0.5),
        Port(id="P2_out", cuboid_id="A2", face="left",   u=0.5, v=0.5),
        # A3 设备
        Port(id="P3_in",  cuboid_id="A3", face="top",    u=0.5, v=0.5),
        Port(id="P3_out", cuboid_id="A3", face="back",   u=0.5, v=0.5),
    ]

    # ── 定义连接组 ──
    connection_groups = [
        ConnectionGroup(port_ids=["P1_out", "P2_out", "P3_out"]),  # 三个输出汇聚
    ]

    # ── 显示端口坐标 ──
    print("端口世界坐标:")
    cuboid_map = {c.id: c for c in cuboids}
    for p in ports:
        c = cuboid_map[p.cuboid_id]
        wx, wy, wz = c.face_to_world(p.face, p.u, p.v)
        print(f"  {p.id} ({p.cuboid_id}:{p.face}): ({wx:.2f}, {wy:.2f}, {wz:.2f})")

    print()
    print(f"连接组数: {len(connection_groups)}")
    for cg in connection_groups:
        print(f"  端口: {cg.port_ids}")

    # ── 求解 ──
    result = solve_connections(
        cuboids=cuboids,
        ports=ports,
        connection_groups=connection_groups,
        boundary_w=boundary_w,
        boundary_h=boundary_h,
        max_height=8.0,
        population_size=150,
        generations=300,
        tournament_size=3,
        collision_penalty=500.0,
        random_seed=42,
    )

    # ── 输出结果 ──
    print(f"\n{'='*60}")
    print(f"总连接长度: {result['total_length']:.3f}")
    for gi, gr in enumerate(result["groups"]):
        print(f"\n连接组 {gi+1}:")
        print(f"  适应度: {gr['best_fitness']:.3f}")
        print(f"  树总长: {gr['total_length']:.3f}")
        print(f"  碰撞数: {gr['collision_count']}")
        print(f"  Steiner 点数: {len(gr['steiner_points'])}")
        for i, sp in enumerate(gr['steiner_points']):
            print(f"    S{i+1}: ({sp[0]:.2f}, {sp[1]:.2f}, {sp[2]:.2f})")
        print(f"  边:")
        for a, b, d in gr['edges']:
            print(f"    ({a[0]:.2f},{a[1]:.2f},{a[2]:.2f}) -> ({b[0]:.2f},{b[1]:.2f},{b[2]:.2f})  len={d:.3f}")

    # ── 可视化 ──
    if visualize:
        visualize_3d(
            cuboids=cuboids,
            ports=ports,
            result=result,
            save_path=save_path,
            show=show_plot,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="3D Steiner Tree connection optimization using GA."
    )
    parser.add_argument("--no-vis", action="store_true", help="Disable 3D plotting.")
    parser.add_argument("--save-fig", type=str, default=None, help="Save 3D figure to path.")
    parser.add_argument("--no-show", action="store_true", help="Do not open plot window.")
    args = parser.parse_args()

    demo(
        visualize=not args.no_vis,
        save_path=args.save_fig,
        show_plot=not args.no_show,
    )
