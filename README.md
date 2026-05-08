# MEP Generative Design

基于遗传算法和粒子群算法的建筑设备排布与管线连接优化工具。

## 功能

### 二维设备排布
- **GA_layout.py** — 遗传算法，在指定多边形边界内排布矩形设备
- **PSO_layout.py** — 粒子群算法，同样在指定边界内排布
- **auto_boundary_GA.py** — 遗传算法 + **自动生成矩形边界**，边界尺寸参与优化
- **auto_boundary_PSO.py** — 粒子群算法 + 自动生成矩形边界

### 三维管线连接
- **steiner_connect_3D.py** — 正交斯坦纳树连接优化（曼哈顿距离，X/Y/Z 轴对齐布线，碰撞约束）

### 完整管道
- **pipeline_2D_to_3D.py** — 串联 2D 排布 → 3D 连接，支持 JSON 配置

### 测试
- **test_layout.py** — GA vs PSO 对比测试（指定边界）
- **auto_boundary_test.py** — GA vs PSO 对比测试（自动边界）

## 快速开始

```bash
# 单独测试二维排布（自动边界）
python auto_boundary_GA.py

# 单独测试三维正交连接
python steiner_connect_3D.py

# 完整管道（2D → 3D 一键）
python pipeline_2D_to_3D.py

# 使用自定义 JSON 配置
python pipeline_2D_to_3D.py --config example_pipeline_config.json

# GA vs PSO 对比测试
python auto_boundary_test.py --no-plots
```

## 算法特点

| 特性 | 说明 |
|---|---|
| 角度约束 | 10° 整数倍，强偏好水平/竖直（90% 轴对齐初始化） |
| 边界优化 | 自动生成最小可行矩形边界 |
| 碰撞检测 | SAT（分离轴定理）二维碰撞 + OBB/AABB 三维碰撞 |
| 正交布线 | 曼哈顿距离，X→Y→Z 阶梯路径 |
| 端口偏移 | 端口从设备表面外移 0.12 单位，避免碰撞误判 |

## 关键参数

| 参数 | 默认值 | 作用 |
|---|---|---|
| `angle_preference_weight` | 50.0 | 角度偏离惩罚，越大越倾向水平/竖直 |
| `boundary_area_weight` | 0.5 | 边界面积惩罚，越大越倾向紧凑边界 |
| `collision_penalty` | 800.0 | 三维碰撞惩罚，越大越倾向绕障 |
| `population_size` | 80–260 | 种群规模 |
| `generations` / `iterations` | 200–420 | 迭代代数 |

## 依赖

- Python 3.9+
- matplotlib（可选，用于可视化）

## 目录结构

```
.
├── GA_layout.py              # GA 二维排布（指定边界）
├── PSO_layout.py             # PSO 二维排布（指定边界）
├── auto_boundary_GA.py       # GA 二维排布（自动边界）
├── auto_boundary_PSO.py      # PSO 二维排布（自动边界）
├── steiner_connect_3D.py     # 三维正交斯坦纳树连接
├── pipeline_2D_to_3D.py      # 完整管道（2D→3D）
├── test_layout.py            # 指定边界对比测试
├── auto_boundary_test.py     # 自动边界对比测试
├── example_pipeline_config.json  # 管道 JSON 配置示例
└── README.md
```
