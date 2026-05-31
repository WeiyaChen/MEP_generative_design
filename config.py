from __future__ import annotations
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AlgorithmConfig:
    population_size: int = 80
    generations: int = 200
    crossover_rate: float = 0.85
    mutation_rate: float = 0.16
    elite_count: int = 6
    tournament_size: int = 3
    angle_preference_weight: float = 50.0
    boundary_area_weight: float = 0.5
    random_seed: Optional[int] = 7

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AlgorithmConfig":
        if "ga_2d_kwargs" in data:
            data = data["ga_2d_kwargs"]
        return cls(
            population_size=int(data.get("population_size", cls.population_size)),
            generations=int(data.get("generations", cls.generations)),
            crossover_rate=float(data.get("crossover_rate", cls.crossover_rate)),
            mutation_rate=float(data.get("mutation_rate", cls.mutation_rate)),
            elite_count=int(data.get("elite_count", cls.elite_count)),
            tournament_size=int(data.get("tournament_size", cls.tournament_size)),
            angle_preference_weight=float(
                data.get("angle_preference_weight", cls.angle_preference_weight)
            ),
            boundary_area_weight=float(
                data.get("boundary_area_weight", cls.boundary_area_weight)
            ),
            random_seed=data.get("random_seed", cls.random_seed),
        )

    @classmethod
    def from_json(cls, path: str) -> "AlgorithmConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_kwargs(self) -> Dict[str, Any]:
        result = asdict(self)
        if result["random_seed"] is None:
            result.pop("random_seed")
        return result


@dataclass
class PSOConfig:
    swarm_size: int = 40
    iterations: int = 300
    w_start: float = 0.9
    w_end: float = 0.4
    c1: float = 1.8
    c2: float = 1.8
    vmax_cxy_ratio: float = 0.15
    vmax_angle: float = 60.0
    vmax_active: float = 0.35
    angle_preference_weight: float = 50.0
    boundary_area_weight: float = 0.5
    random_seed: Optional[int] = 7

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PSOConfig":
        if "pso_kwargs" in data:
            data = data["pso_kwargs"]
        return cls(
            swarm_size=int(data.get("swarm_size", cls.swarm_size)),
            iterations=int(data.get("iterations", cls.iterations)),
            w_start=float(data.get("w_start", cls.w_start)),
            w_end=float(data.get("w_end", cls.w_end)),
            c1=float(data.get("c1", cls.c1)),
            c2=float(data.get("c2", cls.c2)),
            vmax_cxy_ratio=float(data.get("vmax_cxy_ratio", cls.vmax_cxy_ratio)),
            vmax_angle=float(data.get("vmax_angle", cls.vmax_angle)),
            vmax_active=float(data.get("vmax_active", cls.vmax_active)),
            angle_preference_weight=float(
                data.get("angle_preference_weight", cls.angle_preference_weight)
            ),
            boundary_area_weight=float(
                data.get("boundary_area_weight", cls.boundary_area_weight)
            ),
            random_seed=data.get("random_seed", cls.random_seed),
        )

    @classmethod
    def from_json(cls, path: str) -> "PSOConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_kwargs(self) -> Dict[str, Any]:
        result = asdict(self)
        if result["random_seed"] is None:
            result.pop("random_seed")
        return result


@dataclass
class Equipment3DConfig:
    rect_id: str
    height: float
    ports: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Equipment3DConfig":
        return cls(
            rect_id=str(data["rect_id"]),
            height=float(data["height"]),
            ports=[dict(p) for p in data.get("ports", [])],
        )


@dataclass
class PipelineConfig:
    rectangles: List[Dict[str, Any]] = field(default_factory=list)
    equipment: List[Equipment3DConfig] = field(default_factory=list)
    connections: List[List[str]] = field(default_factory=list)
    layout_seed: int = 42
    ga_2d_kwargs: Dict[str, Any] = field(default_factory=dict)
    ga_3d_kwargs: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        return cls(
            rectangles=[dict(r) for r in data.get("rectangles", [])],
            equipment=[Equipment3DConfig.from_dict(e) for e in data.get("equipment", [])],
            connections=[list(group) for group in data.get("connections", [])],
            layout_seed=int(data.get("layout_seed", 42)),
            ga_2d_kwargs=dict(data.get("ga_2d_kwargs", {})),
            ga_3d_kwargs=dict(data.get("ga_3d_kwargs", {})),
        )

    @classmethod
    def from_json(cls, path: str) -> "PipelineConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
