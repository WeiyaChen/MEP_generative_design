from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from layout_geometry_def import (
    Point,
    EPS,
    VALID_ANGLES_DEG,
    VALID_ANGLES_RAD,
    clamp,
    wrap_angle,
    cross,
    on_segment,
    segments_intersect,
    segments_properly_intersect,
    point_in_polygon,
    polygon_edges,
    polygon_inside_polygon,
    project_polygon,
    convex_overlap_strict,
    axis_alignment_deviation,
    snap_angle_to_10deg,
)


@dataclass
class Gene:
    cx: float
    cy: float
    angle: float
    active: bool
    

@dataclass(frozen=True)
class RectangleSpec:
    id: str
    width: float
    height: float
    rotatable: bool = True
    mandatory: bool = False


@dataclass(frozen=True)
class SquareObstacle:
    x: float
    y: float
    size: float

    def as_polygon(self) -> List[Point]:
        x, y, s = self.x, self.y, self.size
        return [(x, y), (x + s, y), (x + s, y + s), (x, y + s)]


@dataclass(frozen=True)
class PlacedRect:
    id: str
    cx: float
    cy: float
    width: float
    height: float
    angle: float
    mandatory: bool

    @property
    def area(self) -> float:
        return self.width * self.height

    def corners(self) -> List[Point]:
        hw = self.width / 2.0
        hh = self.height / 2.0
        ca = math.cos(self.angle)
        sa = math.sin(self.angle)
        local = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
        pts: List[Point] = []
        for lx, ly in local:
            x = self.cx + lx * ca - ly * sa
            y = self.cy + lx * sa + ly * ca
            pts.append((x, y))
        return pts


def rect_inside_polygon(rect: PlacedRect, polygon: Sequence[Point]) -> bool:
    return polygon_inside_polygon(rect.corners(), polygon)


__all__ = [
    "Point",
    "EPS",
    "VALID_ANGLES_DEG",
    "VALID_ANGLES_RAD",
    "snap_angle_to_10deg",
    "RectangleSpec",
    "SquareObstacle",
    "PlacedRect",
    "Gene",
    "clamp",
    "wrap_angle",
    "cross",
    "on_segment",
    "segments_intersect",
    "segments_properly_intersect",
    "point_in_polygon",
    "polygon_edges",
    "polygon_inside_polygon",
    "project_polygon",
    "convex_overlap_strict",
    "rect_inside_polygon",
    "axis_alignment_deviation",
]
