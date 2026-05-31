from __future__ import annotations
import math
from typing import List, Sequence, Tuple

Point = Tuple[float, float]
EPS = 1e-9

VALID_ANGLES_DEG = [float(i) for i in range(0, 360, 10)]
VALID_ANGLES_RAD = [math.radians(d) for d in VALID_ANGLES_DEG]


def wrap_angle(theta: float) -> float:
    while theta <= -math.pi:
        theta += 2 * math.pi
    while theta > math.pi:
        theta -= 2 * math.pi
    return theta


def snap_angle_to_10deg(theta_rad: float) -> float:
    deg = math.degrees(theta_rad)
    snapped_deg = round(deg / 10.0) * 10.0
    return wrap_angle(math.radians(snapped_deg))


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def cross(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def on_segment(a: Point, b: Point, p: Point) -> bool:
    if abs(cross(a, b, p)) > EPS:
        return False
    return (
        min(a[0], b[0]) - EPS <= p[0] <= max(a[0], b[0]) + EPS
        and min(a[1], b[1]) - EPS <= p[1] <= max(a[1], b[1]) + EPS
    )


def segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    c1 = cross(a, b, c)
    c2 = cross(a, b, d)
    c3 = cross(c, d, a)
    c4 = cross(c, d, b)
    if (c1 * c2 < -EPS) and (c3 * c4 < -EPS):
        return True
    if abs(c1) <= EPS and on_segment(a, b, c):
        return True
    if abs(c2) <= EPS and on_segment(a, b, d):
        return True
    if abs(c3) <= EPS and on_segment(c, d, a):
        return True
    if abs(c4) <= EPS and on_segment(c, d, b):
        return True
    return False


def segments_properly_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    c1 = cross(a, b, c)
    c2 = cross(a, b, d)
    c3 = cross(c, d, a)
    c4 = cross(c, d, b)
    return (c1 * c2 < -EPS) and (c3 * c4 < -EPS)


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        a = polygon[i]
        b = polygon[(i + 1) % n]
        if on_segment(a, b, point):
            return True
        yi, yj = a[1], b[1]
        if (yi > y) != (yj > y):
            x_intersect = (b[0] - a[0]) * (y - yi) / (yj - yi) + a[0]
            if x < x_intersect:
                inside = not inside
    return inside


def polygon_edges(poly: Sequence[Point]) -> List[Tuple[Point, Point]]:
    return [(poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly))]


def polygon_inside_polygon(inner: Sequence[Point], outer: Sequence[Point]) -> bool:
    if not all(point_in_polygon(p, outer) for p in inner):
        return False
    outer_edges = polygon_edges(outer)
    inner_edges = polygon_edges(inner)
    for ie in inner_edges:
        for oe in outer_edges:
            if segments_properly_intersect(ie[0], ie[1], oe[0], oe[1]):
                return False
    return True


def project_polygon(poly: Sequence[Point], axis: Point) -> Tuple[float, float]:
    ax, ay = axis
    values = [p[0] * ax + p[1] * ay for p in poly]
    return min(values), max(values)


def convex_overlap_strict(poly1: Sequence[Point], poly2: Sequence[Point]) -> bool:
    for poly in (poly1, poly2):
        n = len(poly)
        for i in range(n):
            a = poly[i]
            b = poly[(i + 1) % n]
            edge = (b[0] - a[0], b[1] - a[1])
            axis = (-edge[1], edge[0])
            length = math.hypot(axis[0], axis[1])
            if length <= EPS:
                continue
            axis = (axis[0] / length, axis[1] / length)
            min1, max1 = project_polygon(poly1, axis)
            min2, max2 = project_polygon(poly2, axis)
            if (max1 <= min2 + EPS) or (max2 <= min1 + EPS):
                return False
    return True


def axis_alignment_deviation(angle: float) -> float:
    return abs(math.sin(2.0 * angle))


__all__ = [
    "Point",
    "EPS",
    "VALID_ANGLES_DEG",
    "VALID_ANGLES_RAD",
    "wrap_angle",
    "snap_angle_to_10deg",
    "clamp",
    "cross",
    "on_segment",
    "segments_intersect",
    "segments_properly_intersect",
    "point_in_polygon",
    "polygon_edges",
    "polygon_inside_polygon",
    "project_polygon",
    "convex_overlap_strict",
    "axis_alignment_deviation",
]
