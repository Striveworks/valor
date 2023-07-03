import json
from abc import ABC
from dataclasses import dataclass, field
from typing import List, Tuple, Union, Optional

import numpy as np


@dataclass
class BoundingBox:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def __post_init__(self):
        if self.xmin > self.xmax:
            raise ValueError("Cannot have xmin > xmax")
        if self.ymin > self.ymax:
            raise ValueError("Cannot have ymin > ymax")


@dataclass
class Point:
    x: float
    y: float

    def resize(
        self, og_img_h: int, og_img_w: int, new_img_h: int, new_img_w: int
    ) -> "Point":
        h_factor, w_factor = new_img_h / og_img_h, new_img_w / og_img_w
        return Point(x=w_factor * self.x, y=h_factor * self.y)


@dataclass
class Polygon:
    """Class for representing a bounding region."""

    points: List[Point]

    def xy_list(self):
        return [(pt.x, pt.y) for pt in self.points]

    @property
    def xmin(self):
        return min(p.x for p in self.points)

    @property
    def ymin(self):
        return min(p.y for p in self.points)

    @property
    def xmax(self):
        return max(p.x for p in self.points)

    @property
    def ymax(self):
        return max(p.y for p in self.points)

    @classmethod
    def from_xmin_ymin_xmax_ymax(cls, xmin, ymin, xmax, ymax):
        return cls(
            points=[
                Point(xmin, ymin),
                Point(xmin, ymax),
                Point(xmax, ymax),
                Point(xmax, ymin),
            ]
        )
    
@dataclass
class MultiPolygon:
    polygon: List[Polygon]
    hole: Optional[List[Polygon]] = None


@dataclass
class Raster:
    mask: np.ndarray

    def __post_init__(self):
        if self.mask.dtype != bool:
            raise ValueError(
                f"Expecting a binary mask (i.e. of dtype bool) but got dtype {self.mask.dtype}"
            )