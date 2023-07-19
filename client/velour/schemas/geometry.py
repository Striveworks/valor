import io
import json
from abc import ABC
from base64 import b64decode, b64encode
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np
import PIL.Image


@dataclass
class Point:
    x: float
    y: float

    def __post_init__(self):
        if not isinstance(self.x, float | int):
            raise TypeError("Point coordinates should be `float` type.")
        if not isinstance(self.y, float | int):
            raise TypeError("Point coordinates should be `float` type.")
        self.x = float(self.x)
        self.y = float(self.y)
        
    def __hash__(self):
        return hash(f"{self.x},{self.y}")

    def resize(
        self, og_img_h: int, og_img_w: int, new_img_h: int, new_img_w: int
    ) -> "Point":
        h_factor, w_factor = new_img_h / og_img_h, new_img_w / og_img_w
        return Point(x=w_factor * self.x, y=h_factor * self.y)


@dataclass
class Box:
    min: Point
    max: Point

    def __post_init__(self):
        if self.min.x > self.max.x:
            raise ValueError("Cannot have xmin > xmax")
        if self.min.y > self.max.y:
            raise ValueError("Cannot have ymin > ymax")


@dataclass
class BasicPolygon:
    """Class for representing a bounding region."""

    points: List[Point] = field(default_factory=list)

    def __post_init__(self):
        if not isinstance(self.points, list):
            raise TypeError("Member `points` is not a list.")
        for point in self.points:
            if not isinstance(point, Point):
                raise TypeError("Element in points is not a `Point`.")
        if len(set(self.points)) < 3:
            raise ValueError("BasicPolygon needs at least 3 unique points to be valid.")

    def xy_list(self) -> list[Point]:
        return [pt for pt in self.points]

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
    def from_box(cls, box: Box):
        return cls(
            points=[
                Point(box.min.x, box.min.y),
                Point(box.min.x, box.max.y),
                Point(box.max.x, box.max.y),
                Point(box.max.x, box.min.y),
            ]
        )


@dataclass
class Polygon:
    boundary: BasicPolygon
    holes: list[BasicPolygon] = field(default_factory=list)

    def __post_init__(self):
        if not isinstance(self.boundary, BasicPolygon):
            raise TypeError("boundary should be of type `velour.schemas.BasicPolygon`")
        if not isinstance(self.holes, list):
            raise TypeError("holes should be a list of `velour.schemas.BasicPolygon`")
        for hole in self.holes:
            if not isinstance(hole, BasicPolygon):
                raise TypeError("holes list should contain elements of type `velour.schemas.BasicPolygon`")


@dataclass
class BoundingBox:
    polygon: BasicPolygon

    def __post_init__(self):
        if not isinstance(self.polygon, BasicPolygon):
            raise TypeError("polygon should be of type `velour.schemas.BasicPolygon`")
        if len(self.polygon.points) != 4:
            raise ValueError("Bounding box should be made of a 4-point polygon.")

    @classmethod
    def from_extrema(cls, xmin: float, xmax: float, ymin: float, ymax: float):
        return cls(
            polygon=BasicPolygon(
                points=[
                    Point(x=xmin, y=ymin),
                    Point(x=xmax, y=ymin),
                    Point(x=xmax, y=ymax),
                    Point(x=xmin, y=ymax),
                ]
            )
        )


@dataclass
class MultiPolygon:
    polygons: list[Polygon]

    def __post_init__(self):
        if not isinstance(self.polygons, list):
            raise TypeError("polygons should be list of `velour.schemas.Polyon`")
        for polygon in self.polygons:
            if not isinstance(polygon, Polygon):
                raise TypeError("polygons list should contain elements of type `velour.schemas.Polygon`")


@dataclass
class Raster:
    mask: str
    shape: tuple[int,int]

    def __post_init__(self):
        if not isinstance(self.mask, str):
            raise TypeError("mask should be of type `str`")
        if not isinstance(self.shape, tuple):
            raise TypeError("shape should be of type tuple")
        if len(self.shape) != 2:
            raise ValueError("raster currently only supports 2d arrays")
        for dim in self.shape:
            if not isinstance(dim, int):
                raise TypeError("dimesions in shape should be of type `int`")

    @classmethod
    def from_numpy(cls, mask: np.ndarray):
        if len(mask.shape) != 2:
            raise ValueError("raster currently only supports 2d arrays")
        if mask.dtype != bool:
            raise ValueError(
                f"Expecting a binary mask (i.e. of dtype bool) but got dtype {mask.dtype}"
            )
        f = io.BytesIO()
        PIL.Image.fromarray(mask).save(f, format="PNG")
        f.seek(0)
        mask_bytes = f.read()
        f.close()
        return cls(
            mask=b64encode(mask_bytes).decode(),
            shape=mask.shape,
        )

    def decode(self) -> np.ndarray:
        mask_bytes = b64decode(self.mask)
        with io.BytesIO(mask_bytes) as f:
            img = PIL.Image.open(f)
            return np.array(img)
