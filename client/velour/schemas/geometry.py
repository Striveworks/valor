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
        try:
            assert isinstance(self.x, float)
            assert isinstance(self.y, float)
        except AssertionError:
            raise ValueError("Point coordinates should be `float` type.")
        
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
            raise ValueError("Member `points` is not a list.")
        
        try:
            for point in self.points:
                assert isinstance(point, Point)
        except AssertionError:
            raise ValueError("Element in points is not a `Point`.")
        
        if len(set(self.points)) < 3:
            raise ValueError("BasicPolygon needs at least 3 unique points to be valid.")

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
    holes: list[BasicPolygon] = field(default=[], default_factory=list)

    def __post_init__(self):
        assert isinstance(self.boundary, BasicPolygon)
        assert isinstance(self.holes, list)
        if self.holes:
            for hole in self.holes:
                assert isinstance(hole, BasicPolygon)



@dataclass
class BoundingBox:
    polygon: BasicPolygon

    def __post_init__(self):
        assert isinstance(self.polygon, BasicPolygon)
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
        assert isinstance(self.polygons, list)
        for polygon in self.polygons:
            assert isinstance(polygon, Polygon)


@dataclass
class Raster:
    mask: str
    height: int
    width: int

    def __post_init__(self):
        assert isinstance(self.mask, str)
        assert isinstance(self.height, int)
        assert isinstance(self.width, int)

    @classmethod
    def from_numpy(cls, mask: np.ndarray):
        assert len(mask.shape) == 2
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
            height=mask.shape[0],
            width=mask.shape[1],
        )

    def decode(self) -> np.ndarray:
        mask_bytes = b64decode(self.mask)
        with io.BytesIO(mask_bytes) as f:
            img = PIL.Image.open(f)
            return np.array(img)
