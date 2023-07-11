import io
import json
from abc import ABC
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

from base64 import b64decode, b64encode
import numpy as np
import PIL.Image


@dataclass
class Point:
    x: int | float
    y: int | float

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
    holes: list[BasicPolygon] = None


@dataclass
class BoundingBox:
    polygon: Polygon = None
    box: Box = None

    def __post_init__(self):
        if self.polygon == self.box:
            raise ValueError


@dataclass
class MultiPolygon:
    polygons: list[Polygon]


@dataclass
class Raster:
    mask: str
    height: int | float
    width: int | float
    
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
