import io
from base64 import b64decode, b64encode
from dataclasses import dataclass, field
from typing import List

import numpy as np
import PIL.Image


@dataclass
class Point:
    x: float
    y: float

    def __post_init__(self):
        if isinstance(self.x, int):
            self.x = float(self.x)
        if isinstance(self.y, int):
            self.y = float(self.y)

        if not isinstance(self.x, float):
            raise TypeError("Point coordinates should be `float` type.")
        if not isinstance(self.y, float):
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
        # unpack
        if isinstance(self.min, dict):
            self.min = Point(**self.min)
        if isinstance(self.max, dict):
            self.max = Point(**self.max)

        # validate
        if self.min.x > self.max.x:
            raise ValueError("Cannot have xmin > xmax")
        if self.min.y > self.max.y:
            raise ValueError("Cannot have ymin > ymax")


@dataclass
class BasicPolygon:
    """Class for representing a bounding region."""

    points: List[Point] = field(default_factory=list)

    def __post_init__(self):
        # unpack & validate
        if not isinstance(self.points, list):
            raise TypeError("Member `points` is not a list.")
        for i in range(len(self.points)):
            if isinstance(self.points[i], dict):
                self.points[i] = Point(**self.points[i])
            if not isinstance(self.points[i], Point):
                raise TypeError("Element in points is not a `Point`.")
        if len(set(self.points)) < 3:
            raise ValueError(
                "BasicPolygon needs at least 3 unique points to be valid."
            )

    def xy_list(self):
        """Returns list of `Point` objects."""
        return [pt for pt in self.points]

    def tuple_list(self):
        """Returns list of points as tuples (x,y)."""
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
    holes: List[BasicPolygon] = field(default_factory=list)

    def __post_init__(self):
        # unpack & validate
        if isinstance(self.boundary, dict):
            self.boundary = BasicPolygon(**self.boundary)
        if not isinstance(self.boundary, BasicPolygon):
            raise TypeError(
                "boundary should be of type `velour.schemas.BasicPolygon`"
            )
        if self.holes:
            if not isinstance(self.holes, list):
                raise TypeError(
                    f"holes should be a list of `velour.schemas.BasicPolygon`. Got `{type(self.holes)}`."
                )
            for i in range(len(self.holes)):
                if isinstance(self.holes[i], dict):
                    self.holes[i] = BasicPolygon(**self.holes[i])
                if not isinstance(self.holes[i], BasicPolygon):
                    raise TypeError(
                        "holes list should contain elements of type `velour.schemas.BasicPolygon`"
                    )


@dataclass
class BoundingBox:
    polygon: BasicPolygon

    def __post_init__(self):
        if isinstance(self.polygon, dict):
            self.polygon = BasicPolygon(**self.polygon)
        if not isinstance(self.polygon, BasicPolygon):
            raise TypeError(
                "polygon should be of type `velour.schemas.BasicPolygon`"
            )
        if len(self.polygon.points) != 4:
            raise ValueError(
                "Bounding box should be made of a 4-point polygon."
            )

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

    @property
    def xmin(self):
        return self.polygon.xmin

    @property
    def xmax(self):
        return self.polygon.xmax

    @property
    def ymin(self):
        return self.polygon.ymin

    @property
    def ymax(self):
        return self.polygon.ymax


@dataclass
class MultiPolygon:
    polygons: List[Polygon] = field(default_factory=list)

    def __post_init__(self):
        # unpack & validate
        if not isinstance(self.polygons, list):
            raise TypeError(
                "polygons should be list of `velour.schemas.Polyon`"
            )
        for i in range(len(self.polygons)):
            if isinstance(self.polygons[i], dict):
                self.polygons[i] = Polygon(**self.polygons[i])
            if not isinstance(self.polygons[i], Polygon):
                raise TypeError(
                    "polygons list should contain elements of type `velour.schemas.Polygon`"
                )


@dataclass
class Raster:
    mask: str

    def __post_init__(self):
        if not isinstance(self.mask, str):
            raise TypeError("mask should be of type `str`")

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
        )

    def to_numpy(self) -> np.ndarray:
        mask_bytes = b64decode(self.mask)
        with io.BytesIO(mask_bytes) as f:
            img = PIL.Image.open(f)
            return np.array(img)
