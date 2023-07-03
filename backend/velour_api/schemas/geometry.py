import io
import math
from base64 import b64decode
from uuid import uuid4

import PIL.Image
from pydantic import BaseModel, Field, Extra, root_validator, validator


class Point(BaseModel):
    x: float
    y: float

    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, other):
        if not isinstance(other, Point):
            raise TypeError
        return math.isclose(self.x, other.x) and math.isclose(self.y, other.y)

    def __neq__(self, other):
        if not isinstance(other, Point):
            raise TypeError
        return not (self == other)

    def __neg__(self):
        self.x = -self.x
        self.y = -self.y

    def __add__(self, other):
        if not isinstance(other, Point):
            raise TypeError
        newx = self.x + other.x
        newy = self.y + other.y
        return Point(x=newx, y=newy)

    def __sub__(self, other):
        if not isinstance(other, Point):
            raise TypeError
        newx = self.x - other.x
        newy = self.y - other.y
        return Point(x=newx, y=newy)

    def __iadd__(self, other):
        if not isinstance(other, Point):
            raise TypeError
        self = self + other

    def __isub__(self, other):
        if not isinstance(other, Point):
            raise TypeError
        self = self - other

    def dot(self, other):
        if not isinstance(other, Point):
            raise TypeError
        return (self.x * other.x) + (self.y * other.y)


class LineSegment(BaseModel):
    points: tuple[Point, Point]

    def delta_xy(self):
        return self.points[0] - self.points[1]

    def parallel(self, other):
        if not isinstance(other, LineSegment):
            raise TypeError
        d1 = self.delta_xy()
        d2 = self.delta_xy()
        return d1 == d2

    def perpendicular(self, other):
        if not isinstance(other, LineSegment):
            raise TypeError
        d1 = self.delta_xy()
        d2 = -self.delta_xy()
        return d1 == Point(x=d2.y, y=d2.x)


def _validate_single_polygon(points: list[Point]):
    if len(set(points)) < 3:
        raise ValueError(
            "Polygon must be composed of at least three unique points."
        )

    # @TODO (maybe) implement self-intersection check?


def _validate_box_polygon(poly: list[Point]):
    if len(poly) != 4:
        raise ValueError("Box Polygon is composed of exactly four points.")
    elif poly[0] == poly[-1]:
        raise ValueError("Box Polygon requires four unique points.")


class Polygon(BaseModel):
    points: list[Point]

    @validator("points")
    def check_points(cls, v):
        if v is not None:
            _validate_single_polygon(v)
        return v

    @property
    def segments(self):
        plist = self.points + self.points[0]
        return [(plist[i], plist[i + 1]) for i in range(len(self.points))]


class MultiPolygon(BaseModel):
    polygons: list[Polygon]
    holes: list[Polygon] = []

    @validator("polygons")
    def check_polygon(cls, v):
        for poly in v:
            _validate_single_polygon(poly)
        return v

    @validator("holes")
    def check_holes(cls, v):
        if v:
            for hole in v:
                _validate_single_polygon(hole)
        return v


class Box(BaseModel):
    min: Point
    max: Point

    @root_validator(skip_on_failure=True)
    def extrema_check(cls, values):
        if values["max"].x <= values["min"].x:
            raise ValueError("Invalid extrema (x-axis).")
        elif values["max"].y <= values["min"].y:
            raise ValueError("Invalid extrema (y-axis).")
        return values


class BoundingBox(BaseModel):
    polygon: Polygon
    box: Box

    @root_validator(skip_on_failure=True)
    def rigid_or_skewed(cls, values):
        if (values["polygon"] is None) == (values["box"] is None):
            raise ValueError("Must define either polygon or box.")
        return values

    @validator("polygon")
    def enough_pts(cls, v):
        if v is not None:
            _validate_single_polygon(v)
            _validate_box_polygon(v)
        return v

    @property
    def is_rectangular(self):
        if hasattr(self, "_is_rectangular"):
            return self._is_rectangular

        if self.box is not None:
            self._is_rectangular = True
            return self._is_rectangular

        segments = [
            LineSegment(
                points=(self.polygon.points[0], self.polygon.points[1])
            ),
            LineSegment(
                points=(self.polygon.points[1], self.polygon.points[2])
            ),
            LineSegment(
                points=(self.polygon.points[2], self.polygon.points[3])
            ),
            LineSegment(
                points=(self.polygon.points[3], self.polygon.points[0])
            ),
        ]

        # check if segments are parallel
        if not (
            segments[0].parallel(segments[2])
            and segments[1].parallel(segments[3])
        ):
            self._is_rectangular = False
            return self._is_rectangular

        # check if segments are perpendicular
        for i in range(3):
            if not segments[i].perpendicular(segments[i + 1]):
                self._is_rectangular = False
                return self._is_rectangular

        self._is_rectangular = True
        return self._is_rectangular

    @property
    def is_rotated(self):
        if hasattr(self, "_is_rotated"):
            return self._is_rotated

        if not self.is_rectangular:
            self._is_rotated = False
            return self._is_rotated

        # check if rotation exists by seeing if corners do not share values.
        x = set([p.x for p in self.polygon.points])
        y = set([p.y for p in self.polygon.points])
        return (len(x) != 2) and (len(y) != 2)

    @property
    def is_skewed(self):
        return not (self.is_rotated or self.is_rectangular)


def _mask_bytes_to_pil(mask_bytes: bytes) -> PIL.Image.Image:
    with io.BytesIO(mask_bytes) as f:
        return PIL.Image.open(f)


class Raster(BaseModel):
    mask: str = Field(allow_mutation=False)
    height: int | float
    widht: int | float

    class Config:
        extra = Extra.allow
        validate_assignment = True

    @root_validator
    def correct_mask_shape(cls, values):
        mask_size = _mask_bytes_to_pil(b64decode(values["mask"])).size
        image_size = (values["width"], values["height"])
        if mask_size != image_size:
            raise ValueError(
                f"Expected mask and image to have the same size, but got size {mask_size} for the mask and {image_size} for image."
            )
        return values

    @validator("mask")
    def check_png_and_mode(cls, v):
        """Check that the bytes are for a png file and is binary"""
        f = io.BytesIO(b64decode(v))
        img = PIL.Image.open(f)
        f.close()
        if img.format != "PNG":
            raise ValueError(
                f"Expected image format PNG but got {img.format}."
            )
        if img.mode != "1":
            raise ValueError(
                f"Expected image mode to be binary but got mode {img.mode}."
            )
        return v

    @property
    def mask_bytes(self) -> bytes:
        if not hasattr(self, "_mask_bytes"):
            self._mask_bytes = b64decode(self.base64_mask)
        return self._mask_bytes

    @property
    def pil_mask(self) -> PIL.Image:
        return _mask_bytes_to_pil(self.mask_bytes)
