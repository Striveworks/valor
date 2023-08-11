import io
import math
from base64 import b64decode

import PIL.Image
from pydantic import BaseModel, Field, field_validator


class Point(BaseModel):
    x: float
    y: float

    @field_validator("x")
    @classmethod
    def has_x(cls, v):
        if not isinstance(v, float):
            raise ValueError
        return v

    @field_validator("y")
    @classmethod
    def has_y(cls, v):
        if not isinstance(v, float):
            raise ValueError
        return v

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"

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
        return Point(x=-self.x, y=-self.y)

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
        return self + other

    def __isub__(self, other):
        if not isinstance(other, Point):
            raise TypeError
        return self - other

    def dot(self, other):
        if not isinstance(other, Point):
            raise TypeError
        return (self.x * other.x) + (self.y * other.y)


class LineSegment(BaseModel):
    points: tuple[Point, Point]

    def delta_xy(self) -> Point:
        return self.points[0] - self.points[1]

    def parallel(self, other) -> bool:
        if not isinstance(other, LineSegment):
            raise TypeError

        d1 = self.delta_xy()
        d2 = other.delta_xy()

        slope1 = d1.y / d1.x if d1.x else math.inf
        slope2 = d2.y / d2.x if d2.x else math.inf
        return math.isclose(slope1, slope2)

    def perpendicular(self, other) -> bool:
        if not isinstance(other, LineSegment):
            raise TypeError

        d1 = self.delta_xy()
        d2 = other.delta_xy()

        slope1 = d1.y / d1.x if d1.x else math.inf
        slope2 = d2.y / d2.x if d2.x else math.inf

        if slope1 == 0 and math.fabs(slope2) == math.inf:
            return True
        elif math.fabs(slope1) == math.inf and slope2 == 0:
            return True
        elif slope2 == 0:
            return False
        else:
            return math.isclose(slope1, -1.0 / slope2)


class BasicPolygon(BaseModel):
    points: list[Point]

    @field_validator("points")
    @classmethod
    def check_points(cls, v):
        if v is not None:
            if len(set(v)) < 3:
                raise ValueError(
                    "Polygon must be composed of at least three unique points."
                )
            # Remove duplicate of start point
            if v[0] == v[-1]:
                v = v[:-1]
            # @TODO (maybe) implement self-intersection check?
        return v

    @property
    def left(self):
        return min(self.points, key=lambda point: point.x).x

    @property
    def right(self):
        return max(self.points, key=lambda point: point.x).x

    @property
    def top(self):
        return max(self.points, key=lambda point: point.y).y

    @property
    def bottom(self):
        return min(self.points, key=lambda point: point.y).y

    @property
    def width(self):
        return self.right - self.left

    @property
    def height(self):
        return self.top - self.bottom

    @property
    def segments(self) -> list[LineSegment]:
        plist = self.points + [self.points[0]]
        return [
            LineSegment(points=(plist[i], plist[i + 1]))
            for i in range(len(plist) - 1)
        ]

    def __str__(self):
        # in PostGIS polygon has to begin and end at the same point
        pts = self.points
        if pts[0] != pts[-1]:
            pts = pts + [pts[0]]
        points_string = [f"({','.join([str(pt.x), str(pt.y)])})" for pt in pts]
        return f"({','.join(points_string)})"

    def wkt(self, partial: bool = False) -> str:
        # in PostGIS polygon has to begin and end at the same point
        pts = self.points
        if pts[0] != pts[-1]:
            pts = pts + [pts[0]]
        points_string = [" ".join([str(pt.x), str(pt.y)]) for pt in pts]
        wkt_format = f"({', '.join(points_string)})"
        if partial:
            return wkt_format
        return f"POLYGON ({wkt_format})"


class Polygon(BaseModel):
    boundary: BasicPolygon
    holes: list[BasicPolygon] | None = Field(default=None)

    def __str__(self):
        polys = [str(self.boundary)]
        if self.holes:
            for hole in self.holes:
                polys.append(str(hole))
        return f"({','.join(polys)})"

    def wkt(self, partial: bool = False) -> str:
        polys = [self.boundary.wkt(partial=True)]
        if self.holes:
            for hole in self.holes:
                polys.append(hole.wkt(partial=True))
        wkt_format = f"({', '.join(polys)})"
        if partial:
            return wkt_format
        return f"POLYGON {wkt_format}"


class MultiPolygon(BaseModel):
    polygons: list[Polygon]

    def wkt(self) -> str:
        plist = [polygon.wkt(partial=True) for polygon in self.polygons]
        return f"MULTIPOLYGON ({', '.join(plist)})"

    # @TODO: Unsure if keeping this
    # @classmethod
    # def from_wkt(cls, wkt: str | None):
    #     if not wkt:
    #         return None
    #     if re.search("^MULTIPOLYGON", wkt):
    #         polygons = []
    #         poly_text = re.findall("\(\((.*)\)\)", wkt)[0].split("),(")
    #         for poly in poly_text:
    #             points = []
    #             for numerics in poly.strip().split(","):
    #                 coords = numerics.strip().split(" ")
    #                 points.append(
    #                     Point(
    #                         x=float(coords[0]),
    #                         y=float(coords[1]),
    #                     )
    #                 )
    #             polygons.append(BasicPolygon(points=points))

    #         if len(polygons) == 1:
    #             return cls(
    #                 boundary=polygons[0],
    #                 holes=None,
    #             )
    #         elif polygons:
    #             return cls(
    #                 boundary=polygons[0],
    #                 holes=polygons[1:],
    #             )
    #     raise ValueError


class BoundingBox(BaseModel):
    polygon: BasicPolygon

    @field_validator("polygon")
    @classmethod
    def valid_polygon(cls, v):
        if len(set(v.points)) != 4:
            raise ValueError(
                "bounding box polygon requires exactly 4 unique points."
            )
        return v

    @classmethod
    def from_extrema(cls, xmin: float, ymin: float, xmax: float, ymax: float):
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
    def left(self):
        return self.polygon.left

    @property
    def right(self):
        return self.polygon.right

    @property
    def top(self):
        return self.polygon.top

    @property
    def bottom(self):
        return self.polygon.bottom

    @property
    def width(self):
        return self.polygon.width

    @property
    def height(self):
        return self.polygon.height

    def is_rectangular(self):

        # retrieve segments
        segments = self.polygon.segments

        # check if segments are parallel
        if not (
            segments[0].parallel(segments[2])
            and segments[1].parallel(segments[3])
        ):
            return False

        # check if segments are perpendicular
        for i in range(3):
            if not segments[i].perpendicular(segments[i + 1]):
                return False

        return True

    def is_rotated(self):
        # check if rectangular
        if not self.is_rectangular():
            return False

        # check if rotation exists by seeing if corners do not share values.
        x = set([p.x for p in self.polygon.points])
        y = set([p.y for p in self.polygon.points])
        return (len(x) != 2) and (len(y) != 2)

    def is_skewed(self):
        return not (self.is_rotated() or self.is_rectangular())

    def wkt(self) -> str:
        return self.polygon.wkt()


class Raster(BaseModel):
    mask: str = Field(frozen=True)
    # model_config = ConfigDict(extra="allow", validate_assignment=True)

    @field_validator("mask")
    @classmethod
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
            self._mask_bytes = b64decode(self.mask)
        return self._mask_bytes

    @property
    def pil_mask(self) -> PIL.Image:
        with io.BytesIO(self.mask_bytes) as f:
            return PIL.Image.open(f)
