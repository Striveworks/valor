import io
import json
import math
import re
from base64 import b64decode
from enum import Enum
from uuid import uuid4

import PIL.Image
from pydantic import BaseModel, Extra, Field, root_validator, validator


class Point(BaseModel):
    x: float
    y: float

    @validator("x")
    def has_x(cls, v):
        if not isinstance(v, float):
            raise ValueError
        return v

    @validator("y")
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
        
        def _slope(d):
            if d.x < 0 and d.y >= 0:
                d.x = math.fabs(d.x)
            elif d.y < 0 and d.x >= 0:
                d.y = math.fabs(d.y)
            return d
        
        d1 = _slope(self.delta_xy())
        d2 = _slope(other.delta_xy())
        return d1 == d2

    def perpendicular(self, other) -> bool:
        if not isinstance(other, LineSegment):
            raise TypeError
        
        def _slope(d):
            if d.x < 0 and d.y >= 0:
                d.x = math.fabs(d.x)
            elif d.y < 0 and d.x >= 0:
                d.y = math.fabs(d.y)
            return d

        d1 = _slope(self.delta_xy())
        d2 = _slope(-other.delta_xy())
        return d1 == Point(x=d2.y, y=d2.x)


class BasicPolygon(BaseModel):
    points: list[Point]

    @validator("points")
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
    def segments(self) -> list[LineSegment]:
        plist = self.points + [self.points[0]]
        return [
            LineSegment(points=(plist[i], plist[i + 1])) 
            for i in range(len(plist)-1)
        ]

    def __str__(self):
        # in PostGIS polygon has to begin and end at the same point
        pts = self.points
        if pts[0] != pts[-1]:
            pts = pts + [pts[0]]
        points_string = [
            f"({','.join([str(pt.x), str(pt.y)])})"
            for pt in pts
        ]
        return f"({','.join(points_string)})"

    def wkt(self, partial: bool = False) -> str:
        # in PostGIS polygon has to begin and end at the same point
        pts = self.points
        if pts[0] != pts[-1]:
            pts = pts + [pts[0]]
        points_string = [
            ' '.join([str(pt.x), str(pt.y)])
            for pt in pts
        ]
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
        plist = [
            polygon.wkt(partial=True)
            for polygon in self.polygons
        ]
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

    @validator("polygon")
    def valid_polygon(cls, v):
        if len(set(v.points)) != 4:
            raise ValueError("bounding box polygon requires exactly 4 unique points.")
            
    def is_rectangular(self):
        print(self.polygon)

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
    mask: str = Field(allow_mutation=False)
    shape: tuple[float, float]

    class Config:
        extra = Extra.allow
        validate_assignment = True

    @root_validator
    def correct_mask_shape(cls, values):
        def _mask_bytes_to_pil(mask_bytes):
            with io.BytesIO(mask_bytes) as f:
                return PIL.Image.open(f)

        mask_size = _mask_bytes_to_pil(b64decode(values["mask"])).size
        image_size = values["shape"]
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
            self._mask_bytes = b64decode(self.mask)
        return self._mask_bytes

    @property
    def pil_mask(self) -> PIL.Image:
        with io.BytesIO(self.mask_bytes) as f:
            return PIL.Image.open(f)


class GeoJSON(BaseModel):
    geometry_type: str
    coordinates: list

    @root_validator
    def validate_coordinates(cls, values):
        try:
            if values["geometry_type"] == "Polygon":
                for subpolygon in values["coordinates"]:
                    for coord in subpolygon:
                        assert len(coord) == 2
                        assert isinstance(coord[0], float | int)
                        assert isinstance(coord[1], float | int)
            elif values["geometry_type"] == "MultiPolygon":
                for polygon in values["coordinates"]:
                    for subpolygon in polygon:
                        for coord in polygon:
                            assert len(coord) == 2
                            assert isinstance(coord[0], float | int)
                            assert isinstance(coord[1], float | int)
        except:
            raise ValueError
        return values

    @classmethod
    def from_json(cls, geojson: str):
        data = json.loads(geojson)
        assert "type" in data
        assert "coordinates" in data
        return cls(geometry_type=data["type"], coordinates=data["coordinates"])

    def polygon(self) -> Polygon | None:
        if self.geometry_type != "Polygon":
            return None
        polygons = [
            BasicPolygon(
                points=[Point(x=coord[0], y=coord[1]) for coord in poly]
            )
            for poly in self.coordinates
        ]
        assert len(polygons) > 0
        return Polygon(
            boundary=polygons[0],
            holes=polygons[1:] if len(polygons) > 1 else None,
        )

    def multipolygon(self) -> MultiPolygon | None:
        if self.geometry_type != "MultiPolygon":
            return None

        multipolygons = []
        for subpolygon in self.coordinates:
            polygons = [
                BasicPolygon(
                    points=[Point(x=coord[0], y=coord[1]) for coord in poly]
                )
                for poly in subpolygon
            ]
            multipolygons.append(
                Polygon(
                    boundary=polygons[0],
                    holes=polygons[1:] if len(polygons) > 1 else None,
                )
            )
        assert len(multipolygons) > 0
        return MultiPolygon(polygons=multipolygons)
