import io
import math
from base64 import b64decode, b64encode

import numpy as np
import PIL.Image
from geoalchemy2.functions import (
    ST_AddBand,
    ST_AsRaster,
    ST_GeomFromText,
    ST_MakeEmptyRaster,
    ST_MapAlgebra,
    ST_SnapToGrid,
)
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import ScalarSelect, select


class Point(BaseModel):
    """
    Describes a point in geometric space.

    Attributes
    ----------
    x : float
        The x-coordinate of the point.
    y : float
        The y-coordinate of the point.

    Raises
    ------
    ValueError
        If an x or y-coordinate isn't passed.
    """

    x: float
    y: float

    @field_validator("x")
    @classmethod
    def _has_x(cls, v):
        """Validate that the object has a x-coordinate"""

        if not isinstance(v, float):
            raise ValueError
        return v

    @field_validator("y")
    @classmethod
    def _has_y(cls, v):
        """Validate that the object has a y-coordinate"""
        if not isinstance(v, float):
            raise ValueError
        return v

    def __str__(self) -> str:
        """Converts the object into a string."""
        return f"({self.x}, {self.y})"

    def __hash__(self) -> int:
        """Hashes the object"""
        return hash((self.x, self.y))

    def __eq__(self, other) -> bool:
        """
        Checks if the `Point` is close to another point using `math.isclose`.

        Parameters
        ----------
        other : Point
            The object to compare against.

        Returns
        ----------
        boolean
            A boolean describing whether the two objects are equal.

        Raises
        ----------
        TypeError
            If comparing an object of a different type.
        """
        if not isinstance(other, Point):
            raise TypeError
        return math.isclose(self.x, other.x) and math.isclose(self.y, other.y)

    def __neq__(self, other) -> bool:
        """
        Checks if the `Point` is not equal to another `Point`.

        Parameters
        ----------
        other : Point
            The object to compare against.

        Returns
        ----------
        boolean
            A boolean describing whether the two objects are equal.

        Raises
        ----------
        TypeError
            If comparing an object of a different type.
        """
        if not isinstance(other, Point):
            raise TypeError
        return not (self == other)

    def __neg__(self):
        """
        Return the inverse of the `Point` in coordinate space.

        Returns
        ----------
        Point
            A `Point` with inverse coordinates..
        """
        return Point(x=-self.x, y=-self.y)

    def __add__(self, other):
        """
        Add the coordinates of two `Points` and return a new `Point`.

        Parameters
        ----------
        other : Point
            The object to add.

        Returns
        ----------
        Point
            A `Point`.

        Raises
        ----------
        TypeError
            If adding an object of a different type.
        """
        if not isinstance(other, Point):
            raise TypeError
        newx = self.x + other.x
        newy = self.y + other.y
        return Point(x=newx, y=newy)

    def __sub__(self, other):
        """
        Subtract the coordinates of two `Points` and return a new `Point`.

        Parameters
        ----------
        other : Point
            The object to subtract.

        Returns
        ----------
        Point
            A `Point`.

        Raises
        ----------
        TypeError
            If subtracting an object of a different type.
        """
        if not isinstance(other, Point):
            raise TypeError
        newx = self.x - other.x
        newy = self.y - other.y
        return Point(x=newx, y=newy)

    def __iadd__(self, other):
        """
        Add the coordinates of two `Points` and return a new `Point`.

        Parameters
        ----------
        other : Point
            The object to add.

        Returns
        ----------
        Point
            A `Point`.

        Raises
        ----------
        TypeError
            If adding an object of a different type.
        """
        if not isinstance(other, Point):
            raise TypeError
        return self + other

    def __isub__(self, other):
        """
        Subtract the coordinates of two `Points` and return a new `Point`.

        Parameters
        ----------
        other : Point
            The object to subtract.

        Returns
        ----------
        Point
            A `Point`.

        Raises
        ----------
        TypeError
            If subtracting an object of a different type.
        """
        if not isinstance(other, Point):
            raise TypeError
        return self - other

    def dot(self, other):
        """
        Multiply the x and y-coordinates of two `Points`.

        Parameters
        ----------
        other : Point
            The object to subtract.

        Returns
        ----------
        Point
            A `Point`.

        Raises
        ----------
        TypeError
            If multiplying an object of a different type.
        """
        if not isinstance(other, Point):
            raise TypeError
        return (self.x * other.x) + (self.y * other.y)

    def wkt(self) -> str:
        """
        Returns the well-known text (WKT) representation of the object.

        Returns
        ----------
        str
            The WKT representation of the shape.
        """
        return f"POINT ({self.x} {self.y})"


class LineSegment(BaseModel):
    """
    Describes a line segment in geometric space.

    Attributes
    ----------
    points: Tuple[Point, Point]
        The coordinates of two points creating the line.
    """

    points: tuple[Point, Point]

    def delta_xy(self) -> Point:
        """
        Return the change in x and y over the start and end points of the line.

        Returns
        ----------
        Point
            A `Point` with the coordinates subtracted.
        """
        return self.points[0] - self.points[1]

    def parallel(self, other) -> bool:
        """
        Check whether two lines are parallel.

        Parameters
        ----------
        other : LineSegment
            The other line to compare against.

        Returns
        ----------
        bool
            Whether the lines are parallel.

        Raises
        ----------
        TypeError
            If other isn't of the correct type.
        """
        if not isinstance(other, LineSegment):
            raise TypeError

        d1 = self.delta_xy()
        d2 = other.delta_xy()

        slope1 = d1.y / d1.x if d1.x else math.inf
        slope2 = d2.y / d2.x if d2.x else math.inf
        return math.isclose(slope1, slope2)

    def perpendicular(self, other) -> bool:
        """
        Check whether two lines are perpendicular.

        Parameters
        ----------
        other : LineSegment
            The other line to compare against.

        Returns
        ----------
        bool
            Whether the lines are perpendicular.

        Raises
        ----------
        TypeError
            If other isn't of the correct type.
        """
        """Check whether two lines are perpendicular."""
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
    """
    Describes a polygon in geometric space.

    Attributes
    ----------
    points: List[Point]
        The coordinates of the geometry.

    Raises
    ------
    ValueError
        If less than three points are passed.
    """

    points: list[Point]

    @field_validator("points")
    @classmethod
    def _check_points(cls, v):
        if v is not None:
            if len(set(v)) < 3:
                raise ValueError(
                    "Polygon must be composed of at least three unique points."
                )
            # Remove duplicate of start point
            if v[0] == v[-1]:
                v = v[:-1]

        return v

    @property
    def left(self):
        """
        Returns the left-most point of the geometry.

        Returns
        ----------
        float | int
            A coordinate.
        """
        return min(self.points, key=lambda point: point.x).x

    @property
    def right(self):
        """
        Returns the right-most point of the geometry.

        Returns
        ----------
        float | int
            A coordinate.
        """
        return max(self.points, key=lambda point: point.x).x

    @property
    def top(self):
        """
        Returns the top-most point of the geometry.

        Returns
        ----------
        float | int
            A coordinate.
        """
        return max(self.points, key=lambda point: point.y).y

    @property
    def bottom(self):
        """
        Returns the bottom-most point of the geometry.

        Returns
        ----------
        float | int
            A coordinate.
        """
        return min(self.points, key=lambda point: point.y).y

    @property
    def width(self):
        """
        Returns the width of the geometry.

        Returns
        ----------
        float | int
            The width of the geometry.
        """
        return self.right - self.left

    @property
    def height(self):
        """
        Returns the height of the geometry.

        Returns
        ----------
        float | int
            The height of the geometry.
        """
        return self.top - self.bottom

    @property
    def segments(self) -> list[LineSegment]:
        """
        Returns a list of line segments for the polygon.

        Returns
        ----------
        List[LineSegment]
            A list of segments.
        """
        plist = self.points + [self.points[0]]
        return [
            LineSegment(points=(plist[i], plist[i + 1]))
            for i in range(len(plist) - 1)
        ]

    def __str__(self):
        """Converts the object to a string. In PostGIS, a polygon has to begin and end at the same point"""

        pts = self.points
        if pts[0] != pts[-1]:
            pts = pts + [pts[0]]
        points_string = [f"({','.join([str(pt.x), str(pt.y)])})" for pt in pts]
        return f"({','.join(points_string)})"

    def wkt(self, partial: bool = False) -> str:
        """
        Returns the well-known text (WKT) representation of the object.

        Parameters
        ----------
        partial : bool
            Whether to return the full WKT string or not.

        Returns
        ----------
        str
            The WKT representation of the shape.
        """
        # in PostGIS polygon has to begin and end at the same point
        pts = self.points
        if pts[0] != pts[-1]:
            pts = pts + [pts[0]]
        points_string = [" ".join([str(pt.x), str(pt.y)]) for pt in pts]
        wkt_format = f"({', '.join(points_string)})"
        if partial:
            return wkt_format
        return f"POLYGON ({wkt_format})"

    def offset(self, x: float = 0, y: float = 0):
        """
        Translates the geometry by an offset.

        Parameters
        ----------
        x : int, default=0
            The x-axis offset.
        y : int, default=0
            The y-axis offset.
        """
        self.points = [Point(x=pt.x + x, y=pt.y + y) for pt in self.points]


class Polygon(BaseModel):
    """
    Describes a polygon in geometric space.

    Attributes
    ----------
    boundary : BasicPolygon
        The polygon itself.
    holes : List[BasicPolygon]
        Any holes that exist within the polygon.
    """

    boundary: BasicPolygon
    holes: list[BasicPolygon] | None = Field(default=None)

    def __str__(self):
        """Converts the object to a string."""
        polys = [str(self.boundary)]
        if self.holes:
            for hole in self.holes:
                polys.append(str(hole))
        return f"({','.join(polys)})"

    @property
    def left(self):
        """
        Returns the left-most point of the geometry.

        Returns
        ----------
        float | int
            A coordinate.
        """
        return self.boundary.left

    @property
    def right(self):
        """
        Returns the right-most point of the geometry.

        Returns
        ----------
        float | int
            A coordinate.
        """
        return self.boundary.right

    @property
    def top(self):
        """
        Returns the top-most point of the geometry.

        Returns
        ----------
        float | int
            A coordinate.
        """
        return self.boundary.top

    @property
    def bottom(self):
        """
        Returns the bottom-most point of the geometry.

        Returns
        ----------
        float | int
            A coordinate.
        """
        return self.boundary.bottom

    @property
    def width(self):
        """
        Returns the width of the geometry.

        Returns
        ----------
        float | int
            The width of the geometry.
        """
        return self.boundary.width

    @property
    def height(self):
        """
        Returns the height of the geometry.

        Returns
        ----------
        float | int
            The height of the geometry.
        """
        return self.boundary.height

    def wkt(self, partial: bool = False) -> str:
        """
        Returns the well-known text (WKT) representation of the object.

        Parameters
        ----------
        partial : bool
            Whether to return the full WKT string or not.

        Returns
        ----------
        str
            The WKT representation of the shape.
        """
        polys = [self.boundary.wkt(partial=True)]
        if self.holes:
            for hole in self.holes:
                polys.append(hole.wkt(partial=True))
        wkt_format = f"({', '.join(polys)})"
        if partial:
            return wkt_format
        return f"POLYGON {wkt_format}"

    def offset(self, x: float = 0, y: float = 0):
        """
        Translates the geometry by an offset.

        Parameters
        ----------
        x : int, default=0
            The x-axis offset.
        y : int, default=0
            The y-axis offset.
        """
        self.boundary.offset(x, y)
        if self.holes:
            for idx in range(len(self.holes)):
                self.holes[idx].offset(x, y)


class MultiPolygon(BaseModel):
    """
    Describes a multipolygon in geometric space.

    Attributes
    ----------
    polygons: List[Polygon]
        A list of polygons that make up the `MultiPolygon`.

    Raises
    ------
    ValueError
        If less than three points are passed.
    """

    polygons: list[Polygon]

    @property
    def left(self):
        """
        Returns the left-most point of the geometry.

        Returns
        ----------
        float | int
            A coordinate.
        """
        return min([poly.left for poly in self.polygons])

    @property
    def right(self):
        """
        Returns the right-most point of the geometry.

        Returns
        ----------
        float | int
            A coordinate.
        """
        return max([poly.right for poly in self.polygons])

    @property
    def top(self):
        """
        Returns the top-most point of the geometry.

        Returns
        ----------
        float | int
            A coordinate.
        """
        return max([poly.top for poly in self.polygons])

    @property
    def bottom(self):
        """
        Returns the bottom-most point of the geometry.

        Returns
        ----------
        float | int
            A coordinate.
        """
        return min([poly.bottom for poly in self.polygons])

    @property
    def width(self):
        """
        Returns the width of the geometry.

        Returns
        ----------
        float | int
            The width of the geometry.
        """
        return self.right - self.left

    @property
    def height(self):
        """
        Returns the height of the geometry.

        Returns
        ----------
        float | int
            The height of the geometry.
        """
        return self.top - self.bottom

    def wkt(self) -> str:
        """
        Returns the well-known text (WKT) representation of the object`.

        Returns
        ----------
        str
            The WKT representation of the shape.
        """
        plist = [polygon.wkt(partial=True) for polygon in self.polygons]
        return f"MULTIPOLYGON ({', '.join(plist)})"

    def offset(self, x: float = 0, y: float = 0):
        """
        Translates the geometry by an offset.

        Parameters
        ----------
        x : int, default=0
            The x-axis offset.
        y : int, default=0
            The y-axis offset.
        """
        for idx in range(len(self.polygons)):
            self.polygons[idx].offset(x, y)


class BoundingBox(BaseModel):
    """
    Describes a bounding box in geometric space.

    Attributes
    ----------
    polygons: BasicPolygon
        A polygon describing the bounding box.

    Raises
    ------
    ValueError
        If the number of points != 4.
    """

    polygon: BasicPolygon

    @field_validator("polygon")
    @classmethod
    def _validate_polygon(cls, v):
        """Validates the number of points in the polygon."""
        if len(set(v.points)) != 4:
            raise ValueError(
                "bounding box polygon requires exactly 4 unique points."
            )
        return v

    @classmethod
    def from_extrema(cls, xmin: float, ymin: float, xmax: float, ymax: float):
        """
        Create a bounding box from extrema.

        Parameters
        ----------
        xmin: float
            The minimum x-coordinate.
        ymin: float
            The minimum y-coordinate.
        xmax: float
            The maximum x-coordinate.
        ymax: float
            The maximum y-coordinate.


        Returns
        ------
        BoundingBox
            The bounding box created from the extrema.
        """
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
        """
        Returns the left-most point of the geometry.

        Returns
        ----------
        float | int
            A coordinate.
        """
        return self.polygon.left

    @property
    def right(self):
        """
        Returns the right-most point of the geometry.

        Returns
        ----------
        float | int
            A coordinate.
        """
        return self.polygon.right

    @property
    def top(self):
        """
        Returns the top-most point of the geometry.

        Returns
        ----------
        float | int
            A coordinate.
        """
        return self.polygon.top

    @property
    def bottom(self):
        """
        Returns the bottom-most point of the geometry.

        Returns
        ----------
        float | int
            A coordinate.
        """
        return self.polygon.bottom

    @property
    def width(self):
        """
        Returns the width of the geometry.

        Returns
        ----------
        float | int
            The width of the geometry.
        """
        return self.polygon.width

    @property
    def height(self):
        """
        Returns the height of the geometry.

        Returns
        ----------
        float | int
            The height of the geometry.
        """
        return self.polygon.height

    def is_rectangular(self):
        """
        Asserts whether the bounding box is rectangular.

        Returns
        ----------
        bool
            Whether the polygon is rectangular or not.
        """
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
        """
        Asserts whether the bounding box is rotated.

        Returns
        ----------
        bool
            Whether the polygon is rotated or not.
        """
        # check if rectangular
        if not self.is_rectangular():
            return False

        # check if rotation exists by seeing if corners do not share values.
        x = set([p.x for p in self.polygon.points])
        y = set([p.y for p in self.polygon.points])
        return (len(x) != 2) and (len(y) != 2)

    def is_skewed(self):
        """
        Asserts whether the bounding box is skewed.

        Returns
        ----------
        bool
            Whether the polygon is skewed or not.
        """
        return not (self.is_rotated() or self.is_rectangular())

    def wkt(self) -> str:
        """
        Returns the well-known text (WKT) representation of the object.

        Returns
        ----------
        str
            The WKT representation of the shape.
        """
        return self.polygon.wkt()

    def offset(self, x: float = 0, y: float = 0):
        """
        Translates the geometry by an offset.

        Parameters
        ----------
        x : int, default=0
            The x-axis offset.
        y : int, default=0
            The y-axis offset.
        """
        self.polygon.offset(x, y)


class Raster(BaseModel):
    """
    Describes a raster in geometric space.

    Attributes
    ----------
    mask : str
        The mask describing the raster.
    geometry : BoundingBox | Polygon | MultiPolygon, optional
        Option to define raster by a geometry. Overrides the bitmask.

    Raises
    ------
    ValueError
        If the image format is not PNG.
        If the image mode is not binary.
    """

    mask: str = Field(frozen=True)
    geometry: BoundingBox | Polygon | MultiPolygon | None = None

    @field_validator("mask")
    @classmethod
    def _check_png_and_mode(cls, v):
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

    @classmethod
    def from_numpy(cls, mask: np.ndarray):
        """
        Create a mask from a numpy array.

        Parameters
        ----------
        mask : np:ndarray
            A numpy array.

        Returns
        ----------
        Raster
            The raster object.

        Raises
        ----------
        ValueError
            If the array has more than two dimensions.
            If the array contains non-boolean elements.
        """
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

    @classmethod
    def from_geometry(
        cls,
        geometry: BoundingBox | Polygon | MultiPolygon,
        height: int | float,
        width: int | float,
    ):
        """
        Create a Raster object from a geometry.

        Parameters
        ----------
        geometry : BoundingBox | Polygon | MultiPolygon
            Defines the bitmask as a geometry. Overrides any existing mask.
        height : int | float
            The intended height of the binary mask.
        width : int | float
            The intended width of the binary mask.

        Returns
        -------
        schemas.Raster
        """
        r = cls.from_numpy(np.full((int(height), int(width)), False))
        r.geometry = geometry
        return r

    def to_numpy(self) -> np.ndarray:
        """
        Convert the mask into an array.

        Returns
        ----------
        np.ndarray
            An array representing a mask.
        """
        mask_bytes = b64decode(self.mask)
        with io.BytesIO(mask_bytes) as f:
            img = PIL.Image.open(f)
            return np.array(img)

    @property
    def mask_bytes(self) -> bytes:
        """
        Serialize the mask into bytes.

        Returns
        ----------
        bytes
            A byte object.

        """
        if not hasattr(self, "_mask_bytes"):
            self._mask_bytes = b64decode(self.mask)
        return self._mask_bytes

    @property
    def array(self) -> np.ndarray:
        """
        Convert the mask into an array.

        Returns
        ----------
        np.ndarray
            An array representing a mask.

        """
        return self.to_numpy()

    @property
    def height(self) -> int:
        """
        Get the height of the raster.

        Returns
        -------
        int
            The height of the binary mask.
        """
        return self.array.shape[0]

    @property
    def width(self) -> int:
        """
        Get the width of the raster.

        Returns
        -------
        int
            The width of the binary mask.
        """
        return self.array.shape[1]

    def wkt(self) -> ScalarSelect | bytes:
        """
        Converts raster schema into a postgis-compatible type.

        Returns
        -------
        ScalarSelect | bytes
            A valid input to the models.Annotation.raster column.
        """
        if self.geometry:
            empty_raster = ST_AddBand(
                ST_MakeEmptyRaster(
                    self.width,
                    self.height,
                    0,  # upperleftx
                    0,  # upperlefty
                    1,  # scalex
                    1,  # scaley
                    0,  # skewx
                    0,  # skewy
                    0,  # srid
                ),
                "8BUI",
            )
            geom_raster = ST_AsRaster(
                ST_SnapToGrid(
                    ST_GeomFromText(self.geometry.wkt()),
                    1.0,
                ),
                1.0,  # scalex
                1.0,  # scaley
                "8BUI",  # pixeltype
                1,  # value
                0,  # nodataval
            )
            return select(
                ST_MapAlgebra(
                    empty_raster,
                    geom_raster,
                    "[rast2]",
                    "8BUI",
                    "UNION",
                )
            ).scalar_subquery()
        else:
            return self.mask_bytes
