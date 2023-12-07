import io
from base64 import b64decode, b64encode
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import PIL.Image


@dataclass
class Point:
    """
    Represents a point in 2D space.

    Parameters
    ----------
    x : Union[float, int]
        The x-coordinate of the point.
    y : Union[float, int]
        The y-coordinate of the point.

    Attributes
    ----------
    x : float
        The x-coordinate of the point.
    y : float
        The y-coordinate of the point.

    Raises
    ------
    TypeError
        If the coordinates are not of type `float` or convertible to `float`.
    """

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
        """
        Resize the point coordinates based on the scaling factors.

        Parameters
        ----------
        og_img_h : int
            Original image height.
        og_img_w : int
            Original image width.
        new_img_h : int
            New image height.
        new_img_w : int
            New image width.

        Returns
        -------
        Point
            Resized point based on the scaling factors.
        """
        h_factor, w_factor = new_img_h / og_img_h, new_img_w / og_img_w
        return Point(x=w_factor * self.x, y=h_factor * self.y)


@dataclass
class Box:
    """
    Represents a 2D box defined by minimum and maximum points.

    Parameters
    ----------
    min : Union[Point, dict]
        The minimum point of the box. Can be a `Point` object or a dictionary
        with keys 'x' and 'y' representing the coordinates.
    max : Union[Point, dict]
        The maximum point of the box. Can be a `Point` object or a dictionary
        with keys 'x' and 'y' representing the coordinates.

    Attributes
    ----------
    min : Point
        The minimum point of the box.
    max : Point
        The maximum point of the box.

    Raises
    ------
    ValueError
        If the x-coordinate of `min` is greater than the x-coordinate of `max`.
        If the y-coordinate of `min` is greater than the y-coordinate of `max`.
    """

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
    """
    Class for representing a bounding region.

    Attributes
    ----------
    points : List[Point], optional
        List of `Point` objects representing the vertices of the polygon. Defaults to an empty list.

    Raises
    ------
    TypeError
        If `points` is not a list or an element in `points` is not a `Point`.
    ValueError
        If the number of unique points in `points` is less than 3,
        making the BasicPolygon invalid.
    """

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

    def xy_list(self) -> List[Point]:
        """
        Returns a list of `Point` objects representing the vertices of the polygon.

        Returns
        -------
        List[Point]
            List of `Point` objects.
        """
        return self.points.copy()

    def tuple_list(self) -> List[Tuple[float, float]]:
        """
        Returns a list of points as tuples (x, y).

        Returns
        -------
        List[Tuple[float, float]]
            List of points as tuples.
        """
        return [(pt.x, pt.y) for pt in self.points]

    @property
    def xmin(self):
        """Minimum x-coordinate of the polygon."""
        return min(p.x for p in self.points)

    @property
    def ymin(self):
        """Minimum y-coordinate of the polygon."""
        return min(p.y for p in self.points)

    @property
    def xmax(self):
        """Maximum x-coordinate of the polygon."""
        return max(p.x for p in self.points)

    @property
    def ymax(self):
        """Maximum y-coordinate of the polygon."""
        return max(p.y for p in self.points)

    @classmethod
    def from_box(cls, box: Box):
        """
        Create a BasicPolygon from a Box.

        Parameters
        ----------
        box : Box
            The box to convert to a BasicPolygon.

        Returns
        -------
        BasicPolygon
            A BasicPolygon created from the provided Box.
        """
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
    """
    Represents a polygon with a boundary and optional holes.

    Parameters
    ----------
    boundary : BasicPolygon or dict
        The outer boundary of the polygon. Can be a `BasicPolygon` object or a
        dictionary with the necessary information to create a `BasicPolygon`.
    holes : List[BasicPolygon], optional
        List of holes inside the polygon. Defaults to an empty list.

    Raises
    ------
    TypeError
        If `boundary` is not a `BasicPolygon` or cannot be converted to one.
        If `holes` is not a list or an element in `holes` is not a `BasicPolygon`.
    """

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
    """
    Represents a bounding box defined by a 4-point polygon.

    Parameters
    ----------
    polygon : BasicPolygon or dict
        The 4-point polygon defining the bounding box. Can be a `BasicPolygon` object
        or a dictionary with the necessary information to create a `BasicPolygon`.

    Raises
    ------
    TypeError
        If `polygon` is not a `BasicPolygon` or cannot be converted to one.
    ValueError
        If the number of points in `polygon` is not equal to 4, making it invalid as a bounding box.
    """

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
        """
        Create a BoundingBox from extrema values.

        Parameters
        ----------
        xmin : float
            Minimum x-coordinate of the bounding box.
        xmax : float
            Maximum x-coordinate of the bounding box.
        ymin : float
            Minimum y-coordinate of the bounding box.
        ymax : float
            Maximum y-coordinate of the bounding box.

        Returns
        -------
        BoundingBox
            A BoundingBox created from the provided extrema values.
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
    def xmin(self):
        """Minimum x-coordinate of the bounding box."""
        return self.polygon.xmin

    @property
    def xmax(self):
        """Maximum x-coordinate of the bounding box."""
        return self.polygon.xmax

    @property
    def ymin(self):
        """Minimum y-coordinate of the bounding box."""
        return self.polygon.ymin

    @property
    def ymax(self):
        """Maximum y-coordinate of the bounding box."""
        return self.polygon.ymax


@dataclass
class MultiPolygon:
    """
    Represents a collection of polygons.

    Parameters
    ----------
    polygons : List[Polygon], optional
        List of `Polygon` objects. Defaults to an empty list.

    Raises
    ------
    TypeError
        If `polygons` is not a list or an element in `polygons` is not a `Polygon`.
    """

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
    """
    Represents a raster image or binary mask.

    Parameters
    ----------
    mask : str
        Base64-encoded string representing the raster mask.

    Raises
    ------
    TypeError
        If `mask` is not a string.
    """

    mask: str

    def __post_init__(self):
        if not isinstance(self.mask, str):
            raise TypeError("mask should be of type `str`")

    @classmethod
    def from_numpy(cls, mask: np.ndarray):
        """
        Create a Raster object from a NumPy array.

        Parameters
        ----------
        mask : np.ndarray
            The 2D binary array representing the mask.

        Returns
        -------
        Raster
            A Raster object created from the provided NumPy array.

        Raises
        ------
        ValueError
            If the input array is not 2D or not of dtype bool.
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

    def to_numpy(self) -> np.ndarray:
        """
        Convert the base64-encoded mask to a NumPy array.

        Returns
        -------
        np.ndarray
            A 2D binary array representing the mask.
        """
        mask_bytes = b64decode(self.mask)
        with io.BytesIO(mask_bytes) as f:
            img = PIL.Image.open(f)
            return np.array(img)
