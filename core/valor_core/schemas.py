import io
import json
import math
from base64 import b64decode, b64encode
from dataclasses import dataclass
from typing import Any

import numpy as np
import PIL.ImageDraw as ImageDraw
from PIL import Image
from valor_core import geometry


def _generate_type_error(received_value: Any, expected_type: str):
    """Raise a TypeError with a specific error string format."""
    raise TypeError(
        f"Expected value of type '{expected_type}', received value '{received_value}' with type '{type(received_value).__name__}'."
    )


def _validate_type_point(v: Any) -> None:
    """
    Validates geometric point values.

    Parameters
    ----------
    v : Any
        The value to validate.

    Raises
    ------
    TypeError
        If the value is not of type 'tuple' or 'list'.
    ValueError
        If the point is not an (x,y) position.
    """
    if not isinstance(v, (tuple, list)):
        _generate_type_error(v, "tuple[float, float] or list[float]")
    elif not (
        len(v) == 2
        and isinstance(v[0], (int, float, np.number))
        and isinstance(v[1], (int, float, np.number))
    ):
        raise TypeError(
            f"Expected point to have two numeric values representing an (x, y) pair. Received '{v}'."
        )


def _validate_type_multipoint(v: Any) -> None:
    """
    Validates geometric multipoint values.

    Parameters
    ----------
    v : Any
        The value to validate.

    Raises
    ------
    TypeError
        If the value is not of type 'list'.
    ValueError
        If there are no points or they are not (x,y) positions.
    """
    if not isinstance(v, list):
        _generate_type_error(
            v, "list[tuple[float, float]] or list[list[float]]"
        )
    elif not v:
        raise TypeError("List cannot be empty.")
    for point in v:
        _validate_type_point(point)


def _validate_type_linestring(v: Any) -> None:
    """
    Validates geometric linestring values.

    Parameters
    ----------
    v : Any
        The value to validate.

    Raises
    ------
    TypeError
        If the value is not of type 'list'.
    ValueError
        If the value does not conform to the linestring requirements.
    """
    _validate_type_multipoint(v)
    if len(v) < 2:
        raise TypeError(f"A line requires two or more points. Received '{v}'.")


def _validate_type_multilinestring(v: Any) -> None:
    """
    Validates geometric multilinestring values.

    Parameters
    ----------
    v : Any
        The value to validate.

    Raises
    ------
    TypeError
        If the value is not of type 'list'.
    ValueError
        If the value does not conform to the multilinestring requirements.
    """
    if not isinstance(v, list):
        return _generate_type_error(
            v, "list[list[tuple[float, float]]] or list[list[list[float]]]"
        )
    elif not v:
        raise ValueError("List cannot be empty.")
    for line in v:
        _validate_type_linestring(line)


def _validate_type_polygon(v: Any) -> None:
    """
    Validates geometric polygon values.

    Parameters
    ----------
    v : Any
        The value to validate.

    Raises
    ------
    TypeError
        If the value is not of type 'list'.
    ValueError
        If the value does not conform to the polygon requirements.
    """
    if not isinstance(v, list):
        raise TypeError("Expected value to be a list.")

    _validate_type_multilinestring(v)
    for line in v:
        if not (len(line) >= 4 and line[0] == line[-1]):
            raise ValueError(
                "A polygon is defined by a list of at least four points with the first and last points being equal."
            )


def _validate_type_box(v: Any) -> None:
    """
    Validates geometric box values.

    Parameters
    ----------
    v : Any
        The value to validate.

    Raises
    ------
    TypeError
        If the value is not of type 'list'.
    ValueError
        If the value does not conform to the box requirements.
    """
    _validate_type_polygon(v)
    if not (len(v) == 1 and len(v[0]) == 5 and v[0][0] == v[0][-1]):
        raise ValueError(
            "Boxes are defined by five points with the first and last being equal."
        )

    if geometry.is_skewed(v[0]):
        raise NotImplementedError("Skewed boxes are not implemented yet.")


def _validate_geojson(geojson: dict) -> None:
    """
    Validates that a dictionary conforms to the GeoJSON geometry specification.

    Parameters
    ----------
    geojson: dict
        The dictionary to validate.

    Raises
    ------
    TypeError
        If the passed in value is not a dictionary.
        If the GeoJSON 'type' attribute is not supported.
    ValueError
        If the dictionary does not conform to the GeoJSON format.
    """
    map_str_to_geojson_validator = {
        "point": _validate_type_point,
        "multipoint": _validate_type_multipoint,
        "linestring": _validate_type_linestring,
        "multilinestring": _validate_type_multilinestring,
        "polygon": _validate_type_polygon,
    }
    # validate geojson
    if not isinstance(geojson, dict):
        raise TypeError(
            f"Expected a GeoJSON dictionary as input, received '{geojson}'."
        )
    elif not (
        set(geojson.keys()) == {"type", "coordinates"}
        and (geometry_type := geojson.get("type"))
        and (geometry_value := geojson.get("coordinates"))
    ):
        raise ValueError(
            f"Expected geojson to be a dictionary with keys 'type' and 'coordinates'. Received value '{geojson}'."
        )

    # validate type
    geometry_type = geometry_type.lower()
    if geometry_type not in map_str_to_geojson_validator:
        raise TypeError(
            f"Class '{geometry_type}' is not a supported GeoJSON geometry type."
        )

    # validate coordinates
    try:
        map_str_to_geojson_validator[geometry_type](geometry_value)
    except (ValueError, ValueError) as e:
        raise ValueError(
            f"Value does not conform to '{geometry_type}'. Validation error: {str(e)}"
        )


@dataclass
class Point:
    """
    Describes a Point in (x,y) coordinates.

    Attributes
    ----------
    value : tuple[int | float, int | float]
        A list of coordinates describing the Point.

    Raises
    ------
    ValueError
        If the value doesn't conform to the type.
    """

    value: tuple[int | float, int | float]

    def __post_init__(self):
        """Validate instantiated class."""

        _validate_type_point(self.value)

    @classmethod
    def from_dict(cls, geojson: dict) -> "Point":
        """
        Create a Point from a GeoJSON in dictionary format.

        Parameters
        ----------
        geojson: dict[str, str | list[int | float]]
            A Point value in GeoJSON format.
        """
        geometry = GeoJSON(**geojson).geometry
        if not isinstance(geometry, Point):
            raise TypeError(f"GeoJSON is for a different type '{geojson}'.")
        return geometry

    def to_dict(self) -> dict[str, str | list[int | float]]:
        """
        Create a dictionary that represents the Point in GeoJSON format.

        Returns
        ----------
        dict[str, str | list[int | float]]
            A Point value in GeoJSON format.
        """
        return {"type": "Point", "coordinates": list(self.value)}

    @classmethod
    def from_json(cls, geojson: str) -> "Point":
        """
        Create a Point from a GeoJSON in json format.

        Parameters
        ----------
        geojson: str
            A Point value in GeoJSON format.
        """
        return cls.from_dict(json.loads(geojson))

    def to_json(self) -> str:
        """
        Create a json string that represents the Point in GeoJSON format.

        Returns
        ----------
        str
            A Point value in GeoJSON format.
        """
        return json.dumps(self.to_dict())

    def to_wkt(self) -> str:
        """
        Casts the geometric object into a string using Well-Known-Text (WKT) Format.

        Returns
        -------
        str
            The WKT formatted string.
        """
        return f"POINT ({self.value[0]} {self.value[1]})"

    def resize(
        self,
        og_img_h=10,
        og_img_w=10,
        new_img_h=100,
        new_img_w=100,
    ):
        h_ratio = new_img_h / og_img_h
        w_ratio = new_img_w / og_img_w
        return Point((self.value[0] * h_ratio, self.value[1] * w_ratio))

    @property
    def x(self):
        return self.value[0]

    @property
    def y(self):
        return self.value[1]

    def __hash__(self):
        return hash(str([float(x) for x in self.value]))


@dataclass
class MultiPoint:
    """
    Describes a MultiPoint in (x,y) coordinates.

    Attributes
    ----------
    value : list[tuple[int | float, int | float]]
        A list of coordinates describing the MultiPoint.

    Raises
    ------
    ValueError
        If the value doesn't conform to the type.
    """

    value: list[tuple[int | float, int | float]]

    def __post_init__(self):
        """Validate instantiated class."""

        _validate_type_multipoint(self.value)

    @classmethod
    def from_dict(cls, geojson: dict) -> "MultiPoint":
        """
        Create a MultiPoint from a GeoJSON in dictionary format.

        Parameters
        ----------
        geojson: dict[str, str | list[list[int | float]]]
            A MultiPoint value in GeoJSON format.
        """
        geometry = GeoJSON(**geojson).geometry
        if not isinstance(geometry, MultiPoint):
            raise TypeError(f"GeoJSON is for a different type '{geojson}'.")
        return geometry

    def to_dict(self) -> dict[str, str | list[list[int | float]]]:
        """
        Create a dictionary that represents the MultiPoint in GeoJSON format.

        Returns
        ----------
        dict[str, str | list[list[int | float]]]
            A MultiPoint value in GeoJSON format.
        """
        return {
            "type": "MultiPoint",
            "coordinates": [list(point) for point in self.value],
        }

    @classmethod
    def from_json(cls, geojson: str) -> "MultiPoint":
        """
        Create a MultiPoint from a GeoJSON in json format.

        Parameters
        ----------
        geojson: str
            A MultiPoint value in GeoJSON format.
        """
        return cls.from_dict(json.loads(geojson))

    def to_json(self) -> str:
        """
        Create a json string that represents the MultiPoint in GeoJSON format.

        Returns
        ----------
        str
            A MultiPoint value in GeoJSON format.
        """
        return json.dumps(self.to_dict())

    def to_wkt(self) -> str:
        """
        Casts the geometric object into a string using Well-Known-Text (WKT) Format.

        Returns
        -------
        str
            The WKT formatted string.
        """
        points = ", ".join(
            [f"({point[0]} {point[1]})" for point in self.value]
        )
        return f"MULTIPOINT ({points})"


@dataclass
class LineString:
    """
    Describes a LineString in (x,y) coordinates.

    Attributes
    ----------
    value : list[tuple[int | float, int | float]]
        A list of coordinates describing the LineString.

    Raises
    ------
    ValueError
        If the value doesn't conform to the type.
    """

    value: list[tuple[int | float, int | float]]

    def __post_init__(self):
        """Validate instantiated class."""

        _validate_type_linestring(self.value)

    @classmethod
    def from_dict(cls, geojson: dict) -> "LineString":
        """
        Create a LineString from a GeoJSON in dictionary format.

        Parameters
        ----------
        geojson: dict[str, str | list[list[int | float]]]
            A LineString value in GeoJSON format.
        """
        geometry = GeoJSON(**geojson).geometry
        if not isinstance(geometry, LineString):
            raise TypeError(f"GeoJSON is for a different type '{geojson}'.")
        return geometry

    def to_dict(self) -> dict[str, str | list[list[int | float]]]:
        """
        Create a dictionary that represents the LineString in GeoJSON format.

        Returns
        ----------
        dict[str, str | list[list[int | float]]]
            A LineString value in GeoJSON format.
        """
        return {
            "type": "LineString",
            "coordinates": [list(point) for point in self.value],
        }

    @classmethod
    def from_json(cls, geojson: str) -> "LineString":
        """
        Create a LineString from a GeoJSON in json format.

        Parameters
        ----------
        geojson: str
            A LineString value in GeoJSON format.
        """
        return cls.from_dict(json.loads(geojson))

    def to_json(self) -> str:
        """
        Create a json string that represents the LineString in GeoJSON format.

        Returns
        ----------
        str
            A LineString value in GeoJSON format.
        """
        return json.dumps(self.to_dict())

    def to_wkt(self) -> str:
        """
        Casts the geometric object into a string using Well-Known-Text (WKT) Format.

        Returns
        -------
        str
            The WKT formatted string.
        """
        points = ", ".join([f"{point[0]} {point[1]}" for point in self.value])
        return f"LINESTRING ({points})"


@dataclass
class MultiLineString:
    """
    Describes a MultiLineString in (x,y) coordinates.

    Attributes
    ----------
    value : list[list[tuple[int | float, int | float]]]
        A list of coordinates describing the MultiLineString.

    Raises
    ------
    ValueError
        If the value doesn't conform to the type.
    """

    value: list[list[tuple[int | float, int | float]]]

    def __post_init__(self):
        """Validate instantiated class."""

        _validate_type_multilinestring(self.value)

    @classmethod
    def from_dict(cls, geojson: dict) -> "MultiLineString":
        """
        Create a MultiLineString from a GeoJSON in dictionary format.

        Parameters
        ----------
        geojson: dict[str, str | list[list[list[int | float]]]]
            A MultiLineString value in GeoJSON format.
        """
        geometry = GeoJSON(**geojson).geometry
        if not isinstance(geometry, MultiLineString):
            raise TypeError(f"GeoJSON is for a different type '{geojson}'.")
        return geometry

    def to_dict(self) -> dict[str, str | list[list[list[int | float]]]]:
        """
        Create a dictionary that represents the MultiLineString in GeoJSON format.

        Returns
        ----------
        dict[str, str | list[list[list[int | float]]]]
            A MultiLineString value in GeoJSON format.
        """
        return {
            "type": "MultiLineString",
            "coordinates": [
                [list(point) for point in line] for line in self.value
            ],
        }

    @classmethod
    def from_json(cls, geojson: str) -> "MultiLineString":
        """
        Create a MultiLineString from a GeoJSON in json format.

        Parameters
        ----------
        geojson: str
            A MultiLineString value in GeoJSON format.
        """
        return cls.from_dict(json.loads(geojson))

    def to_json(self) -> str:
        """
        Create a json string that represents the MultiLineString in GeoJSON format.

        Returns
        ----------
        str
            A MultiLineString value in GeoJSON format.
        """
        return json.dumps(self.to_dict())

    def to_wkt(self) -> str:
        """
        Casts the geometric object into a string using Well-Known-Text (WKT) Format.

        Returns
        -------
        str
            The WKT formatted string.
        """
        points = "),(".join(
            [
                ", ".join([f"{point[0]} {point[1]}" for point in line])
                for line in self.value
            ]
        )
        return f"MULTILINESTRING (({points}))"


@dataclass
class Polygon:
    """
    Describes a Polygon in (x,y) coordinates.

    Attributes
    ----------
    value : list[list[tuple[int | float, int | float]]]
        A list of coordinates describing the Box.

    Raises
    ------
    ValueError
        If the value doesn't conform to the type.
    """

    value: list[list[tuple[int, int]]] | list[list[tuple[float, float]]]

    def __post_init__(self):
        """Validate instantiated class."""

        if not (
            isinstance(self.value, list)
            and len(self.value) > 0
            and isinstance(self.value[0], list)
        ):
            raise TypeError("Expected list of lists.")
        _validate_type_polygon(self.value)

    @classmethod
    def from_dict(cls, geojson: dict) -> "Polygon":
        """
        Create a Polygon from a GeoJSON in dictionary format.

        Parameters
        ----------
        geojson: dict[str, str | list[list[list[int | float]]]]
            A Polygon value in GeoJSON format.
        """
        geometry = GeoJSON(**geojson).geometry
        if not isinstance(geometry, Polygon):
            raise TypeError(f"GeoJSON is for a different type '{geojson}'.")
        return geometry

    def to_dict(self) -> dict[str, str | list[list[list[int | float]]]]:
        """
        Create a dictionary that represents the Polygon in GeoJSON format.

        Returns
        ----------
        dict[str, str | list[list[list[int | float]]]]
            A Polygon value in GeoJSON format.
        """
        return {
            "type": "Polygon",
            "coordinates": [
                [list(point) for point in subpolygon]
                for subpolygon in self.value
            ],
        }

    @classmethod
    def from_json(cls, geojson: str) -> "Polygon":
        """
        Create a Polygon from a GeoJSON in json format.

        Parameters
        ----------
        geojson: str
            A Polygon value in GeoJSON format.
        """
        return cls.from_dict(json.loads(geojson))

    def to_json(self) -> str:
        """
        Create a json string that represents the Polygon in GeoJSON format.

        Returns
        ----------
        str
            A Polygon value in GeoJSON format.
        """
        return json.dumps(self.to_dict())

    def to_wkt(self) -> str:
        """
        Casts the geometric object into a string using Well-Known-Text (WKT) Format.

        Returns
        -------
        str
            The WKT formatted string.
        """
        coords = "),(".join(
            [
                ", ".join([f"{point[0]} {point[1]}" for point in subpolygon])
                for subpolygon in self.value
            ]
        )
        return f"POLYGON (({coords}))"

    @property
    def boundary(self):
        """
        The boundary of the polygon.

        Returns
        -------
        list[Tuple(float, float)]
            A list of points.
        """
        value = self.value
        if value is None:
            raise ValueError("Polygon is 'None'")
        return value[0]

    @property
    def holes(self):
        """
        Any holes in the polygon.

        Returns
        -------
        list[list[Tuple(float, float)]]
            A list of holes.
        """
        value = self.value
        if value is None:
            raise ValueError("Polygon is 'None'")
        return value[1:]

    @property
    def xmin(self) -> float:
        """
        Minimum x-value.

        Returns
        -------
        float
        """
        return min([p[0] for p in self.boundary])

    @property
    def xmax(self) -> float:
        """
        Maximum x-value.

        Returns
        -------
        float
        """
        return max([p[0] for p in self.boundary])

    @property
    def ymin(self) -> float:
        """
        Minimum y-value.

        Returns
        -------
        float
        """
        return min([p[1] for p in self.boundary])

    @property
    def ymax(self) -> float:
        """
        Maximum y-value.

        Returns
        -------
        float
        """
        return max([p[1] for p in self.boundary])

    def to_array(self) -> np.ndarray:
        """
        Convert Polygon to an array.

        Returns
        -------
        np.ndarray
        """
        return np.array(self.value[0])

    def to_coordinates(self) -> list[list[dict[str, int | float]]]:
        """
        Convert Polygon to a nested list of coordinates.

        Returns
        -------
        np.ndarray
        """
        return [[{"x": points[0], "y": points[1]} for points in self.value[0]]]


@dataclass
class Box:
    """
    Describes a Box in (x,y) coordinates.

    Attributes
    ----------
    value : list[list[tuple[int | float, int | float]]]
        A list of coordinates describing the Box.

    Raises
    ------
    ValueError
        If the value doesn't conform to the type.
    """

    value: list[list[tuple[int, int]]] | list[list[tuple[float, float]]]

    def __post_init__(self):
        """Validate instantiated class."""

        _validate_type_box(self.value)

    @classmethod
    def from_extrema(
        cls,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
    ):
        """
        Create a box from extrema.

        Parameters
        ----------
        xmin: float
            The minimum x-coordinate.
        xmax: float
            The maximum x-coordinate.
        ymin: float
            The minimum y-coordinate.
        ymax: float
            The maximum y-coordinate.
        """
        if xmin >= xmax or ymin >= ymax:
            raise ValueError(
                "Minimums cannot be greater-than or equal to maximums."
            )
        return cls(
            value=[
                [
                    (xmin, ymin),
                    (xmax, ymin),
                    (xmax, ymax),
                    (xmin, ymax),
                    (xmin, ymin),
                ]
            ]
        )

    @classmethod
    def from_dict(cls, geojson: dict) -> "Box":
        """
        Create a Box from a GeoJSON Polygon in dictionary format.

        Parameters
        ----------
        geojson: dict[str, str | list[list[list[int | float]]]]
            A Polygon value in GeoJSON format.
        """
        return cls(value=Polygon.from_dict(geojson).value)

    def to_dict(self) -> dict[str, str | list[list[list[int | float]]]]:
        """
        Create a dictionary that represents the Box using a GeoJSON Polygon.

        Returns
        ----------
        dict[str, str | list[list[list[int | float]]]]
            A Polygon value in GeoJSON format.
        """
        return Polygon(value=self.value).to_dict()

    @classmethod
    def from_json(cls, geojson: str) -> "Box":
        """
        Create a Box from a GeoJSON Polygon in json format.

        Parameters
        ----------
        geojson: str
            A Polygon value in GeoJSON format.
        """
        return cls.from_dict(json.loads(geojson))

    def to_json(self) -> str:
        """
        Create a json string that represents the Box using a GeoJSON Polygon.

        Returns
        ----------
        str
            A Polygon value in GeoJSON format.
        """
        return json.dumps(self.to_dict())

    def to_wkt(self) -> str:
        """
        Casts the geometric object into a string using Well-Known-Text (WKT) Format.

        Note that 'Box' is not a supported geometry so the output will use the format for 'Polygon'.

        Returns
        -------
        str
            The WKT formatted string.
        """
        return Polygon(value=self.value).to_wkt()

    def to_array(
        self,
    ) -> np.ndarray:
        """
        Convert Box to an array.

        Returns
        -------
        np.ndarray
        """
        return np.array(self.value[0])

    def to_coordinates(self) -> list[list[dict[str, int | float]]]:
        """
        Convert Polygon to a nested list of coordinates.

        Returns
        -------
        np.ndarray
        """
        return [[{"x": points[0], "y": points[1]} for points in self.value[0]]]

    @property
    def xmin(self):
        return min([point[0] for point in self.value[0]])

    @property
    def xmax(self):
        return max([point[0] for point in self.value[0]])

    @property
    def ymin(self):
        return min([point[1] for point in self.value[0]])

    @property
    def ymax(self):
        return max([point[1] for point in self.value[0]])


@dataclass
class GeoJSON:
    type: str
    coordinates: (
        list[float]
        | list[list[float]]
        | list[list[list[float]]]
        | list[list[list[list[float]]]]
    )

    def __post_init__(self):
        """Validate instantiated class."""

        _validate_geojson({"type": self.type, "coordinates": self.coordinates})

    @property
    def geometry(
        self,
    ) -> Point | MultiPoint | LineString | MultiLineString | Polygon:
        map_str_to_type = {
            "Point": Point,
            "MultiPoint": MultiPoint,
            "LineString": LineString,
            "MultiLineString": MultiLineString,
            "Polygon": Polygon,
        }
        return map_str_to_type[self.type](value=self.coordinates)

    def to_wkt(self) -> str:
        """
        Converts the GeoJSON to a string in Well-Known-Text (WKT) formatting.

        Returns
        -------
        str
            The geometry in WKT format.
        """
        return self.geometry.to_wkt()


@dataclass
class Raster:
    """
    Represents a binary mask.

    Parameters
    ----------
    value : dict[str, np.ndarray | str | None], optional
        An raster value.

    Attributes
    ----------
    area
    array
    geometry
    height
    width

    Raises
    ------
    TypeError
        If `encoding` is not a string.

    Examples
    --------
    Generate a random mask.
    >>> import numpy.random
    >>> height = 640
    >>> width = 480
    >>> array = numpy.random.rand(height, width)

    Convert to binary mask.
    >>> mask = (array > 0.5)

    Create Raster.
    >>> Raster(mask)
    """

    mask: np.ndarray

    def __post_init__(self):
        """Validate instantiated class."""

        if not isinstance(self.mask, np.ndarray):
            raise TypeError(
                "Raster should contain a numpy array describing the Raster mask."
            )
        if len(self.mask.shape) != 2:
            raise ValueError("raster only supports 2d arrays")

        if self.mask is not None and self.mask.dtype != bool:
            raise ValueError(
                f"Expecting a binary mask (i.e. of dtype bool) but got dtype {self.mask.dtype}"
            )

    def encode_value(self) -> Any:
        """Encode object to JSON compatible dictionary."""
        value = self.mask
        if value is None:
            return None

        if self.mask is not None:
            f = io.BytesIO()
            Image.fromarray(self.mask).save(f, format="PNG")
            f.seek(0)
            mask_bytes = f.read()
            f.close()
            decoded_mask_bytes = b64encode(mask_bytes).decode()
        else:
            decoded_mask_bytes = None
        return {
            "mask": decoded_mask_bytes,
        }

    @classmethod
    def decode_value(cls, mask: Any):
        """Decode object from JSON compatible dictionary."""
        mask_bytes = b64decode(mask)
        with io.BytesIO(mask_bytes) as f:
            img = Image.open(f)
            value = np.array(img)

        return cls(mask=value)

    def to_array(self) -> np.ndarray:
        """
        Convert Raster to a numpy array.

        Returns
        -------
        np.ndarray | None
            A 2D binary array representing the mask if it exists.
        """
        return self.mask

    @classmethod
    def from_coordinates(
        cls,
        coordinates: list[list[dict[str, int]]] | list[list[dict[str, float]]],
        height: int,
        width: int,
    ):
        """
        Create a Raster object from coordinates.

        Parameters
        ----------
        coordinates : list[list[dict[str, int]]]
            Defines the bitmask as a nested list of coordinates.
        height : int
            The intended height of the binary mask.
        width : int
            The intended width of the binary mask.

        Returns
        -------
        schemas.Raster
        """
        if not (isinstance(coordinates, list)):
            raise TypeError(
                "coordinates should either be an empty list, or it should be a list of lists containing dictionaries with 'x' and 'y' keys."
            )

        if len(coordinates) > 0 and not (
            isinstance(coordinates[0], list)
            and len(coordinates[0]) > 0
            and isinstance(coordinates[0][0], dict)
            and all(
                all(set(pt.keys()) == {"x", "y"} for pt in contour)
                for contour in coordinates
            )
        ):
            raise TypeError(
                "Coordinates should either be an empty list, or it should be a list of lists containing dictionaries with 'x' and 'y' keys."
            )

        if not (
            all(
                all(pt["x"] >= 0 and pt["y"] >= 0 for pt in contour)
                for contour in coordinates
            )
        ):
            raise ValueError(
                "Coordinates cannot be negative when converting to a raster."
            )

        contours = [
            [(min(pt["x"], width), min(pt["y"], height)) for pt in contour]
            for contour in coordinates
        ]

        img = Image.new("1", (width, height), 0)

        for contour in contours:
            if len(contour) >= 2:
                ImageDraw.Draw(img).polygon(contour, outline=1, fill=1)

        return cls(np.array(img))

    @classmethod
    def from_geometry(cls, geometry: Box | Polygon, height: int, width: int):
        """
        Create a Raster object from a geometry.

        Parameters
        ----------
        coordinates : list[list[dict[str, int]]]
            Defines the bitmask as a nested list of coordinates.
        height : int
            The intended height of the binary mask.
        width : int
            The intended width of the binary mask.

        Returns
        -------
        schemas.Raster
        """
        if not (isinstance(geometry, Box) or isinstance(geometry, Polygon)):
            raise TypeError("Geometry should be a Box or Polygon.")

        return cls.from_coordinates(
            geometry.to_coordinates(), height=height, width=width
        )


@dataclass
class Embedding:
    """
    Represents a model embedding.

    Parameters
    ----------
    value : list[float], optional
        An embedding value.
    """

    value: list[int] | list[float] | None = None

    def __post_init__(self):
        """Validate instantiated class."""

        if not isinstance(self.value, list):
            raise TypeError(
                f"Expected type 'list[float] | None' received type '{type(self.value)}'"
            )
        elif len(self.value) < 1:
            raise ValueError("embedding should have at least one dimension")


@dataclass
class Datum:
    """
    A class used to store information about a datum for either a 'GroundTruth' or a 'Prediction'.

    Attributes
    ----------
    uid : str
        The UID of the datum.
    text : str, optional
        If the datum is a piece of text, then this field should contain the text.
    metadata : dict[str, Any]
        A dictionary of metadata that describes the datum.

    Examples
    --------
    >>> Datum(uid="uid1")
    >>> Datum(uid="uid1", metadata={})
    >>> Datum(uid="uid1", metadata={"foo": "bar", "pi": 3.14})
    >>> Datum(uid="uid2", text="What is the capital of Kenya?")
    """

    uid: str | None = None
    text: str | None = None
    metadata: dict | None = None

    def __post_init__(
        self,
    ):
        """Validate instantiated class."""

        if not isinstance(self.uid, (str, type(None))):
            raise TypeError(
                f"Expected 'uid' to be of type 'str' or 'None', got {type(self.uid).__name__}"
            )
        if not isinstance(self.text, (str, type(None))):
            raise TypeError(
                f"Expected 'text' to be of type 'str' or 'None', got {type(self.text).__name__}"
            )
        if not isinstance(self.metadata, (dict, type(None))):
            raise TypeError(
                f"Expected 'metadata' to be of type 'dict' or 'None', got {type(self.metadata).__name__}"
            )


@dataclass
class Label:
    """
    An object for labeling datasets, models, and annotations.

    Attributes
    ----------
    key : str
        The label key. (e.g. 'class', 'category')
    value : str
        The label's value. (e.g. 'dog', 'cat')
    score : float, optional
        A score assigned to the label in the case of a prediction.
    """

    key: str
    value: str
    score: float | None = None

    def __post_init__(self):
        """Validate instantiated class."""

        if not isinstance(self.key, str):
            raise TypeError(
                f"Expected 'key' to be of type 'str', got {type(self.key).__name__}"
            )

        if not isinstance(self.value, str):
            raise TypeError(
                f"Expected 'value' to be of type 'str', got {type(self.value).__name__}"
            )

        if self.score is not None and not isinstance(
            self.score,
            (
                float,
                int,
            ),
        ):
            raise TypeError(
                f"Expected 'score' to be of type 'float' or 'int' or 'None', got {type(self.score).__name__}"
            )

        # Ensure score is a float if provided as int
        if isinstance(self.score, int):
            self.score = float(self.score)

    def __eq__(self, other):
        """
        Defines how labels are compared to one another.

        Parameters
        ----------
        other : Label
            The object to compare with the label.

        Returns
        ----------
        bool
            A boolean describing whether the two objects are equal.
        """
        if (
            not hasattr(other, "key")
            or not hasattr(other, "key")
            or not hasattr(other, "score")
        ):
            return False

        # if the scores aren't the same type return False
        if (other.score is None) != (self.score is None):
            return False

        if self.score is None or other.score is None:
            scores_equal = other.score is None and self.score is None
        else:
            scores_equal = math.isclose(self.score, other.score)

        return (
            scores_equal
            and self.key == other.key
            and self.value == other.value
        )

    def __hash__(self) -> int:
        """
        Defines how a 'Label' is hashed.

        Returns
        ----------
        int
            The hashed 'Label'.
        """
        return hash(f"key:{self.key},value:{self.value},score:{self.score}")


@dataclass
class Annotation:
    """
    A class used to annotate `GroundTruths` and `Predictions`.

    Attributes
    ----------
    metadata: dict[str, Any]
        A dictionary of metadata that describes the `Annotation`.
    labels: list[Label], optional
        A list of labels to use for the `Annotation`.
    bounding_box: schemas.Box
        A bounding box to assign to the `Annotation`.
    polygon: BoundingPolygon
        A polygon to assign to the `Annotation`.
    raster: Raster
        A raster to assign to the `Annotation`.
    embedding: list[float]
        An embedding, described by a list of values with type float and a maximum length of 16,000.
    is_instance: bool, optional
        A boolean describing whether we should treat the Raster attached to an annotation as an instance segmentation or not. If set to true, then the Annotation will be validated for use in object detection tasks. If set to false, then the Annotation will be validated for use in semantic segmentation tasks.
    implied_task_types: list[str], optional
        The validated task types that are applicable to each Annotation. Doesn't need to bet set by the user.
    text: str, optional
        A piece of text to assign to the 'Annotation'.
    context_list: list[str], optional
        A list of contexts to assign to the 'Annotation'.

    Examples
    --------

    Classification
    >>> Annotation.create(
    ...     labels=[
    ...         Label(key="class", value="dog"),
    ...         Label(key="category", value="animal"),
    ...     ]
    ... )

    Object-Detection schemas.Box
    >>> annotation = Annotation(
    ...     labels=[Label(key="k1", value="v1")],
    ...     bounding_box=box2,
    ... )

    Object-Detection schemas.Polygon
    >>> annotation = Annotation(
    ...     labels=[Label(key="k1", value="v1")],
    ...     polygon=BoundingPolygon(...),
    ... )

     Raster
    >>> annotation = Annotation(
    ...     labels=[Label(key="k1", value="v1")],
    ...     raster=Raster(...),
    ...     is_instance=True
    ... )

    Object-Detection with all supported Geometries defined.
    >>> Annotation(
    ...     labels=[Label(key="k1", value="v1")],
    ...     bounding_box=schemas.Box(...),
    ...     polygon=BoundingPolygon(...),
    ...     raster=Raster(...),
    ...     is_instance=True,
    ... )

    Semantic-Segmentation Raster
    >>> annotation = Annotation(
    ...     labels=[Label(key="k1", value="v1")],
    ...     raster=Raster(...),
    ...     is_instance=False # or None
    ... )

    Text Generation Annotation with text and context_list. Not all text generation tasks require both text and context.
    >>> annotation = Annotation(
    ...     text="Abraham Lincoln was the 16th President of the United States.",
    ...     context_list=["Lincoln was elected the 16th president of the United States in 1860.", "Abraham Lincoln was born on February 12, 1809, in a one-room log cabin on the Sinking Spring Farm in Hardin County, Kentucky."],
    ... )
    """

    labels: list[Label] | None = None
    metadata: dict | None = None
    bounding_box: Box | None = None
    polygon: Polygon | Box | None = None
    raster: Raster | None = None
    embedding: Embedding | None = None
    is_instance: bool | None = None
    implied_task_types: list[str] | None = None
    text: str | None = None
    context_list: list[str] | None = None

    def __post_init__(self):
        """Validate instantiated class."""

        if self.labels is not None:
            if not isinstance(self.labels, list):
                raise TypeError(
                    f"Expected 'labels' to be of type 'list', got {type(self.labels).__name__}"
                )
            if not all(isinstance(label, Label) for label in self.labels):
                raise TypeError(
                    "All items in 'labels' must be of type 'Label'"
                )

        if not isinstance(self.metadata, (dict, type(None))):
            raise TypeError(
                f"Expected 'metadata' to be of type 'dict' or 'None', got {type(self.metadata).__name__}"
            )

        if not isinstance(self.bounding_box, (Box, type(None))):
            raise TypeError(
                f"Expected 'bounding_box' to be of type 'schemas.Box' or 'None', got {type(self.bounding_box).__name__}"
            )

        if not isinstance(self.polygon, (Polygon, Box, type(None))):
            raise TypeError(
                f"Expected 'polygon' to be of type 'schemas.Polygon' or 'None', got {type(self.polygon).__name__}"
            )

        if not isinstance(self.raster, (Raster, type(None))):
            raise TypeError(
                f"Expected 'raster' to be of type 'schemas.Raster' or 'None', got {type(self.raster).__name__}"
            )

        if not isinstance(self.embedding, (Embedding, type(None))):
            raise TypeError(
                f"Expected 'embedding' to be of type 'Embedding' or 'None', got {type(self.embedding).__name__}"
            )

        if not isinstance(self.is_instance, (bool, type(None))):
            raise TypeError(
                f"Expected 'is_instance' to be of type 'bool' or 'None', got {type(self.is_instance).__name__}"
            )

        if not isinstance(self.implied_task_types, (list, type(None))):
            raise TypeError(
                f"Expected 'implied_task_types' to be of type 'list' or 'None', got {type(self.implied_task_types).__name__}"
            )
        if self.implied_task_types is not None and not all(
            isinstance(task_type, str) for task_type in self.implied_task_types
        ):
            raise TypeError(
                "All items in 'implied_task_types' must be of type 'str'"
            )

        if not isinstance(self.text, (str, type(None))):
            raise TypeError(
                f"Expected 'text' to be of type 'str' or 'None', got {type(self.text).__name__}"
            )

        if self.context_list is not None:
            if not isinstance(self.context_list, list):
                raise TypeError(
                    f"Expected 'context_list' to be of type 'list' or 'None', got {type(self.context_list).__name__}"
                )

            if not all(
                isinstance(context, str) for context in self.context_list
            ):
                raise TypeError(
                    "All items in 'context_list' must be of type 'str'"
                )


@dataclass
class GroundTruth:
    """
    An object describing a ground truth (e.g., a human-drawn bounding box on an image).

    Attributes
    ----------
    datum : Datum
        The datum associated with the groundtruth.
    annotations : list[Annotation]
        The list of annotations associated with the groundtruth.

    Examples
    --------
    >>> GroundTruth(
    ...     datum=Datum(uid="uid1"),
    ...     annotations=[
    ...         Annotation(
    ...             labels=[Label(key="k1", value="v1")],
    ...         )
    ...     ]
    ... )
    """

    datum: Datum
    annotations: list[Annotation]

    def __post_init__(
        self,
    ):
        """Validate instantiated class."""

        if not isinstance(self.datum, Datum):
            raise TypeError(
                f"Expected 'datum' to be of type 'Datum', got {type(self.datum).__name__}"
            )

        if not isinstance(self.annotations, list):
            raise TypeError(
                f"Expected 'annotations' to be of type 'list', got {type(self.annotations).__name__}"
            )
        if not all(
            isinstance(annotation, Annotation)
            for annotation in self.annotations
        ):
            raise TypeError(
                "All items in 'annotations' must be of type 'Annotation'"
            )


@dataclass
class Prediction:
    """
    An object describing a prediction (e.g., a machine-drawn bounding box on an image).

    Attributes
    ----------
    datum : Datum
        The datum associated with the prediction.
    annotations : list[Annotation]
        The list of annotations associated with the prediction.

    Examples
    --------
    >>> Prediction(
    ...     datum=Datum(uid="uid1"),
    ...     annotations=[
    ...         Annotation(
    ...             labels=[
    ...                 Label(key="k1", value="v1", score=0.9),
    ...                 Label(key="k1", value="v1", score=0.1)
    ...             ],
    ...         )
    ...     ]
    ... )
    """

    datum: Datum
    annotations: list[Annotation]

    def __post_init__(self):
        """Validate instantiated class."""

        if not isinstance(self.datum, Datum):
            raise TypeError(
                f"Expected 'datum' to be of type 'Datum', got {type(self.datum).__name__}"
            )

        if not isinstance(self.annotations, list):
            raise TypeError(
                f"Expected 'annotations' to be of type 'list', got {type(self.annotations).__name__}"
            )
        if not all(
            isinstance(annotation, Annotation)
            for annotation in self.annotations
        ):
            raise TypeError(
                "All items in 'annotations' must be of type 'Annotation'"
            )
