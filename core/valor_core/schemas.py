import io
import json
import math
from base64 import b64decode, b64encode
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict, Union

import numpy as np
import PIL.Image
from valor_core import enums


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
                "A polygon is defined by a line of at least four points with the first and last points being equal."
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

    # check that the box is axis-aligned
    unique_x_values = set()
    unique_y_values = set()

    for x, y in v[0]:
        unique_x_values.add(x)
        unique_y_values.add(y)

    if (len(unique_x_values) != 2) or (len(unique_y_values) != 2):
        raise NotImplementedError(
            "Expected an axis-aligned Box, but found too many unique values. Rotated and skewed bounding boxes are not yet supported."
        )


def _validate_type_multipolygon(v: Any) -> None:
    """
    Validates geometric multipolygon values.

    Parameters
    ----------
    v : Any
        The value to validate.v

    Raises
    ------
    TypeError
        If the value is not of type 'list'.
    ValueError
        If the value does not conform to the multipolygon requirements.
    """
    if not isinstance(v, list):
        _generate_type_error(
            v,
            "list[list[list[tuple[float, float]]]] or list[list[list[list[float]]]]",
        )
    elif not v:
        raise ValueError("List cannot be empty.")
    for polygon in v:
        _validate_type_polygon(polygon)


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
        "multipolygon": _validate_type_multipolygon,
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

    value: Union[list[list[tuple[int, int]]], list[list[tuple[float, float]]]]

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
        List[Tuple(float, float)]
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
        List[List[Tuple(float, float)]]
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

    value: Union[list[list[tuple[int, int]]], list[list[tuple[float, float]]]]

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
class MultiPolygon:
    """
    Describes a MultiPolygon in (x,y) coordinates.

    Attributes
    ----------
    value : list[list[list[list[int | float]]]]
        A list of coordinates describing the MultiPolygon.

    Raises
    ------
    ValueError
        If the value doesn't conform to the type.
    """

    value: list[list[list[tuple[int | float, int | float]]]]

    def __post_init__(self):
        """Validate instantiated class."""

        _validate_type_multipolygon(self.value)

    @classmethod
    def from_dict(cls, geojson: dict) -> "MultiPolygon":
        """
        Create a MultiPolygon from a GeoJSON in dictionary format.

        Parameters
        ----------
        geojson: dict[str, str | list[list[list[list[int | float]]]]]
            A MultiPolygon value in GeoJSON format.
        """
        geometry = GeoJSON(**geojson).geometry
        if not isinstance(geometry, MultiPolygon):
            raise TypeError(f"GeoJSON is for a different type '{geojson}'.")
        return geometry

    @classmethod
    def from_json(cls, geojson: str) -> "MultiPolygon":
        """
        Create a dictionary that represents the MultiPolygon in GeoJSON format.

        Returns
        ----------
        dict[str, str | list[list[list[list[int | float]]]]]
            A MultiPolygon value in GeoJSON format.
        """
        return cls.from_dict(json.loads(geojson))

    def to_dict(self) -> dict[str, str | list[list[list[list[int | float]]]]]:
        """
        Create a MultiPolygon from a GeoJSON in json format.

        Parameters
        ----------
        geojson: str
            A MultiPolygon value in GeoJSON format.
        """
        return {
            "type": "MultiPolygon",
            "coordinates": [
                [
                    [list(point) for point in subpolygon]
                    for subpolygon in polygon
                ]
                for polygon in self.value
            ],
        }

    def to_json(self) -> str:
        """
        Create a json string that represents the MultiPolygon in GeoJSON format.

        Returns
        ----------
        str
            A MultiPolygon value in GeoJSON format.
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
        polygons = [
            "("
            + "),(".join(
                [
                    ",".join(
                        [f"{point[0]} {point[1]}" for point in subpolygon]
                    )
                    for subpolygon in polygon
                ]
            )
            + ")"
            for polygon in self.value
        ]
        coords = "),(".join(polygons)
        return f"MULTIPOLYGON (({coords}))"

    def to_array(self) -> np.ndarray:
        """
        Convert MultiPolygon to an array.

        Returns
        -------
        np.ndarray
        """
        return np.array(self.value[0][0])


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
    ) -> (
        Point
        | MultiPoint
        | LineString
        | MultiLineString
        | Polygon
        | MultiPolygon
    ):
        map_str_to_type = {
            "Point": Point,
            "MultiPoint": MultiPoint,
            "LineString": LineString,
            "MultiLineString": MultiLineString,
            "Polygon": Polygon,
            "MultiPolygon": MultiPolygon,
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


class RasterData(TypedDict):
    mask: Optional[np.ndarray]
    geometry: Optional[Union[Box, Polygon, MultiPolygon]]


@dataclass
class Raster:
    """
    Represents a binary mask.

    Parameters
    ----------
    value : Dict[str, typing.Union[np.ndarray, str, None]], optional
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
    >>> Raster.from_numpy(mask)
    """

    value: RasterData

    def __post_init__(self):
        """Validate instantiated class."""

        if not isinstance(self.value, dict):
            raise TypeError(
                "Raster should contain a dictionary describing a mask and optionally a geometry."
            )
        elif set(self.value.keys()) != {"mask", "geometry"}:
            raise ValueError(
                "Raster should be described by a dictionary with keys 'mask' and 'geometry'"
            )
        elif not (
            (
                isinstance(self.value["mask"], np.ndarray)
                and self.value["geometry"] is None
            )
            or (
                self.value["mask"] is None
                and isinstance(self.value["geometry"], (Polygon, MultiPolygon))
            )
        ):
            raise TypeError(
                "Only mask or geometry should be populated, but not both. if populated, we expect mask to have type np.ndarray, and expected geometry to have type Polygon or MultiPolygon."
            )

        if (
            self.value["mask"] is not None
            and len(self.value["mask"].shape) != 2
        ):
            raise ValueError("raster only supports 2d arrays")

        if self.value["mask"] is not None and self.value["mask"].dtype != bool:
            raise ValueError(
                f"Expecting a binary mask (i.e. of dtype bool) but got dtype {self.value['mask'].dtype}"
            )

    def encode_value(self) -> Any:
        """Encode object to JSON compatible dictionary."""
        value = self.value
        if value is None:
            return None

        if self.value["mask"] is not None:
            f = io.BytesIO()
            PIL.Image.fromarray(self.value["mask"]).save(f, format="PNG")
            f.seek(0)
            mask_bytes = f.read()
            f.close()
            decoded_mask_bytes = b64encode(mask_bytes).decode()
        else:
            decoded_mask_bytes = None
        return {
            "mask": decoded_mask_bytes,
            "geometry": self.value["geometry"],
        }

    @classmethod
    def decode_value(cls, value: Any):
        """Decode object from JSON compatible dictionary."""
        if not (
            isinstance(value, dict)
            and set(value.keys()) == {"mask", "geometry"}
        ):
            raise ValueError(
                f"Improperly formatted raster encoding. Received '{value}'"
            )
        mask_bytes = b64decode(value["mask"])
        with io.BytesIO(mask_bytes) as f:
            img = PIL.Image.open(f)
            value = {
                "mask": np.array(img),
                "geometry": value["geometry"],
            }
        return cls(value=value)

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

        Raises
        ------
        ValueError
            If the input array is not 2D or not of dtype bool.
        """
        return cls(value={"mask": mask, "geometry": None})

    @classmethod
    def from_geometry(
        cls,
        geometry: Union[Box, Polygon, MultiPolygon],
    ):
        """
        Create a Raster object from a geometric mask.

        Parameters
        ----------
        geometry : Union[Box, Polygon, MultiPolygon]
            Defines the bitmask as a geometry. Overrides any existing mask.
        height : int
            The intended height of the binary mask.
        width : int
            The intended width of the binary mask.

        Returns
        -------
        Raster
        """
        return cls(value={"mask": None, "geometry": geometry})

    def to_array(self) -> np.ndarray | None:
        """
        Convert Raster to a numpy array.

        Returns
        -------
        Optional[np.ndarray]
            A 2D binary array representing the mask if it exists.
        """
        if self.value["geometry"] is not None:
            return self.value["geometry"].to_array()
        else:
            return (
                self.value["mask"]
                if self.value["mask"] is not None
                else np.array([])
            )

    @property
    def geometry(self) -> Union[Box, Polygon, MultiPolygon, None]:
        """
        The geometric mask if it exists.

        Returns
        -------
        Box | Polygon | MultiPolygon | None
            The geometry if it exists.
        """
        return self.value["geometry"]


@dataclass
class Embedding:
    """
    Represents a model embedding.

    Parameters
    ----------
    value : List[float], optional
        An embedding value.
    """

    value: Optional[Union[list[int], list[float]]] = None

    def __post_init__(self):
        """Validate instantiated class."""

        if not isinstance(self.value, list):
            raise TypeError(
                f"Expected type 'Optional[List[float]]' received type '{type(self.value)}'"
            )
        elif len(self.value) < 1:
            raise ValueError("embedding should have at least one dimension")


@dataclass
class Datum:
    """
    A class used to store information about a datum for either a 'GroundTruth' or a 'Prediction'.

    Attributes
    ----------
    uid : String
        The UID of the datum.
    metadata : Dictionary
        A dictionary of metadata that describes the datum.

    Examples
    --------
    >>> Datum(uid="uid1")
    >>> Datum(uid="uid1", metadata={})
    >>> Datum(uid="uid1", metadata={"foo": "bar", "pi": 3.14})
    """

    uid: Optional[str] = None
    metadata: Optional[dict] = None

    def __post_init__(
        self,
    ):
        """Validate instantiated class."""

        if not isinstance(self.uid, (str, type(None))):
            raise TypeError(
                f"Expected 'uid' to be of type 'str' or 'None', got {type(self.uid).__name__}"
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
    score: Optional[float] = None

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
    metadata: Dictionary
        A dictionary of metadata that describes the `Annotation`.
    labels: List[Label], optional
        A list of labels to use for the `Annotation`.
    bounding_box: schemas.Box
        A bounding box to assign to the `Annotation`.
    polygon: BoundingPolygon
        A polygon to assign to the `Annotation`.
    raster: Raster
        A raster to assign to the `Annotation`.
    embedding: List[float]
        An embedding, described by a list of values with type float and a maximum length of 16,000.
    is_instance: bool, optional
        A boolean describing whether we should treat the Raster attached to an annotation as an instance segmentation or not. If set to true, then the Annotation will be validated for use in object detection tasks. If set to false, then the Annotation will be validated for use in semantic segmentation tasks.
    implied_task_types: list[str], optional
        The validated task types that are applicable to each Annotation. Doesn't need to bet set by the user.

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
    """

    labels: List[Label]
    metadata: Optional[dict] = None
    bounding_box: Optional[Box] = None
    polygon: Optional[Union[Polygon, Box]] = None
    raster: Optional[Raster] = None
    embedding: Optional[Embedding] = None
    is_instance: Optional[bool] = None
    implied_task_types: Optional[List[str]] = None

    def __post_init__(self):
        """Validate instantiated class."""

        if not isinstance(self.labels, list):
            raise TypeError(
                f"Expected 'labels' to be of type 'list', got {type(self.labels).__name__}"
            )
        if not all(isinstance(label, Label) for label in self.labels):
            raise TypeError("All items in 'labels' must be of type 'Label'")

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


@dataclass
class EvaluationParameters:
    """
    Defines optional parameters for evaluation methods.

    Attributes
    ----------
    label_map: Optional[List[List[List[str]]]]
        Optional mapping of individual labels to a grouper label. Useful when you need to evaluate performance using labels that differ across datasets and models.
    metrics: List[str], optional
        The list of metrics to compute, store, and return to the user.
    iou_thresholds_to_compute: List[float], optional
        A list of floats describing which Intersection over Unions (IoUs) to use when calculating metrics (i.e., mAP).
    iou_thresholds_to_return: List[float], optional
        A list of floats describing which Intersection over Union (IoUs) thresholds to calculate a metric for. Must be a subset of `iou_thresholds_to_compute`.
    recall_score_threshold: float, default=0
        The confidence score threshold for use when determining whether to count a prediction as a true positive or not while calculating Average Recall.
    pr_curve_iou_threshold: float, optional
            The IOU threshold to use when calculating precision-recall curves for object detection tasks. Defaults to 0.5.
    pr_curve_max_examples: int
        The maximum number of datum examples to store when calculating PR curves.
    """

    label_map: Optional[Dict[Label, Label]] = None
    metrics_to_return: Optional[List[enums.MetricType]] = None
    iou_thresholds_to_compute: Optional[List[float]] = None
    iou_thresholds_to_return: Optional[List[float]] = None
    convert_annotations_to_type: Optional[enums.AnnotationType] = None
    recall_score_threshold: float = 0.0
    pr_curve_iou_threshold: float = 0.5
    pr_curve_max_examples: int = 1

    def __post_init__(self):
        """Validate instantiated class."""

        if not isinstance(self.label_map, (dict, type(None))):
            raise TypeError(
                f"Expected 'label_map' to be of type 'dict' or 'None', got {type(self.label_map).__name__}"
            )
        if self.label_map and not isinstance(self.label_map, dict):
            raise TypeError("label_map should be a dictionary of Labels.")

        if self.label_map is not None and not all(
            isinstance(k, Label) and isinstance(v, Label)
            for k, v in self.label_map.items()
        ):
            raise TypeError(
                "All keys and values in 'label_map' must be of type 'Label'"
            )

        if not isinstance(self.metrics_to_return, (list, type(None))):
            raise TypeError(
                f"Expected 'metrics_to_return' to be of type 'list' or 'None', got {type(self.metrics_to_return).__name__}"
            )
        if self.metrics_to_return is not None and not all(
            isinstance(metric, enums.MetricType)
            for metric in self.metrics_to_return
        ):
            raise TypeError(
                "All items in 'metrics_to_return' must be of type 'enums.MetricType'"
            )

        if not isinstance(self.iou_thresholds_to_compute, (list, type(None))):
            raise TypeError(
                f"Expected 'iou_thresholds_to_compute' to be of type 'list' or 'None', got {type(self.iou_thresholds_to_compute).__name__}"
            )
        if self.iou_thresholds_to_compute is not None and not all(
            isinstance(threshold, float)
            for threshold in self.iou_thresholds_to_compute
        ):
            raise TypeError(
                "All items in 'iou_thresholds_to_compute' must be of type 'float'"
            )

        if not isinstance(self.iou_thresholds_to_return, (list, type(None))):
            raise TypeError(
                f"Expected 'iou_thresholds_to_return' to be of type 'list' or 'None', got {type(self.iou_thresholds_to_return).__name__}"
            )
        if self.iou_thresholds_to_return is not None and not all(
            isinstance(threshold, float)
            for threshold in self.iou_thresholds_to_return
        ):
            raise TypeError(
                "All items in 'iou_thresholds_to_return' must be of type 'float'"
            )

        if not isinstance(self.recall_score_threshold, float):
            raise TypeError(
                f"Expected 'recall_score_threshold' to be of type 'float', got {type(self.recall_score_threshold).__name__}"
            )

        if not isinstance(self.pr_curve_iou_threshold, float):
            raise TypeError(
                f"Expected 'pr_curve_iou_threshold' to be of type 'float', got {type(self.pr_curve_iou_threshold).__name__}"
            )

        if not isinstance(self.pr_curve_max_examples, int):
            raise TypeError(
                f"Expected 'pr_curve_max_examples' to be of type 'int', got {type(self.pr_curve_max_examples).__name__}"
            )


@dataclass
class Evaluation:
    parameters: EvaluationParameters
    metrics: List[Dict]
    confusion_matrices: Optional[List[Dict]]
    ignored_pred_labels: Optional[List[Label]]
    missing_pred_labels: Optional[List[Label]]
    meta: Optional[Dict] = None

    def __str__(self) -> str:
        """Dumps the object into a JSON formatted string."""
        return json.dumps(self.__dict__, indent=4)

    def __post_init__(self):
        """Validate instantiated class."""

        if not isinstance(self.parameters, EvaluationParameters):
            raise TypeError(
                f"Expected 'parameters' to be of type 'EvaluationParameters', got {type(self.parameters).__name__}"
            )

        if not isinstance(self.metrics, list):
            raise TypeError(
                f"Expected 'metrics' to be of type 'list', got {type(self.metrics).__name__}"
            )
        if not all(isinstance(metric, dict) for metric in self.metrics):
            raise TypeError("All items in 'metrics' must be of type 'dict'")

        if not isinstance(self.confusion_matrices, (list, type(None))):
            raise TypeError(
                f"Expected 'confusion_matrices' to be of type 'list' or 'None', got {type(self.confusion_matrices).__name__}"
            )
        if self.confusion_matrices is not None and not all(
            isinstance(cm, dict) for cm in self.confusion_matrices
        ):
            raise TypeError(
                "All items in 'confusion_matrices' must be of type 'dict'"
            )

        if not isinstance(self.meta, (dict, type(None))):
            raise TypeError(
                f"Expected 'meta' to be of type 'dict' or 'None', got {type(self.meta).__name__}"
            )

    def to_dict(self) -> dict:
        """
        Defines how a `valor.Evaluation` object is serialized into a dictionary.

        Returns
        ----------
        dict
            A dictionary describing an evaluation.
        """
        return {
            "parameters": self.parameters.__dict__,
            "metrics": self.metrics,
            "confusion_matrices": self.confusion_matrices,
            "ignored_pred_labels": self.ignored_pred_labels,
            "missing_pred_labels": self.missing_pred_labels,
            "meta": self.meta,
        }


@dataclass
class GroundTruth:
    """
    An object describing a ground truth (e.g., a human-drawn bounding box on an image).

    Attributes
    ----------
    datum : Datum
        The datum associated with the groundtruth.
    annotations : List[Annotation]
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
    annotations : List[Annotation]
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


LabelMapType = Dict[Label, Label]
