import datetime
import io
import json
from base64 import b64decode, b64encode
from dataclasses import dataclass
from typing import Any

import numpy as np
import PIL.Image


def generate_type_error(received_value: Any, expected_type: str):
    return TypeError(
        f"Expected value of type '{expected_type}', received value '{received_value}' with type '{type(received_value).__name__}'."
    )


# TODO move validators somewhere else?
def validate_type_bool(v: Any):
    """
    Validates boolean values.

    Parameters
    ----------
    v : Any
        The value to validate.

    Raises
    ------
    TypeError
        If the value is not of type 'bool'.
    """
    if not isinstance(v, bool):
        raise generate_type_error(v, bool.__name__)


def validate_type_integer(v: Any):
    """
    Validates integer values.

    Parameters
    ----------
    v : Any
        The value to validate.

    Raises
    ------
    TypeError
        If the value is not of type 'int'.
    """
    if not isinstance(v, int):
        raise generate_type_error(v, int.__name__)


def validate_type_float(v: Any):
    """
    Validates floating-point values.

    Parameters
    ----------
    v : Any
        The value to validate.

    Raises
    ------
    TypeError
        If the value is not of type 'float'.
    """
    if not isinstance(v, (int, float)):
        raise generate_type_error(v, float.__name__)


def validate_type_string(v: Any):
    """
    Validates string values.

    Parameters
    ----------
    v : Any
        The value to validate.

    Raises
    ------
    TypeError
        If the value is not of type 'str'.
    ValueError
        If the string contains forbidden characters.
    """
    if not isinstance(v, str):
        raise generate_type_error(v, str.__name__)


def validate_type_datetime(v: Any):
    """
    Validates ISO Formatted DateTime values.

    Parameters
    ----------
    v : Any
        The value to validate.

    Raises
    ------
    TypeError
        If the value is not of type 'str'.
    ValueError
        If the value is not formatted correctly.
    """
    if not isinstance(v, str):
        raise generate_type_error(v, "ISO formatted datetime")
    try:
        datetime.datetime.fromisoformat(v)
    except ValueError as e:
        raise ValueError(
            f"DateTime value not provided in correct format: {str(e)}"
        )


def validate_type_date(v: Any):
    """
    Validates ISO Formatted Date values.

    Parameters
    ----------
    v : Any
        The value to validate.

    Raises
    ------
    TypeError
        If the value is not of type 'str'.
    ValueError
        If the value is not formatted correctly.
    """
    if not isinstance(v, str):
        raise generate_type_error(v, "ISO formatted date")
    try:
        datetime.date.fromisoformat(v)
    except ValueError as e:
        raise ValueError(
            f"Date value not provided in correct format: {str(e)}"
        )


def validate_type_time(v: Any):
    """
    Validates ISO Formatted Time values.

    Parameters
    ----------
    v : Any
        The value to validate.

    Raises
    ------
    TypeError
        If the value is not of type 'str'.
    ValueError
        If the value is not formatted correctly.
    """
    if not isinstance(v, str):
        raise generate_type_error(v, "ISO formatted time")
    try:
        datetime.time.fromisoformat(v)
    except ValueError as e:
        raise ValueError(
            f"Time value not provided in correct format: {str(e)}"
        )


def validate_type_duration(v: Any):
    """
    Validates Duration values.

    Parameters
    ----------
    v : Any
        The value to validate.

    Raises
    ------
    TypeError
        If the value is not of type 'float'.
    ValueError
        If the value is not formatted correctly.
    """
    if not isinstance(v, float):
        raise generate_type_error(v, float.__name__)
    try:
        datetime.timedelta(seconds=v)
    except ValueError as e:
        raise ValueError(
            f"Duration value not provided in correct format: {str(e)}"
        )


def validate_type_point(v: Any):
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
        raise generate_type_error(v, "tuple[float, float] or list[float]")
    elif not (
        len(v) == 2
        and isinstance(v[0], (int, float))
        and isinstance(v[1], (int, float))
    ):
        raise ValueError(
            f"Expected point to have two numeric values representing an (x, y) pair. Received '{v}'."
        )


def validate_type_multipoint(v: Any):
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
        raise generate_type_error(
            v, "list[tuple[float, float]] or list[list[float]]"
        )
    elif not v:
        raise ValueError("List cannot be empty.")
    for point in v:
        validate_type_point(point)


def validate_type_linestring(v: Any):
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
    validate_type_multipoint(v)
    if len(v) < 2:
        raise ValueError(
            f"A line requires two or more points. Received '{v}'."
        )


def validate_type_multilinestring(v: Any):
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
        return generate_type_error(
            v, "list[list[tuple[float, float]]] or list[list[list[float]]]"
        )
    elif not v:
        raise ValueError("List cannot be empty.")
    for line in v:
        validate_type_linestring(line)


def validate_type_polygon(v: Any):
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
    validate_type_multilinestring(v)
    for line in v:
        if not (len(line) >= 4 and line[0] == line[-1]):
            raise ValueError(
                "A polygon is defined by a line of at least four points with the first and last points being equal."
            )


def validate_type_box(v: Any):
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
    validate_type_polygon(v)
    if not (len(v) == 1 and len(v[0]) == 5 and v[0][0] == v[0][-1]):
        raise ValueError(
            "Boxes are defined by five points with the first and last being equal."
        )


def validate_type_multipolygon(v: Any):
    """
    Validates geometric multipolygon values.

    Parameters
    ----------
    v : Any
        The value to validate.

    Raises
    ------
    TypeError
        If the value is not of type 'list'.
    ValueError
        If the value does not conform to the multipolygon requirements.
    """
    if not isinstance(v, list):
        raise generate_type_error(
            v,
            "list[list[list[tuple[float, float]]]] or list[list[list[list[float]]]]",
        )
    elif not v:
        raise ValueError("List cannot be empty.")
    for polygon in v:
        validate_type_polygon(polygon)


def validate_geojson(geojson: dict):
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
        "point": validate_type_point,
        "multipoint": validate_type_multipoint,
        "linestring": validate_type_linestring,
        "multilinestring": validate_type_multilinestring,
        "polygon": validate_type_polygon,
        "multipolygon": validate_type_multipolygon,
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
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"Value does not conform to '{geometry_type}'. Validation error: {str(e)}"
        )


def validate_metadata(dictionary: dict):
    """
    Validates that a dictionary conforms to Valor's metadata specification.

    Parameters
    ----------
    dictionary: dict
        The dictionary to validate.

    Raises
    ------
    TypeError
        If the passed in value is not a dictionary.
        If the dictionary keys are not strings.
        If a value type is not supported.
    ValueError
        If the dictionary does not conform to the Valor metadata format.
        If a value is not properly formatted.
    """
    map_str_to_type_validator = {
        "bool": validate_type_bool,
        "integer": validate_type_integer,
        "float": validate_type_float,
        "string": validate_type_string,
        "datetime": validate_type_datetime,
        "date": validate_type_date,
        "time": validate_type_time,
        "duration": validate_type_duration,
        "geojson": validate_geojson,
    }
    if not isinstance(dictionary, dict):
        raise TypeError("Expected 'metadata' to be a dictionary.")
    for key, value in dictionary.items():
        # validate metadata structure
        if not isinstance(key, str):
            raise TypeError("Metadata keys must be of type 'str'.")
        # atomic values don't require explicit typing.
        elif isinstance(value, (bool, int, float, str)):
            continue
        # if a value is not atomic, explicit typing it required.
        elif not isinstance(value, dict) or set(value.keys()) != {
            "type",
            "value",
        }:
            raise ValueError(
                "Metadata values must be described using Valor's typing format."
            )
        # validate metadata type
        type_str = value.get("type")
        if (
            not isinstance(type_str, str)
            or type_str not in map_str_to_type_validator
        ):
            raise TypeError(
                f"Metadata does not support values with type '{type_str}'. Received value '{value.get('value')}'."
            )
        # validate metadata value
        value_ = value.get("value")
        try:
            map_str_to_type_validator[type_str](value_)
        except (
            TypeError,
            ValueError,
        ) as e:
            raise ValueError(
                f"Metadata value '{value_}' failed validation for type '{type_str}'. Validation error: {str(e)}"
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
        validate_type_point(self.value)

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
        validate_type_multipolygon(self.value)

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
        validate_type_linestring(self.value)

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
        validate_type_multilinestring(self.value)

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

    value: list[list[tuple[int | float, int | float]]]

    def __post_init__(self):
        validate_type_polygon(self.value)

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

    value: list[list[tuple[int | float, int | float]]]

    def __post_init__(self):
        validate_type_box(self.value)

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
        validate_type_multipolygon(self.value)

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


@dataclass
class GeoJSON:
    type: str
    coordinates: (
        list[float]
        | list[list[float]]
        | list[list[list[float]]]
        | list[list[list[list[float]]]]
    )

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


@dataclass
class Raster:
    """
    Describes a raster in geometric space.

    Attributes
    ----------
    mask : str
        The mask describing the raster.
    geometry : Box | Polygon | MultiPolygon, optional
        Option to define raster by a geometry. Overrides the bitmask.

    Raises
    ------
    ValueError
        If the image format is not PNG.
        If the image mode is not binary.
    """

    mask: str
    geometry: Box | Polygon | MultiPolygon | None = None

    def __post_init__(self):
        f = io.BytesIO(b64decode(self.mask))
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

    @classmethod
    def from_numpy(cls, mask: np.ndarray) -> "Raster":
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
        geometry: Box | Polygon | MultiPolygon,
        height: int | float,
        width: int | float,
    ) -> "Raster":
        """
        Create a Raster object from a geometry.

        Parameters
        ----------
        geometry : Box | Polygon | MultiPolygon
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
