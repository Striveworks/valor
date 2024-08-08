import io
import json
from base64 import b64decode, b64encode
from dataclasses import dataclass
from typing import Any, Optional, TypedDict, Union

import numpy as np
import PIL.Image


def calculate_bbox_intersection(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """
    Calculate the intersection area between two bounding boxes.

    Parameters
    ----------
    bbox1 : np.ndarray
        Array representing the first bounding bo.
    bbox2 : np.ndarray
        Array representing the second bounding box with the same format as bbox1.

    Returns
    -------
    float
        The area of the intersection between the two bounding boxes.

    Raises
    ------
    ValueError
        If the input bounding boxes do not have the expected shape.
    """
    # Calculate intersection coordinates
    xmin_inter = max(bbox1[:, 0].min(), bbox2[:, 0].min())
    ymin_inter = max(bbox1[:, 1].min(), bbox2[:, 1].min())
    xmax_inter = min(bbox1[:, 0].max(), bbox2[:, 0].max())
    ymax_inter = min(bbox1[:, 1].max(), bbox2[:, 1].max())

    # Calculate width and height of intersection area
    width = max(0, xmax_inter - xmin_inter)
    height = max(0, ymax_inter - ymin_inter)

    # Calculate intersection area
    intersection_area = width * height
    return intersection_area


def calculate_bbox_union(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """
    Calculate the union area between two bounding boxes.

    Parameters
    ----------
    bbox1 : np.ndarray
        Array representing the first bounding box.
    bbox2 : np.ndarray
        Array representing the second bounding box with the same format as bbox1.

    Returns
    -------
    float
        The area of the union between the two bounding boxes.

    Raises
    ------
    ValueError
        If the input bounding boxes do not have the expected shape.
    """
    area1 = (bbox1[:, 0].max() - bbox1[:, 0].min()) * (
        bbox1[:, 1].max() - bbox1[:, 1].min()
    )
    area2 = (bbox2[:, 0].max() - bbox2[:, 0].min()) * (
        bbox2[:, 1].max() - bbox2[:, 1].min()
    )
    union_area = area1 + area2 - calculate_bbox_intersection(bbox1, bbox2)
    return union_area


def calculate_bbox_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.

    Parameters
    ----------
    bbox1 : np.ndarray
        Array representing the first bounding box.
    bbox2 : np.ndarray
        Array representing the second bounding box with the same format as bbox1.

    Returns
    -------
    float
        The IoU between the two bounding boxes. Returns 0 if the union area is zero.

    Raises
    ------
    ValueError
        If the input bounding boxes do not have the expected shape.
    """
    intersection = calculate_bbox_intersection(bbox1, bbox2)
    union = calculate_bbox_union(bbox1, bbox2)
    iou = intersection / union
    return iou


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


def _validate_type_multipolygon(v: Any) -> None:
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
        """
        Validates typing.

        Parameters
        ----------
        value : Any
            The value to validate.

        Raises
        ------
        TypeError
            If the value type is not supported.
        """
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
        """
        Validates

        Parameters
        ----------
        value : Any
            The value to validate.

        Raises
        ------
        TypeError
            If the value type is not supported.
        """
        if not isinstance(self.value, list):
            raise TypeError(
                f"Expected type 'Optional[List[float]]' received type '{type(self.value)}'"
            )
        elif len(self.value) < 1:
            raise ValueError("embedding should have at least one dimension")
