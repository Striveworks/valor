import datetime
from typing import Any


def generate_type_error(received_value: Any, expected_type: str):
    return TypeError(
        f"Expected value of type '{expected_type}', received value '{received_value}' with type '{type(received_value).__name__}'."
    )


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


def deserialize(class_name: str, values: Any) -> Any:
    """
    Deserializes a value from Valor schema formatting into a API schema.

    Parameters
    ----------
    class_name: str
        The name of the schema class.
    values: Any
        The value that is being deserialized.

    Returns
    -------
    Any
        The deserialized value.

    Raises
    ------
    TypeError
        If the value type does not match the calling class.
    """
    if isinstance(values, dict) and set(values.keys()) == {"type", "value"}:
        values_type = values.pop("type")
        if values_type != class_name.lower():
            raise TypeError(
                f"'{class_name}' received value with type '{values_type}'"
            )
        values.pop("type")
    elif not isinstance(values, dict):
        values = {"value": values}
    return values
