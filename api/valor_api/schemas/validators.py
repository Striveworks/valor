import datetime
import re
from typing import Any

from valor_api.enums import TaskType


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
    allowed_special = ["-", "_", "/", "."]
    pattern = re.compile(f"^[a-zA-Z0-9{''.join(allowed_special)}]+$")
    if not pattern.match(v):
        raise ValueError(
            "The provided string contains illegal characters. Please ensure your input consists of only alphanumeric characters, hyphens, underscores, forward slashes, and periods."
        )


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


def _check_if_empty_annotation(values):
    """Checks if the annotation is empty."""
    return (
        not values.labels
        and values.box is None
        and values.polygon is None
        and values.raster is None
        and values.embedding is None
    )


def validate_annotation_by_task_type(values):
    """Validates the contents of an annotation by task type."""
    if _check_if_empty_annotation(values):
        if values.task_type != TaskType.SKIP:
            values.task_type = TaskType.EMPTY
    match values.task_type:
        case TaskType.CLASSIFICATION:
            if not (
                values.labels
                and values.box is None
                and values.polygon is None
                and values.raster is None
                and values.embedding is None
            ):
                raise ValueError(
                    "Annotations with task type `classification` do not support geometries or embeddings."
                )
        case TaskType.OBJECT_DETECTION:
            if not (
                values.labels
                and (
                    values.box is not None
                    or values.polygon is not None
                    or values.raster is not None
                )
                and values.embedding is None
            ):
                raise ValueError(
                    "Annotations with task type `object-detection` do not support embeddings."
                )
        case TaskType.SEMANTIC_SEGMENTATION:
            if not (
                values.labels
                and values.raster is not None
                and values.box is None
                and values.polygon is None
                and values.embedding is None
            ):
                raise ValueError(
                    "Annotations with task type `semantic-segmentation` only supports rasters."
                )
        case TaskType.EMBEDDING:
            if not (
                values.embedding is not None
                and not values.labels
                and values.box is None
                and values.polygon is None
                and values.raster is None
            ):
                raise ValueError(
                    "Annotation with task type `embedding` do not support labels or geometries."
                )
        case TaskType.EMPTY | TaskType.SKIP:
            if not _check_if_empty_annotation(values):
                raise ValueError("Annotation is not empty.")
        case _:
            raise NotImplementedError(
                f"Task type `{values.task_type}` is not supported."
            )
    return values


def validate_groundtruth_annotations(annotations: list[Any]):
    """Check that a label appears once in the annotations for semenatic segmentations"""
    labels = []
    indices = dict()
    for index, annotation in enumerate(annotations):
        if annotation.task_type == TaskType.SEMANTIC_SEGMENTATION:
            for label in annotation.labels:
                if label in labels:
                    raise ValueError(
                        f"Label {label} appears in both annotation {index} and {indices[label]}, but semantic segmentation "
                        "tasks can only have one annotation per label."
                    )
                labels.append(label)
                indices[label] = index


def validate_prediction_annotations(annotations: list[Any]):
    """Validate prediction annotations by task type."""

    labels = []
    indices = dict()
    for index, annotation in enumerate(annotations):
        if annotation.task_type == TaskType.CLASSIFICATION:
            # Check that the label scores sum to 1.
            label_keys_to_sum = {}
            for scored_label in annotation.labels:
                if scored_label.score is None:
                    raise ValueError(
                        f"Missing score for label in {annotation.task_type} task."
                    )
                label_key = scored_label.key
                if label_key not in label_keys_to_sum:
                    label_keys_to_sum[label_key] = 0.0
                label_keys_to_sum[label_key] += scored_label.score
            for k, total_score in label_keys_to_sum.items():
                if abs(total_score - 1) > 1e-5:
                    raise ValueError(
                        "For each label key, prediction scores must sum to 1, but"
                        f" for label key {k} got scores summing to {total_score}."
                    )

        elif annotation.task_type == TaskType.OBJECT_DETECTION:
            # Check that we have scores for all the labels.
            for label in annotation.labels:
                if label.score is None:
                    raise ValueError(
                        f"Missing score for label in {annotation.task_type} task."
                    )

        elif annotation.task_type == TaskType.SEMANTIC_SEGMENTATION:
            for label in annotation.labels:
                # Check that score is not defined.
                if label.score is not None:
                    raise ValueError(
                        "Semantic segmentation tasks cannot have scores; only metrics with "
                        "hard predictions are supported."
                    )
                # Check that a label appears once in the annotations.
                if label in labels:
                    raise ValueError(
                        f"Label {label} appears in both annotation {index} and {indices[label]}, but semantic segmentation "
                        "tasks can only have one annotation per label."
                    )
                labels.append(label)
                indices[label] = index


def validate_dictionary(dictionary: dict):
    map_str_to_type_validator = {
        "bool": validate_type_bool,
        "integer": validate_type_integer,
        "float": validate_type_float,
        "string": validate_type_string,
        "datetime": validate_type_datetime,
        "date": validate_type_date,
        "time": validate_type_time,
        "duration": validate_type_duration,
        "point": validate_type_point,
        "multipoint": validate_type_multipoint,
        "linestring": validate_type_linestring,
        "multilinestring": validate_type_multilinestring,
        "polygon": validate_type_polygon,
        "multipolygon": validate_type_multipolygon,
    }
    if not isinstance(dictionary, dict):
        raise TypeError("Expected 'metadata' to be a dictionary.")
    for key, value in dictionary.items():
        # validate metadata structure
        if not isinstance(key, str):
            raise TypeError("Metadata keys must be of type 'str'.")
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
                f"Metadata does not support values with type '{type_str}'"
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


def validate_geojson(geojson: dict):
    map_str_to_geojson_validator = {
        "Point": validate_type_point,
        "MultiPoint": validate_type_multipoint,
        "LineString": validate_type_linestring,
        "MultiLineString": validate_type_multilinestring,
        "Polygon": validate_type_polygon,
        "MultiPolygon": validate_type_multipolygon,
    }
    # validate geojson
    if not (
        isinstance(geojson, dict)
        and set(geojson.keys()) == {"type", "coordinates"}
        and (geometry_type := geojson.get("type"))
        and (geometry_value := geojson.get("coordinates"))
    ):
        raise ValueError(
            f"Expected geojson to be a dictionary with keys 'type' and 'coordinates'. Received value '{geojson}'."
        )

    # validate type
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


def deserialize(class_name: str, values: Any) -> Any:
    if isinstance(values, dict) and set(values.keys()) == {"type", "value"}:
        values_type = values.pop("type")
        if values_type != class_name.lower():
            raise TypeError(
                f"'{class_name}' received value with type '{values_type}'"
            )
        values.pop("type")
    return values
