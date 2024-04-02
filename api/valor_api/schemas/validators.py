import datetime
import re
from typing import Any

from valor_api.enums import TaskType


def check_type_bool(v: Any) -> bool:
    return isinstance(v, bool)


def check_type_integer(v: Any) -> bool:
    return isinstance(v, int)


def check_type_float(v: Any) -> bool:
    return isinstance(v, float)


def check_type_string(v: Any) -> bool:
    if not isinstance(v, str):
        return False
    validate_string(v)
    return True


def check_type_datetime(v: Any) -> bool:
    if not isinstance(v, str):
        return False
    try:
        datetime.datetime.fromisoformat(v)
    except ValueError:
        return False
    else:
        return True


def check_type_date(v: Any) -> bool:
    if not isinstance(v, str):
        return False
    try:
        datetime.date.fromisoformat(v)
    except ValueError:
        return False
    else:
        return True


def check_type_time(v: Any) -> bool:
    if not isinstance(v, str):
        return False
    try:
        datetime.time.fromisoformat(v)
    except ValueError:
        return False
    else:
        return True


def check_type_duration(v: Any) -> bool:
    if not isinstance(v, float):
        return False
    try:
        datetime.timedelta(seconds=v)
    except ValueError:
        return False
    else:
        return True


def check_type_point(v: Any) -> bool:
    return (
        isinstance(v, (tuple, list))
        and len(v) == 2
        and isinstance(v[0], (int, float))
        and isinstance(v[1], (int, float))
    )


def check_type_multipoint(v: Any) -> bool:
    return isinstance(v, list) and all([check_type_point(pt) for pt in v])


def check_type_linestring(v: Any) -> bool:
    return (
        isinstance(v, list)
        and len(v) >= 2
        and all([check_type_point(pt) for pt in v])
    )


def check_type_multilinestring(v: Any) -> bool:
    return isinstance(v, list) and all(
        [check_type_linestring(line) for line in v]
    )


def check_type_polygon(v: Any) -> bool:
    return check_type_multilinestring(v) and all(
        [(len(line) >= 4 and line[0] == line[-1]) for line in v]
    )


def check_type_multipolygon(v: Any) -> bool:
    return isinstance(v, list) and all(
        check_type_polygon(polygon) for polygon in v
    )


def validate_string(value: str):
    """Validate that a name doesn't contain any forbidden characters"""
    if not isinstance(value, str):
        raise TypeError(f"Value '{value}' is not a string.")
    allowed_special = ["-", "_", "/", "."]
    pattern = re.compile(f"^[a-zA-Z0-9{''.join(allowed_special)}]+$")
    if not pattern.match(value):
        raise ValueError(
            "The provided string contains illegal characters. Please ensure your input consists of only alphanumeric characters, hyphens, underscores, forward slashes, and periods."
        )


def check_if_empty_annotation(values):
    """Checks if the annotation is empty."""
    return (
        not values.labels
        and values.bounding_box is None
        and values.polygon is None
        and values.raster is None
        and values.embedding is None
    )


def validate_annotation_by_task_type(values):
    """Validates the contents of an annotation by task type."""
    if check_if_empty_annotation(values):
        if values.task_type != TaskType.SKIP:
            values.task_type = TaskType.EMPTY
    match values.task_type:
        case TaskType.CLASSIFICATION:
            if not (
                values.labels
                and values.bounding_box is None
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
                    values.bounding_box is not None
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
                and values.bounding_box is None
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
                and values.bounding_box is None
                and values.polygon is None
                and values.raster is None
            ):
                raise ValueError(
                    "Annotation with task type `embedding` do not support labels or geometries."
                )
        case TaskType.EMPTY | TaskType.SKIP:
            if not check_if_empty_annotation(values):
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
        "bool": check_type_bool,
        "integer": check_type_integer,
        "float": check_type_float,
        "string": check_type_string,
        "datetime": check_type_datetime,
        "date": check_type_date,
        "time": check_type_time,
        "duration": check_type_duration,
        "point": check_type_point,
        "multipoint": check_type_multipoint,
        "linestring": check_type_linestring,
        "multilinestring": check_type_multilinestring,
        "polygon": check_type_polygon,
        "multipolygon": check_type_multipolygon,
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
            raise TypeError(
                "Metadata values must be described using Valor's typing format."
            )
        # validate metadata values
        type_ = value.get("type")
        value_ = value.get("value")
        if (
            not isinstance(type_, str)
            or type_ not in map_str_to_type_validator
        ):
            raise TypeError(
                f"Metadata does not support values with type '{type_}'"
            )
        elif not map_str_to_type_validator[type_](value_):
            raise ValueError(
                f"Metadata value '{value_}' failed validation for type '{type_}'"
            )


def validate_geojson(class_name: str, geojson: dict):
    map_str_to_geojson_validator = {
        "point": check_type_point,
        "multipoint": check_type_multipoint,
        "linestring": check_type_linestring,
        "multilinestring": check_type_multilinestring,
        "polygon": check_type_polygon,
        "multipolygon": check_type_multipolygon,
    }
    # validate geojson
    if class_name.lower() not in map_str_to_geojson_validator:
        raise TypeError(
            f"Class '{class_name}' is not a supported GeoJSON geometry."
        )
    elif not isinstance(geojson, dict):
        raise TypeError(
            "GeoJSON should be defined by an object of type 'dict'."
        )
    elif set(geojson.keys()) != {"type", "coordinates"}:
        raise KeyError(
            "Expected geojson to have keys 'type' and 'coordinates'."
        )
    elif geojson.get("type") != class_name:
        raise TypeError(f"GeoJSON type does not match '{class_name}'.")
    # validate coordinates
    if not map_str_to_geojson_validator[class_name.lower()](
        geojson.get("coordinates")
    ):
        raise ValueError(f"Value does not conform to {class_name}.")


def deserialize(class_name: str, data: Any) -> Any:
    if isinstance(data, dict) and set(data.keys()) == {"type", "value"}:
        if not data.get("type") == class_name.lower():
            raise TypeError(
                f"'{class_name}' received value with type '{data.get('type')}'"
            )
        return data.get("value")
    return data
