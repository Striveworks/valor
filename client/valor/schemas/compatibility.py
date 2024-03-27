from typing import Any, Optional, Union

from valor.schemas.symbolic.annotations import (
    BoundingBox,
    BoundingPolygon,
    Raster,
)
from valor.schemas.symbolic.atomics import (
    Date,
    DateTime,
    Duration,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    Time,
)
from valor.schemas.symbolic.structures import (
    Dictionary,
    _get_atomic_type_by_name,
)


def _encode_api_metadata_values(variable):
    if isinstance(
        variable,
        (
            Point,
            MultiPoint,
            LineString,
            MultiLineString,
            Polygon,
            MultiPolygon,
        ),
    ):
        return {
            "geojson": {
                "type": type(variable).__name__.lower(),
                "coordinates": variable.get_value(),
            }
        }
    elif isinstance(variable, (DateTime, Date, Time, Duration)):
        return {type(variable).__name__.lower(): variable.encode_value()}
    else:
        return variable.encode_value()


def _encode_api_metadata(metadata: Dictionary) -> dict:
    return {k: _encode_api_metadata_values(v) for k, v in metadata.items()}


def _encode_api_geometry(
    geometry: Union[
        Polygon, MultiPolygon, BoundingPolygon, BoundingBox, Raster
    ]
):
    value = geometry.get_value()
    if value is None:
        return None
    elif isinstance(geometry, BoundingBox):
        return {
            "polygon": {
                "points": [
                    {"x": pt[0], "y": pt[1]} for pt in geometry.boundary
                ]
            }
        }
    elif isinstance(geometry, (BoundingPolygon, Polygon)):
        return {
            "boundary": {
                "points": [
                    {"x": pt[0], "y": pt[1]} for pt in geometry.boundary
                ]
            },
            "holes": [
                {"points": [{"x": pt[0], "y": pt[1]} for pt in hole]}
                for hole in geometry.holes
            ],
        }
    elif isinstance(geometry, MultiPolygon):
        return {
            "polygons": [
                _encode_api_geometry(poly) for poly in geometry.polygons
            ]
        }
    elif isinstance(geometry, Raster):
        if geometry.geometry is None:
            return geometry.encode_value()
        else:
            value = geometry.encode_value()
            value["geometry"] = _encode_api_geometry(geometry.geometry)
            return value


def encode_api_format(obj: Any) -> dict:
    json = obj.encode_value()
    if "datum" in json:
        json["datum"] = encode_api_format(obj.datum)
    if "annotations" in json:
        json["annotations"] = [
            encode_api_format(annotation) for annotation in obj.annotations
        ]
    if "metadata" in json:
        json["metadata"] = _encode_api_metadata(obj.metadata)
    if "bounding_box" in json:
        json["bounding_box"] = _encode_api_geometry(obj.bounding_box)
    if "polygon" in json:
        json["polygon"] = _encode_api_geometry(obj.polygon)
    if "raster" in json:
        json["raster"] = _encode_api_geometry(obj.raster)
    return json


def _decode_api_metadata_values(value: Any):
    if not isinstance(value, dict):
        return value
    elif set(value.keys()) == {"geojson"}:
        obj = _get_atomic_type_by_name(value["geojson"]["type"])
        return obj.decode_value(value["geojson"]["coordinates"])
    elif len(value) == 1:
        k, v = list(value.items())[0]
        obj = _get_atomic_type_by_name(k)
        return obj.decode_value(v)
    else:
        raise NotImplementedError(str(value))


def _decode_api_metadata(metadata) -> dict:
    return {k: _decode_api_metadata_values(v) for k, v in metadata.items()}


def _decode_api_geometry(value: Optional[dict]):
    if value is None:
        return None
    elif set(value.keys()) == {"polygon"}:
        return [
            [
                (pt["x"], pt["y"])
                for pt in [
                    *value["polygon"]["points"],
                    value["polygon"]["points"][0],
                ]
            ]
        ]
    elif set(value.keys()) == {"boundary", "holes"}:
        return [
            [
                (pt["x"], pt["y"])
                for pt in [
                    *value["boundary"]["points"],
                    value["boundary"]["points"][0],
                ]
            ],
            [
                [
                    (pt["x"], pt["y"])
                    for pt in [*hole["points"], hole["points"][0]]
                ]
                for hole in value["holes"]
            ],
        ]


def decode_api_format(json: dict):
    # objects
    if "datum" in json:
        json["datum"] = decode_api_format(json["datum"])
    if "annotations" in json:
        json["annotations"] = [
            decode_api_format(annotation) for annotation in json["annotations"]
        ]
    # data
    if "metadata" in json:
        json["metadata"] = _decode_api_metadata(json["metadata"])
    if "model_name" in json:
        json.pop("model_name")
    if "dataset_name" in json:
        json.pop("dataset_name")
    if "bounding_box" in json:
        json["bounding_box"] = _decode_api_geometry(json["bounding_box"])
    if "polygon" in json:
        json["polygon"] = _decode_api_geometry(json["polygon"])

    return json
