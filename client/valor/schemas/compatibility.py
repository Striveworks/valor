from typing import Any, Optional, Union

from valor.schemas.symbolic.types import (
    Box,
    Date,
    DateTime,
    Dictionary,
    Duration,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    Raster,
    Time,
    _get_type_by_name,
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
    geometry: Union[Box, Polygon, MultiPolygon, Raster, None]
):
    if geometry is None:
        return None
    elif isinstance(geometry, Box):
        return {
            "polygon": {
                "points": [
                    {"x": pt[0], "y": pt[1]} for pt in geometry.boundary
                ]
            }
        }
    elif isinstance(geometry, Polygon):
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
                _encode_api_geometry(poly) for poly in geometry.to_polygons()
            ]
        }
    elif isinstance(geometry, Raster):
        value = geometry.encode_value()
        if value["geometry"] is not None:
            if Polygon.supports(value["geometry"]):
                value["geometry"] = _encode_api_geometry(
                    Polygon(value["geometry"])
                )
            elif MultiPolygon.supports(value["geometry"]):
                value["geometry"] = _encode_api_geometry(
                    MultiPolygon(value["geometry"])
                )
            else:
                raise ValueError
        return value


def encode_api_format(obj: Any) -> dict:
    """Encodes form client format into api format."""
    json = obj.encode_value()

    # static collection
    if "datum" in json:
        json["datum"] = encode_api_format(obj.datum)
    if "annotations" in json:
        json["annotations"] = [
            encode_api_format(annotation) for annotation in obj.annotations
        ]

    # dictionary
    if "metadata" in json:
        json["metadata"] = _encode_api_metadata(obj.metadata)

    # geometry
    if "box" in json:
        json.pop("box")
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
        obj = _get_type_by_name(value["geojson"]["type"])
        return obj.decode_value(value["geojson"]["coordinates"])
    elif len(value) == 1:
        k, v = list(value.items())[0]
        obj = _get_type_by_name(k)
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
        boundary = [
            (pt["x"], pt["y"])
            for pt in [
                *value["boundary"]["points"],
                value["boundary"]["points"][0],
            ]
        ]
        holes = (
            [
                [
                    (pt["x"], pt["y"])
                    for pt in [*hole["points"], hole["points"][0]]
                ]
                for hole in value["holes"]
            ]
            if value["holes"]
            else None
        )
        return [boundary, *holes] if holes else [boundary]
    elif set(value.keys()) == {"mask", "geometry"}:
        if value["geometry"] is not None:
            value["geometry"] = _decode_api_geometry(value["geometry"])
        return Raster.decode_value(value).get_value()


def decode_api_format(json: dict):
    """Decoded api format into client format."""
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
        json["bounding_box"] = _decode_api_geometry(json["box"])
    if "polygon" in json:
        json["polygon"] = _decode_api_geometry(json["polygon"])
    if "raster" in json:
        json["raster"] = _decode_api_geometry(json["raster"])

    return json
