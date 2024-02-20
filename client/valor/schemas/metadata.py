import datetime
from typing import Dict, List, Union

from valor.typing import is_geojson

GeoJSONPointType = Dict[str, Union[str, List[Union[float, int]]]]
GeoJSONPolygonType = Dict[str, Union[str, List[List[List[Union[float, int]]]]]]
GeoJSONMultiPolygonType = Dict[
    str, Union[str, List[List[List[List[Union[float, int]]]]]]
]
GeoJSONType = Union[
    GeoJSONPointType, GeoJSONPolygonType, GeoJSONMultiPolygonType
]

ValueType = Union[
    bool,
    int,
    float,
    str,
    datetime.datetime,
    datetime.date,
    datetime.time,
    datetime.timedelta,
    GeoJSONType,
]

MetadataType = Dict[str, ValueType]
ConvertibleMetadataType = Dict[
    str,
    Union[
        bool,
        int,
        float,
        str,
        Dict[str, str],
        Dict[str, GeoJSONType],
    ],
]


def _convert_object_to_metadatum(
    value: ValueType,
) -> Union[bool, int, float, str, Dict[str, str], Dict[str, GeoJSONType]]:
    """Converts an object into a valor metadatum."""

    # atomic types
    if (
        isinstance(value, bool)
        or isinstance(value, int)
        or isinstance(value, float)
        or isinstance(value, str)
    ):
        return value

    # datetime
    elif isinstance(value, datetime.datetime):
        return {"datetime": value.isoformat()}
    elif isinstance(value, datetime.date):
        return {"date": value.isoformat()}
    elif isinstance(value, datetime.time):
        return {"time": value.isoformat()}
    elif isinstance(value, datetime.timedelta):
        return {"duration": str(value.total_seconds())}

    # geojson
    elif is_geojson(value):
        return {"geojson": value}

    # not implemented
    else:
        raise NotImplementedError(
            f"Object with type '{type(value)}' is not currently supported."
        )


def _convert_metadatum_to_object(
    value: Union[bool, int, float, str, Dict[str, str], Dict[str, GeoJSONType]]
) -> ValueType:
    """Converts a valor metadatum into an object."""

    # atomic types
    if (
        isinstance(value, bool)
        or isinstance(value, int)
        or isinstance(value, float)
        or isinstance(value, str)
    ):
        return value

    # validate serialized type
    elif not isinstance(value, dict):
        raise TypeError(
            f"Object with type '{type(value)}' is not atomic and not serialized."
        )

    # datetime
    elif "datetime" in value:
        if not isinstance(value["datetime"], str):
            raise TypeError
        return datetime.datetime.fromisoformat(value["datetime"])
    elif "date" in value:
        if not isinstance(value["date"], str):
            raise TypeError
        return datetime.date.fromisoformat(value["date"])
    elif "time" in value:
        if not isinstance(value["time"], str):
            raise TypeError
        return datetime.time.fromisoformat(value["time"])
    elif "duration" in value:
        if not isinstance(value["duration"], str):
            raise TypeError
        return datetime.timedelta(seconds=float(value["duration"]))

    # geojson
    elif "geojson" in value:
        return value["geojson"]

    # not implemented
    else:
        raise TypeError(
            f"Object with type '{value.keys()}' is not currently supported."
        )


def validate_metadata(metadata: MetadataType):
    """Validates metadata dictionary."""
    if not isinstance(metadata, dict):
        raise TypeError("`metadata` should be an object of type `dict`.")
    for key, value in metadata.items():
        if not isinstance(key, str):
            raise TypeError("`metadata` key should have type `str`.")
        if not (
            isinstance(value, int)
            or isinstance(value, float)
            or isinstance(value, str)
            or isinstance(value, datetime.datetime)
            or isinstance(value, datetime.date)
            or isinstance(value, datetime.time)
            or is_geojson(value)
        ):
            raise TypeError(
                "`metadata` value should have type `str`, `int`, `float`, `datetime` or `geojson`."
            )


def dump_metadata(metadata: MetadataType) -> ConvertibleMetadataType:
    """Converts metadata to API-compatible dictionary."""
    return {
        key: _convert_object_to_metadatum(value)
        for key, value in metadata.items()
    }


def load_metadata(metadata: ConvertibleMetadataType) -> MetadataType:
    """Converts API metadata to Client-compatible dictionary."""
    return {
        key: _convert_metadatum_to_object(value)
        for key, value in metadata.items()
    }
