import datetime
import numpy as np
from typing import Dict, List, Union, Mapping, Sequence, Any

from velour.schemas import geometry, constraints

ValueType = Union[int, float, str, bool, Dict[str, str]]

MetadataValueType = Union[
    int,
    float,
    str,
    bool,
    datetime.datetime,
    datetime.date,
    datetime.time,
    datetime.timedelta,
]
MetadataType = Mapping[str, MetadataValueType]
DictMetadataType = Dict[str, MetadataValueType]
ConvertibleMetadataType = Mapping[
    str,
    Union[
        MetadataValueType,
        Dict[str, str],
    ],
]

GeometryType = Union[
    geometry.Point,
    geometry.Polygon,
    geometry.BoundingBox,
    geometry.MultiPolygon,
    geometry.Raster,
]

GeoJSONPointType = Dict[str, Union[str, List[Union[float, int]]]]
GeoJSONPolygonType = Dict[str, Union[str, List[List[List[Union[float, int]]]]]]
GeoJSONMultiPolygonType = Dict[
    str, Union[str, List[List[List[List[Union[float, int]]]]]]
]
GeoJSONType = Union[
    GeoJSONPointType, GeoJSONPolygonType, GeoJSONMultiPolygonType
]


def is_numeric(value: Any) -> bool:
    """
    Checks whether the value input is a numeric type.

    Parameters
    ----------
    value : Any
        The value to check.

    Returns
    -------
    bool
        Whether the value is a number.
    """
    return (
        isinstance(value, int)
        or isinstance(value, float)
        or isinstance(value, np.floating)
    )


def is_floating(value: Any) -> bool:
    """
    Checks whether the value input is a floating point type.

    Parameters
    ----------
    value : Any
        The value to check.

    Returns
    -------
    bool
        Whether the value is a floating point number.
    """
    return (
        isinstance(value, float)
        or isinstance(value, np.floating)
    )
