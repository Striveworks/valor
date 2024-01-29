import datetime
from typing import Dict, List, Union, Mapping, Sequence

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
