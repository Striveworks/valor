import datetime
from typing import Dict, List, Union
from velour.schemas import geometry

ValueType = Union[int, float, str, bool, Dict[str, str]]

MetadataType = Dict[
    str,
    Union[
        int, 
        float, 
        str, 
        bool, 
        datetime.datetime, 
        datetime.date, 
        datetime.time,
        datetime.timedelta,
    ],
]

GeometryType = Union[
    geometry.Point, 
    geometry.Polygon, 
    geometry.BoundingBox, 
    geometry.MultiPolygon, 
    geometry.Raster
]

GeoJSONPointType = Dict[str, Union[str, List[Union[float, int]]]]
GeoJSONPolygonType = Dict[str, Union[str, List[List[List[Union[float, int]]]]]]
GeoJSONMultiPolygonType = Dict[
    str, Union[str, List[List[List[List[Union[float, int]]]]]]
]
GeoJSONType = Union[
    GeoJSONPointType, GeoJSONPolygonType, GeoJSONMultiPolygonType
]