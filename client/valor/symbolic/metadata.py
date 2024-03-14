from typing import Any, Optional, Dict

from valor.symbolic.modifiers import Variable
from valor.symbolic.atomics import (
    Integer,
    Float,
    String,
    Bool,
    DateTime,
    Date,
    Time,
    Duration,
)
from valor.symbolic.geojson import (
    Point,
    MultiPoint,
    LineString,
    MultiLineString,
    Polygon,
    MultiPolygon,
)
from valor.symbolic.annotations import (
    BoundingBox,
    BoundingPolygon,
    Raster,
)

class MetadataValue():
    def __init__(self, name: str, key: str):
        self._name = name
        self._key = key

    def __eq__(self, other: Any):
        return self._router(fn="__eq__", other=other)

    def __ne__(self, other: Any):
        return self._router(fn="__ne__", other=other)

    def __gt__(self, other: Any):
        return self._router(fn="__gt__", other=other)

    def __ge__(self, other: Any):
        return self._router(fn="__ge__", other=other)

    def __lt__(self, other: Any):
        return self._router(fn="__lt__", other=other)

    def __le__(self, other: Any):
        return self._router(fn="__le__", other=other)

    def intersects(self, other: Any):
        return self._router(fn="intersects", other=other)

    def inside(self, other: Any):
        return self._router(fn="inside", other=other)

    def outside(self, other: Any):
        return self._router(fn="outside", other=other)

    def is_none(self, other: Any):
        return self._router(fn="is_none", other=other)

    def is_not_none(self, other: Any):
        return self._router(fn="is_not_none", other=other)

    @property
    def area(self):
        return Float(name=self._name, key=self._key, attribute="area")

    def _router(self, fn: str, other: Any):
        type_ = type(other)
        if Bool.supports(other):
            obj = Bool
        elif String.supports(other):
            obj = String
        elif Integer.supports(other):
            obj = Integer
        elif Float.supports(other):
            obj = Float        
        elif DateTime.supports(other):
            obj = DateTime
        elif Date.supports(other):
            obj = Date
        elif Time.supports(other):
            obj = Time
        elif Duration.supports(other):
            obj = Duration
        elif Point.supports(other):
            obj = Point
        elif MultiPoint.supports(other):
            obj = MultiPoint
        elif LineString.supports(other):
            obj = LineString
        elif MultiLineString.supports(other):
            obj = MultiLineString
        elif Polygon.supports(other):
            obj = Polygon
        elif MultiPolygon.supports(other):
            obj = MultiPolygon
        elif BoundingBox.supports(other):
            obj = BoundingBox
        elif BoundingPolygon.supports(other):
            obj = BoundingPolygon
        elif Raster.supports(other):
            obj = Raster
        else:
            raise NotImplementedError(str(other))

        return obj(
            name=self._name,
            key=self._key,
        ).__getattribute__(fn)(other)


class Metadata(Variable):

    def __init__(
        self,
        value: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
    ):
        super().__init__(value=value, name=name)

    @staticmethod
    def supports(value: Any) -> bool:
        return type(value) in {dict, Metadata}

    def __getitem__(self, key: str):
        if self.is_symbolic():
            return MetadataValue(name=self._name, key=key)
        return self.value[key]

    def __setitem__(self, key: str, value: Any):
        if self.is_symbolic():
            raise NotImplementedError("Cannot set symbolic value.")
        self.value[key] = value

    def to_dict(self):
        if self.is_symbolic:
            return {"symbol": self._name}
        else:
            return {self._name: self.value}