from typing import Any, Dict, Optional

from valor.symbolic.atomics import (
    Bool,
    Date,
    DateTime,
    Duration,
    Float,
    Integer,
    String,
    Time,
)
from valor.symbolic.geojson import (
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)
from valor.symbolic.modifiers import Symbol, Variable


class MetadataValue:
    def __init__(self, name: str, key: str, attribute: Optional[str] = None):
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
        return Float.symbolic(name=self._name, key=self._key, attribute="area")

    def _router(self, fn: str, other: Any):
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
        else:
            raise NotImplementedError(str(other))

        return obj.symbolic(name=self._name, key=self._key,).__getattribute__(
            fn
        )(other)


class Metadata(Variable):
    def __init__(
        self,
        value: Optional[Dict[str, Any]],
    ):
        super().__init__(value)

    @classmethod
    def supports(cls, value: Any) -> bool:
        return type(value) in {dict, Metadata}

    def __getitem__(self, key: str):
        if type(self._value) is Symbol:
            return MetadataValue(name=self._value._name, key=key)
        elif value := self.get_value():
            return value[key]
        raise ValueError

    def __setitem__(self, key: str, value: Any):
        if self.is_symbolic():
            raise NotImplementedError("Cannot set symbolic value.")
        elif self._value:
            self._value[key] = value
        raise ValueError

    def to_dict(self):
        if type(self._value) is Symbol:
            return {"symbol": self._value._name}
        else:
            return self._value
