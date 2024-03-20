import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from valor.symbolic.modifiers import Equatable, Quantifiable, Spatial, Symbol


class Bool(Equatable):
    """
    Bool wrapper.
    """

    @classmethod
    def supports(cls, value: Any) -> bool:
        return isinstance(value, bool)


class Integer(Quantifiable):
    @classmethod
    def supports(cls, value: Any) -> bool:
        return isinstance(value, (int, np.integer))


class Float(Quantifiable):
    @classmethod
    def supports(cls, value: Any) -> bool:
        return isinstance(value, (float, np.floating)) or Integer.supports(
            value
        )


class String(Equatable):
    @classmethod
    def supports(cls, value: Any) -> bool:
        return isinstance(value, str)


class DateTime(Quantifiable):
    @classmethod
    def supports(cls, value: Any) -> bool:
        return type(value) is datetime.datetime

    @classmethod
    def decode_value(cls, value: str):
        return cls(value=datetime.datetime.fromisoformat(value))

    def encode_value(self):
        return self.get_value().isoformat()


class Date(Quantifiable):
    @classmethod
    def supports(cls, value: Any) -> bool:
        return type(value) is datetime.date

    @classmethod
    def decode_value(cls, value: str):
        return cls(value=datetime.date.fromisoformat(value))

    def encode_value(self):
        return self.get_value().isoformat()


class Time(Quantifiable):
    @classmethod
    def supports(cls, value: Any) -> bool:
        return type(value) is datetime.time

    @classmethod
    def decode_value(cls, value: str):
        return cls(value=datetime.time.fromisoformat(value))

    def encode_value(self):
        return self.get_value().isoformat()


class Duration(Quantifiable):
    @classmethod
    def supports(cls, value: Any) -> bool:
        return type(value) is datetime.timedelta

    @classmethod
    def decode_value(cls, value: int):
        return cls(value=datetime.timedelta(seconds=value))

    def encode_value(self):
        return self.get_value().total_seconds()


class Point(Spatial):
    """
    Represents a point in 2D space.

    Parameters
    ----------
    x : Union[float, int]
        The x-coordinate of the point.
    y : Union[float, int]
        The y-coordinate of the point.

    Attributes
    ----------
    x : float
        The x-coordinate of the point.
    y : float
        The y-coordinate of the point.

    Raises
    ------
    TypeError
        If the coordinates are not of type `float` or convertible to `float`.

    Examples
    --------
    >>> Point(value=(1,2))
    Point(x=1.0, y=2.0)
    """

    def __init__(
        self,
        value: Optional[Tuple[float, float]] = None,
        symbol: Optional[Symbol] = None,
    ):
        super().__init__(value=value, symbol=symbol)

    @classmethod
    def supports(cls, value: Any) -> bool:
        if isinstance(value, tuple):
            return (
                len(value) == 2
                and isinstance(value[0], (int, float, np.floating))
                and isinstance(value[1], (int, float, np.floating))
            )
        else:
            return issubclass(type(value), Point)


class MultiPoint(Spatial):
    def __init__(
        self,
        value: Optional[List[Tuple[float, float]]] = None,
        symbol: Optional[Symbol] = None,
    ):
        super().__init__(value=value, symbol=symbol)

    @classmethod
    def supports(cls, value: Any) -> bool:
        if isinstance(value, list):
            for point in value:
                if not Point.supports(point):
                    return False
            return True
        else:
            return issubclass(type(value), MultiPoint)


class LineString(Spatial):
    def __init__(
        self,
        value: Optional[List[Tuple[float, float]]] = None,
        symbol: Optional[Symbol] = None,
    ):
        super().__init__(value=value, symbol=symbol)

    @classmethod
    def supports(cls, value: Any) -> bool:
        if MultiPoint.supports(value):
            return len(value) >= 2
        else:
            return issubclass(type(value), LineString)


class MultiLineString(Spatial):
    def __init__(
        self,
        value: Optional[List[List[Tuple[float, float]]]] = None,
        symbol: Optional[Symbol] = None,
    ):
        super().__init__(value=value, symbol=symbol)

    @classmethod
    def supports(cls, value: Any) -> bool:
        if isinstance(value, list):
            for line in value:
                if not LineString.supports(line):
                    return False
            return True
        else:
            return issubclass(type(value), MultiLineString)


class Polygon(Spatial):
    """
    Represents a polygon with a boundary and optional holes.

    Parameters
    ----------
    boundary : BasicPolygon or dict
        The outer boundary of the polygon. Can be a `BasicPolygon` object or a
        dictionary with the necessary information to create a `BasicPolygon`.
    holes : List[BasicPolygon], optional
        List of holes inside the polygon. Defaults to an empty list.

    Raises
    ------
    TypeError
        If `boundary` is not a `BasicPolygon` or cannot be converted to one.
        If `holes` is not a list, or an element in `holes` is not a `BasicPolygon`.

    Examples
    --------
    Create component polygons with BasicPolygon
    >>> basic_polygon1 = BasicPolygon(...)
    >>> basic_polygon2 = BasicPolygon(...)
    >>> basic_polygon3 = BasicPolygon(...)

    Create a polygon from a basic polygon.
    >>> polygon1 = Polygon(
    ...     boundary=basic_polygon1,
    ... )

    Create a polygon with holes.
    >>> polygon2 = Polygon(
    ...     boundary=basic_polygon1,
    ...     holes=[basic_polygon2, basic_polygon3],
    ... )
    """

    def __init__(
        self,
        value: Optional[List[List[Tuple[float, float]]]] = None,
        symbol: Optional[Symbol] = None,
    ):
        super().__init__(value=value, symbol=symbol)

    @classmethod
    def supports(cls, value: Any) -> bool:
        if MultiLineString.supports(value):
            for line in value:
                if not (len(line) >= 4 and line[0] == line[-1]):
                    return False
            return True
        else:
            return issubclass(type(value), Polygon)

    @property
    def area(self):
        if not isinstance(self._value, Symbol):
            raise ValueError
        return Float.symbolic(name=self._value._name, attribute="area")


class MultiPolygon(Spatial):
    """
    Represents a collection of polygons.

    Parameters
    ----------
    polygons : List[Polygon], optional
        List of `Polygon` objects. Defaults to an empty list.

    Raises
    ------
    TypeError
        If `polygons` is not a list, or an element in `polygons` is not a `Polygon`.

    Examples
    --------
    >>> MultiPolygon(
    ...     polygons=[
    ...         Polygon(...),
    ...         Polygon(...),
    ...         Polygon(...),
    ...     ]
    ... )
    """

    def __init__(
        self,
        value: Optional[List[List[List[Tuple[float, float]]]]] = None,
        symbol: Optional[Symbol] = None,
    ):
        super().__init__(value=value, symbol=symbol)

    @classmethod
    def supports(cls, value: Any) -> bool:
        if isinstance(value, list):
            for poly in value:
                if not Polygon.supports(poly):
                    return False
            return True
        else:
            return issubclass(type(value), MultiPolygon)

    @property
    def area(self):
        if not isinstance(self._value, Symbol):
            raise ValueError
        return Float.symbolic(name=self._value._name, attribute="area")


def _get_atomic_type_by_value(other: Any):
    if Bool.supports(other):
        return Bool
    elif String.supports(other):
        return String
    elif Integer.supports(other):
        return Integer
    elif Float.supports(other):
        return Float
    elif DateTime.supports(other):
        return DateTime
    elif Date.supports(other):
        return Date
    elif Time.supports(other):
        return Time
    elif Duration.supports(other):
        return Duration
    elif MultiPolygon.supports(other):
        return MultiPolygon
    elif Polygon.supports(other):
        return Polygon
    elif MultiLineString.supports(other):
        return MultiLineString
    elif LineString.supports(other):
        return LineString
    elif MultiPoint.supports(other):
        return MultiPoint
    elif Point.supports(other):
        return Point
    else:
        raise NotImplementedError(str(type(other).__name__))


def _get_atomic_type_by_name(name: str):
    name = name.lower()
    if name == "bool":
        return Bool
    elif name == "string":
        return String
    elif name == "integer":
        return Integer
    elif name == "float":
        return Float
    elif name == "datetime":
        return DateTime
    elif name == "date":
        return Date
    elif name == "time":
        return Time
    elif name == "duration":
        return Duration
    elif name == "multipolygon":
        return MultiPolygon
    elif name == "polygon":
        return Polygon
    elif name == "multilinestring":
        return MultiLineString
    elif name == "linestring":
        return LineString
    elif name == "multipoint":
        return MultiPoint
    elif name == "point":
        return Point
    else:
        raise NotImplementedError(name)


class DictionaryValue:
    def __init__(
        self,
        symbol: Symbol,
        key: str,
    ):
        self._key = key
        self._owner = symbol._owner
        self._name = symbol._name
        if symbol._attribute:
            raise ValueError("Symbol attribute should not be defined.")
        if symbol._key:
            raise ValueError("Symbol key should not be defined.")

    def __eq__(self, other: Any):
        return self.generate(fn="__eq__", other=other)

    def __ne__(self, other: Any):
        return self.generate(fn="__ne__", other=other)

    def __gt__(self, other: Any):
        return self.generate(fn="__gt__", other=other)

    def __ge__(self, other: Any):
        return self.generate(fn="__ge__", other=other)

    def __lt__(self, other: Any):
        return self.generate(fn="__lt__", other=other)

    def __le__(self, other: Any):
        return self.generate(fn="__le__", other=other)

    def intersects(self, other: Any):
        return self.generate(fn="intersects", other=other)

    def inside(self, other: Any):
        return self.generate(fn="inside", other=other)

    def outside(self, other: Any):
        return self.generate(fn="outside", other=other)

    def is_none(self, other: Any):
        return self.generate(fn="is_none", other=other)

    def is_not_none(self, other: Any):
        return self.generate(fn="is_not_none", other=other)

    @property
    def area(self):
        return Float.symbolic(
            owner=self._owner, name=self._name, key=self._key, attribute="area"
        )

    def generate(self, other: Any, fn: str):
        obj = _get_atomic_type_by_value(other)
        sym = obj.symbolic(owner=self._owner, name=self._name, key=self._key)
        return sym.__getattribute__(fn)(other)


class Dictionary(Equatable):
    @classmethod
    def definite(
        cls,
        value: Optional[Dict[str, Any]] = None,
    ):
        value = value if value else dict()
        value = {
            k: _get_atomic_type_by_value(v).definite(v)
            for k, v in value.items()
        }
        return super().definite(value)

    @classmethod
    def supports(cls, value: Any) -> bool:
        return type(value) in {dict, Dictionary}

    @classmethod
    def decode_value(cls, value: dict) -> Any:
        return {
            k: _get_atomic_type_by_name(v["type"]).decode_value(v["value"])
            for k, v in value.items()
        }

    def encode_value(self) -> dict:
        return {k: v.to_dict() for k, v in self.items()}

    def __getitem__(self, key: str):
        if isinstance(self._value, Symbol):
            return DictionaryValue(symbol=self._value, key=key)
        return self.get_value()[key]

    def __setitem__(self, key: str, value: Any):
        if isinstance(self._value, Symbol):
            raise NotImplementedError(
                "Symbols do not support the setting of values."
            )
        self.get_value()[key] = value

    def items(self):
        if isinstance(self._value, Symbol):
            raise NotImplementedError("Variable is symbolic")
        return self._value.items() if self._value else dict.items({})
