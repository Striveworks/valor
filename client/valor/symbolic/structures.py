import typing
from typing import Any, Optional

from valor.symbolic.atomics import (
    Bool,
    Date,
    DateTime,
    Duration,
    Equatable,
    Float,
    Integer,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    String,
    Symbol,
    Time,
    Variable,
)


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


T = typing.TypeVar("T")


class List(Equatable):

    _registered_classes = dict()

    @classmethod
    def __class_getitem__(cls, item_cls: type):

        if not issubclass(item_cls, Variable):
            raise TypeError

        if item_cls in cls._registered_classes:
            return cls._registered_classes[item_cls]

        class ValueList(List):
            def __init__(
                self,
                value: Optional[Any] = None,
                symbol: Optional[Symbol] = None,
            ):
                if value is not None:
                    value = [
                        element
                        if isinstance(element, item_cls)
                        else item_cls(element)
                        for element in value
                    ]
                super().__init__(value=value, symbol=symbol)

            @classmethod
            def definite(cls, value: Any):
                if value is None:
                    value = list()
                return cls(value=value)

            @classmethod
            def symbolic(
                cls,
                name: str | None = None,
                key: str | None = None,
                attribute: str | None = None,
                owner: str | None = None,
            ):
                if name is None:
                    name = f"list[{item_cls.__name__.lower()}]"
                return super().symbolic(name, key, attribute, owner)

            @classmethod
            def __validate__(cls, value: list):
                if not isinstance(value, list):
                    raise TypeError(
                        f"Expected type '{list}' received type '{type(value)}'"
                    )
                for element in value:
                    if not isinstance(element, item_cls):
                        raise TypeError(
                            f"Expected list elements with type '{item_cls}' received type '{type(element)}'"
                        )

            @classmethod
            def decode_value(cls, value: Any):
                if not value:
                    return []
                if issubclass(type(value), Variable) and issubclass(
                    item_cls, Variable
                ):
                    return [
                        item_cls.decode_value(element) for element in value
                    ]

            def encode_value(self):
                return [element.encode_value() for element in self.get_value()]

            def to_dict(self) -> dict:
                if isinstance(self._value, Symbol):
                    return self._value.to_dict()
                else:
                    return {
                        "type": f"list[{item_cls.__name__.lower()}]",
                        "value": self.encode_value(),
                    }

            def __getitem__(self, __key: int) -> Float:
                return self.get_value()[__key]

            def __setitem__(self, __key: int, __value: item_cls):
                value = self.get_value()
                if value is None:
                    raise TypeError
                value[__key] = __value

            def __iter__(self) -> typing.Iterator[item_cls]:
                return iter([element for element in self.get_value()])

            @staticmethod
            def get_element_type():
                return item_cls

        cls._registered_classes[item_cls] = ValueList
        return ValueList

    def __getitem__(self, __key: int) -> Any:
        raise NotImplementedError

    def __setitem__(self, __key: int, __value: Any):
        raise NotImplementedError

    def __iter__(self) -> typing.Iterator[Any]:
        raise NotImplementedError


class DictionaryValue:
    def __init__(
        self,
        symbol: Symbol,
        key: str,
    ):
        self._key = key
        self._owner = symbol._owner
        self._name = symbol._name
        if symbol._key:
            raise ValueError("Symbol key should not be defined.")
        if symbol._attribute:
            raise ValueError("Symbol attribute should not be defined.")

    def __eq__(self, other: Any):
        return self._generate(fn="__eq__", other=other)

    def __ne__(self, other: Any):
        return self._generate(fn="__ne__", other=other)

    def __gt__(self, other: Any):
        return self._generate(fn="__gt__", other=other)

    def __ge__(self, other: Any):
        return self._generate(fn="__ge__", other=other)

    def __lt__(self, other: Any):
        return self._generate(fn="__lt__", other=other)

    def __le__(self, other: Any):
        return self._generate(fn="__le__", other=other)

    def intersects(self, other: Any):
        return self._generate(fn="intersects", other=other)

    def inside(self, other: Any):
        return self._generate(fn="inside", other=other)

    def outside(self, other: Any):
        return self._generate(fn="outside", other=other)

    def is_none(self, other: Any):
        return self._generate(fn="is_none", other=other)

    def is_not_none(self, other: Any):
        return self._generate(fn="is_not_none", other=other)

    @property
    def area(self):
        return Float.symbolic(
            owner=self._owner, name=self._name, key=self._key, attribute="area"
        )

    def _generate(self, other: Any, fn: str):
        obj = _get_atomic_type_by_value(other)
        sym = obj.symbolic(owner=self._owner, name=self._name, key=self._key)
        return sym.__getattribute__(fn)(other)


class Dictionary(Equatable):
    def __init__(
        self,
        value: Optional[typing.Dict[str, Any]] = None,
        symbol: Symbol | None = None,
    ):
        if isinstance(value, dict):
            _value = dict()
            for k, v in value.items():
                if isinstance(v, Variable):
                    if v.is_symbolic:
                        raise ValueError(
                            "Dictionary does not accpet symbols as values."
                        )
                    _value[k] = v
                else:
                    _value[k] = _get_atomic_type_by_value(v).definite(v)
            value = _value
        super().__init__(value, symbol)

    @classmethod
    def definite(
        cls,
        value: Optional[typing.Dict[str, Any]] = None,
    ):
        value = value if value else dict()
        return super().definite(value)

    @classmethod
    def __validate__(cls, value: Any):
        if not isinstance(value, dict):
            raise TypeError(
                f"Expected type '{dict}' received type '{type(value)}'"
            )

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
