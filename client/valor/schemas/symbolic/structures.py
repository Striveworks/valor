import typing
from typing import Any, Optional

from valor.schemas.symbolic.atomics import (
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
    Nullable,
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


T = typing.TypeVar("T", bound=Variable)


class List(typing.Generic[T], Equatable):

    _registered_classes = dict()

    @classmethod
    def __class_getitem__(cls, item_class: typing.Type[T]):

        if item_class in cls._registered_classes:
            return cls._registered_classes[item_class]

        class ValueList(Equatable):
            def __init__(
                self,
                value: Optional[Any] = None,
                symbol: Optional[Symbol] = None,
            ):
                if value is not None:
                    if not isinstance(value, list):
                        raise TypeError(
                            f"Expected a value with type 'List[{item_class.__name__}]' but received type '{type(value).__name__}'"
                        )
                    vlist = []
                    for item in value:
                        if isinstance(item, item_class):
                            vlist.append(item)
                        elif isinstance(item, dict) and set(item.keys()) != {
                            "type",
                            "value",
                        }:
                            vlist.append(item_class.definite(**item))
                        else:
                            vlist.append(item_class.definite(item))
                    value = vlist
                super().__init__(value=value, symbol=symbol)

            @classmethod
            def definite(cls, value: Any):
                if value is None:
                    value = list()
                return cls(value=value)

            @classmethod
            def symbolic(
                cls,
                name: Optional[str] = None,
                key: Optional[str] = None,
                attribute: Optional[str] = None,
                owner: Optional[str] = None,
            ):
                if name is None:
                    name = f"list[{item_class.__name__.lower()}]"
                return super().symbolic(name, key, attribute, owner)

            @classmethod
            def __validate__(cls, value: list):
                if not isinstance(value, list):
                    raise TypeError(
                        f"Expected type '{list}' received type '{type(value)}'"
                    )
                for element in value:
                    if not item_class.supports(element) and not issubclass(
                        type(element), Variable
                    ):
                        raise TypeError(
                            f"Expected list elements with type '{item_class}' received type '{type(element)}'"
                        )

            @classmethod
            def decode_value(cls, value: Any):
                if not value:
                    return cls(value=[])
                return cls(
                    value=[
                        item_class.decode_value(element) for element in value
                    ]
                )

            def encode_value(self):
                return [element.encode_value() for element in self.get_value()]

            def to_dict(self) -> dict:
                if isinstance(self._value, Symbol):
                    return self._value.to_dict()
                else:
                    return {
                        "type": f"list[{item_class.__name__.lower()}]",
                        "value": self.encode_value(),
                    }

            def __getitem__(self, __key: int) -> T:
                return self.get_value()[__key]

            def __setitem__(self, __key: int, __value: Any):
                vlist = self.get_value()
                vlist[__key] = item_class.preprocess(__value)

            def __iter__(self) -> typing.Iterator[T]:
                return iter([element for element in self.get_value()])

            def __len__(self):
                return len(self.get_value())

            @staticmethod
            def get_element_type():
                return item_class

        cls._registered_classes[item_class] = ValueList
        return ValueList

    def __getitem__(self, __key: int) -> T:
        raise NotImplementedError

    def __setitem__(self, __key: int, __value: Any):
        raise NotImplementedError

    def __iter__(self) -> typing.Iterator[T]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class DictionaryValue(Nullable):
    def __init__(self, symbol: Symbol):
        if not isinstance(symbol, Symbol):
            raise ValueError(
                "DictionaryValue should only be initialized as a symbol."
            )
        if symbol._attribute:
            raise ValueError(
                "DictionaryValue symbol should not contain attribute."
            )
        if not symbol._key:
            raise ValueError("DictionaryValue symbol should contain key.")
        super().__init__(value=None, symbol=symbol)

    @classmethod
    def definite(cls, value: Any):
        raise NotImplementedError(
            "DictionaryValue should only be initialized as a symbol."
        )

    @classmethod
    def symbolic(
        cls,
        name: Optional[str] = None,
        key: Optional[str] = None,
        attribute: Optional[str] = None,
        owner: Optional[str] = None,
    ):
        return cls(
            Symbol(
                name=name if name else cls.__name__.lower(),
                key=key,
                attribute=attribute,
                owner=owner,
            )
        )

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

    def is_none(self):
        return super().is_none()

    def is_not_none(self):
        return super().is_not_none()

    @property
    def area(self):
        symbol = self.get_symbol()
        return Float.symbolic(
            owner=symbol._owner,
            name=symbol._name,
            key=symbol._key,
            attribute="area",
        )

    def _generate(self, other: Any, fn: str):
        if isinstance(other, Variable):
            obj = type(other)
        else:
            obj = _get_atomic_type_by_value(other)
        symbol = self.get_symbol()
        sym = obj.symbolic(
            owner=symbol._owner,
            name=symbol._name,
            attribute=symbol._attribute,
            key=symbol._key,
        )
        return sym.__getattribute__(fn)(other)


class Dictionary(Equatable):
    def __init__(
        self,
        value: Optional[typing.Dict[str, Any]] = None,
        symbol: Optional[Symbol] = None,
    ):
        if isinstance(value, dict):
            _value = dict()
            for k, v in value.items():
                if v is None:
                    raise ValueError(
                        "Dictionary does not accept 'None' as a value."
                    )
                elif isinstance(v, Variable):
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
        for k, v in value.items():
            if not isinstance(k, str):
                raise TypeError("Dictionary keys must be of type 'str'")

    @classmethod
    def decode_value(cls, value: dict) -> Any:
        return cls(
            {
                k: _get_atomic_type_by_name(v["type"]).decode_value(v["value"])
                for k, v in value.items()
            }
        )

    def encode_value(self) -> dict:
        return {k: v.to_dict() for k, v in self.items()}

    def __getitem__(self, key: str):
        if self.is_symbolic:
            symbol = self.get_symbol()
            return DictionaryValue.symbolic(
                owner=symbol._owner,
                name=symbol._name,
                attribute=None,
                key=key,
            )
        else:
            value = self.get_value()
            if not value:
                raise KeyError(key)
            return value[key]

    def __setitem__(self, key: str, value: Any):
        if not isinstance(value, Variable):
            obj = _get_atomic_type_by_value(value)
            value = obj.definite(value)
        self.get_value()[key] = value

    def __len__(self) -> int:
        return len(self.get_value())

    def pop(self, key: str):
        value = self.get_value()
        if not value:
            raise KeyError(key)
        return value.pop(key)

    def items(self):
        if isinstance(self._value, Symbol):
            raise NotImplementedError("Variable is symbolic")
        return self._value.items() if self._value else dict.items({})
