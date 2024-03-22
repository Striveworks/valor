import datetime
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from valor.symbolic.functions import (
    And,
    Eq,
    Ge,
    Gt,
    Inside,
    Intersects,
    IsNotNull,
    IsNull,
    Le,
    Lt,
    Ne,
    Negate,
    Or,
    Outside,
    Xor,
)


class Symbol:
    def __init__(
        self,
        name: str,
        key: Optional[str] = None,
        attribute: Optional[str] = None,
        owner: Optional[str] = None,
    ):
        self._owner = owner.lower() if owner else None
        self._name = name.lower()
        self._key = key.lower() if key else None
        self._attribute = attribute.lower() if attribute else None

    def __repr__(self):
        ret = f"{type(self).__name__}("
        if self._owner:
            ret += f"owner='{self._owner}', "
        ret += f"name='{self._name}'"
        if self._key:
            ret += f", key='{self._key}'"
        if self._attribute:
            ret += f", attribute='{self._attribute}'"
        ret += ")"
        return ret

    def __str__(self):
        ret = ""
        if self._owner:
            ret += f"{self._owner}."
        ret += self._name
        if self._key is not None:
            ret += f"['{self._key}']"
        if self._attribute:
            ret += f".{self._attribute}"
        return ret

    def to_dict(self):
        return {
            "type": "symbol",
            "value": {
                "owner": self._owner,
                "name": self._name,
                "key": self._key,
                "attribute": self._attribute,
            },
        }


class Variable:
    def __init__(
        self,
        value: Optional[Any] = None,
        symbol: Optional[Symbol] = None,
    ):
        if (value is not None) and (symbol is not None):
            raise TypeError(
                f"{type(self).__name__} cannot be symbolic and contain a value at the same time."
            )
        elif symbol is not None and not isinstance(symbol, Symbol):
            raise TypeError(
                f"{type(self).__name__} symbol should have type 'Symbol' or be set to 'None'"
            )
        elif symbol is None:
            self.__validate__(value)
        self._value = symbol if symbol else value

    def __repr__(self) -> str:
        return self._value.__repr__()

    def __str__(self) -> str:
        return str(self._value)

    @classmethod
    def definite(
        cls,
        value: Any,
    ):
        """Initialize variable with a value."""
        if not cls.supports(value):
            raise TypeError(
                f"Value `{value}` with type `{type(value).__name__}` is not a supported type for `{cls.__name__}`"
            )
        return cls(value=value)

    @classmethod
    def symbolic(
        cls,
        name: Optional[str] = None,
        key: Optional[str] = None,
        attribute: Optional[str] = None,
        owner: Optional[str] = None,
    ):
        """Initialize variable as a symbol."""
        name = cls.__name__ if not name else name
        return cls(
            symbol=Symbol(
                name=name,
                key=key,
                attribute=attribute,
                owner=owner,
            )
        )

    @classmethod
    def list(cls):

        item_class = cls

        class ValueList(Variable):
            @classmethod
            def definite(cls, value: Any):
                if value is None:
                    value = list()
                return cls(value=value)

            @classmethod
            def __validate__(cls, value: List[Any]):
                if not isinstance(value, list):
                    raise TypeError(
                        f"Expected type '{list}' received type '{type(value)}'"
                    )
                for element in value:
                    if not isinstance(element, item_class):
                        raise TypeError(
                            f"Expected list elements with type '{item_class}' received type '{type(element)}'"
                        )

            @classmethod
            def decode_value(cls, value: Any):
                if not value:
                    return []
                if issubclass(type(value), Variable):
                    return [
                        item_class.decode_value(element) for element in value
                    ]

            def encode_value(self):
                return [element.encode_value() for element in self.get_value()]

            def __getitem__(self, __key: int) -> cls:
                return self.get_value()[__key]

            def __setitem__(self, __key: int, __value: cls):
                value = self.get_value()
                if value is None:
                    raise TypeError
                value[__key] = __value

            def __iter__(self) -> Iterator[cls]:
                return iter([element for element in self.get_value()])

        return ValueList

    @classmethod
    def preprocess(cls, value: Any):
        """
        This method converts any type to an instance of the variable class.

        It will raise an error if a value is unsupported.

        Parameters
        ----------
        value : Any
            An instance of a variable, value, or symbol.

        Raises
        ------
        TypeError
            If a value or variable instance is of an incompatible type.
        """
        if isinstance(value, cls):
            return value
        elif isinstance(value, Symbol):
            return cls(symbol=value)
        elif cls.supports(value):
            return cls(value=value)
        raise TypeError(
            f"{cls.__name__} does not support operations with value '{value}' of type '{type(value).__name__}'."
        )

    @classmethod
    def __validate__(cls, value: Any):
        """Validates typing."""
        raise NotImplementedError(
            f"Variable of type `{cls.__name__}` cannot be assigned a value."
        )

    @classmethod
    def supports(cls, value: Any) -> bool:
        """Checks if value is a supported type."""
        try:
            cls.__validate__(value)
        except (TypeError, ValueError):
            return False
        else:
            return True

    @classmethod
    def decode_value(cls, value: Any):
        """Decode object value from JSON compatible dictionary."""
        return cls(value=value)

    def encode_value(self) -> Any:
        """Encode object value to JSON compatible dictionary."""
        return self.get_value()

    def to_dict(self) -> dict:
        if isinstance(self._value, Symbol):
            return self._value.to_dict()
        else:
            return {
                "type": type(self).__name__.lower(),
                "value": self.encode_value(),
            }

    @property
    def is_symbolic(self) -> bool:
        """Returns whether variable is symbolic."""
        return isinstance(self._value, Symbol)

    @property
    def is_value(self) -> bool:
        """Returns whether variable contains a value."""
        return not isinstance(self._value, Symbol)

    def get_value(self) -> Any:
        """Retrieve value, if it exists."""
        if isinstance(self._value, Symbol):
            raise TypeError(
                f"{type(self).__name__} is symbolic and does not contain a value."
            )
        return self._value

    def get_symbol(self) -> Symbol:
        """Retrieve symbol, if it exists."""
        if not isinstance(self._value, Symbol):
            raise TypeError(f"{type(self).__name__} is a valued object.")
        return self._value

    def __eq__(self, value: Any):
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '__eq__'"
        )

    def __ne__(self, value: Any):
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '__ne__'"
        )

    def __gt__(self, value: Any):
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '__gt__'"
        )

    def __ge__(self, value: Any):
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '__ge__'"
        )

    def __lt__(self, value: Any):
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '__lt__'"
        )

    def __le__(self, value: Any):
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '__le__'"
        )


class Bool(Variable):
    @classmethod
    def __validate__(cls, value: Any):
        if not isinstance(value, bool):
            raise TypeError(
                f"Expected type '{bool}' received type '{type(value)}'"
            )

    def __eq__(self, value: Any):
        other = self.preprocess(value)
        if self.is_value and other.is_value:
            return type(self)(self.get_value() is other.get_value())
        return Eq(self, other)

    def __ne__(self, value: Any):
        other = self.preprocess(value)
        if self.is_value and other.is_value:
            return type(self)(self.get_value() is not other.get_value())
        return Ne(self, other)

    def __and__(self, value: Any):
        other = self.preprocess(value)
        if self.is_value and other.is_value:
            return type(self)(self.get_value() and other.get_value())
        return And(self, other)

    def __or__(self, value: Any):
        other = self.preprocess(value)
        if self.is_value and other.is_value:
            return type(self)(self.get_value() or other.get_value())
        return Or(self, other)

    def __xor__(self, value: Any):
        other = self.preprocess(value)
        if self.is_value and other.is_value:
            return self != value
        return Xor(self, other)

    def __invert__(self):
        if self.is_value:
            return type(self)(not self.get_value())
        return Negate(self)


class Equatable(Variable):
    def __eq__(self, value: Any):
        other = self.preprocess(value)
        if self.is_value and other.is_value:
            lhs = self.get_value()
            rhs = other.get_value()
            if lhs is None:
                return Bool(rhs is None)
            elif rhs is None:
                return Bool(lhs is None)
            else:
                return Bool(lhs == rhs)
        return Eq(self, other)

    def __ne__(self, value: Any):
        other = self.preprocess(value)
        if self.is_value and other.is_value:
            lhs = self.get_value()
            rhs = other.get_value()
            if lhs is None:
                return Bool(rhs is not None)
            elif rhs is None:
                return Bool(lhs is not None)
            else:
                return Bool(lhs != rhs)
        return Ne(self, other)

    def in_(self, vlist: List[Any]):
        return Or(*[(self == v) for v in vlist])

    def __hash__(self):
        return hash(str(self))


class Quantifiable(Equatable):
    def __gt__(self, value: Any):
        other = self.preprocess(value)
        if self.is_value and other.is_value:
            return Bool(self.get_value() > other.get_value())
        return Gt(self, other)

    def __ge__(self, value: Any):
        other = self.preprocess(value)
        if self.is_value and other.is_value:
            return Bool(self.get_value() >= other.get_value())
        return Ge(self, other)

    def __lt__(self, value: Any):
        other = self.preprocess(value)
        if self.is_value and other.is_value:
            return Bool(self.get_value() < other.get_value())
        return Lt(self, other)

    def __le__(self, value: Any):
        other = self.preprocess(value)
        if self.is_value and other.is_value:
            return Bool(self.get_value() <= other.get_value())
        return Le(self, other)


class Nullable(Variable):
    def is_none(self):
        if self.is_value:
            return Bool(self.get_value() is None)
        return IsNull(self)

    def is_not_none(self):
        if self.is_value:
            return Bool(self.get_value() is not None)
        return IsNotNull(self)

    def get_value(self) -> Optional[Any]:
        """Re-typed to output 'Optional[Any]'"""
        return super().get_value()


class Spatial(Variable):
    def intersects(self, other: Any):
        return Intersects(self, self.preprocess(other))

    def inside(self, other: Any):
        return Inside(self, self.preprocess(other))

    def outside(self, other: Any):
        return Outside(self, self.preprocess(other))


class Integer(Quantifiable):
    @classmethod
    def __validate__(cls, value: Any):
        if not isinstance(value, (int, np.integer)):
            raise TypeError(
                f"Expected type '{int}' received type '{type(value)}'"
            )


class Float(Quantifiable):
    @classmethod
    def __validate__(cls, value: Any):
        if not isinstance(value, (float, np.floating)):
            raise TypeError(
                f"Expected type '{float}' received type '{type(value)}'"
            )


class String(Equatable):
    @classmethod
    def __validate__(cls, value: Any):
        if not isinstance(value, str):
            raise TypeError(
                f"Expected type '{str}' received type '{type(value)}'"
            )


class DateTime(Quantifiable):
    @classmethod
    def __validate__(cls, value: Any):
        if not isinstance(value, datetime.datetime):
            raise TypeError(
                f"Expected type '{datetime.datetime}' received type '{type(value)}'"
            )

    @classmethod
    def decode_value(cls, value: str):
        return cls(value=datetime.datetime.fromisoformat(value))

    def encode_value(self):
        return self.get_value().isoformat()


class Date(Quantifiable):
    @classmethod
    def __validate__(cls, value: Any):
        if not isinstance(value, datetime.date):
            raise TypeError(
                f"Expected type '{datetime.date}' received type '{type(value)}'"
            )

    @classmethod
    def decode_value(cls, value: str):
        return cls(value=datetime.date.fromisoformat(value))

    def encode_value(self):
        return self.get_value().isoformat()


class Time(Quantifiable):
    @classmethod
    def __validate__(cls, value: Any):
        if not isinstance(value, datetime.time):
            raise TypeError(
                f"Expected type '{datetime.time}' received type '{type(value)}'"
            )

    @classmethod
    def decode_value(cls, value: str):
        return cls(value=datetime.time.fromisoformat(value))

    def encode_value(self):
        return self.get_value().isoformat()


class Duration(Quantifiable):
    @classmethod
    def __validate__(cls, value: Any):
        if not isinstance(value, datetime.timedelta):
            raise TypeError(
                f"Expected type '{datetime.timedelta}' received type '{type(value)}'"
            )

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
    def __validate__(cls, value: Any):
        if not isinstance(value, tuple):
            raise TypeError(
                f"Expected type '{Tuple[float, float]}' received type '{type(value)}'"
            )
        elif len(value) != 2:
            raise ValueError("")
        for item in value:
            if not isinstance(item, (int, float, np.floating)):
                raise TypeError(
                    f"Expected type '{float}' received type '{type(item)}'"
                )


class MultiPoint(Spatial):
    def __init__(
        self,
        value: Optional[List[Tuple[float, float]]] = None,
        symbol: Optional[Symbol] = None,
    ):
        super().__init__(value=value, symbol=symbol)

    @classmethod
    def __validate__(cls, value: Any):
        if not isinstance(value, list):
            raise TypeError(
                f"Expected '{List[Tuple[float, float]]}' received type '{type(value)}'"
            )
        for point in value:
            Point.__validate__(point)


class LineString(Spatial):
    def __init__(
        self,
        value: Optional[List[Tuple[float, float]]] = None,
        symbol: Optional[Symbol] = None,
    ):
        super().__init__(value=value, symbol=symbol)

    @classmethod
    def __validate__(cls, value: Any):
        MultiPoint.__validate__(value)
        if len(value) < 2:
            raise ValueError(
                "At least two points are required to make a line."
            )


class MultiLineString(Spatial):
    def __init__(
        self,
        value: Optional[List[List[Tuple[float, float]]]] = None,
        symbol: Optional[Symbol] = None,
    ):
        super().__init__(value=value, symbol=symbol)

    @classmethod
    def __validate__(cls, value: Any):
        if not isinstance(value, list):
            raise TypeError(
                f"Expected type '{List[List[Tuple[float, float]]]}' received type '{type(value)}'"
            )
        for line in value:
            LineString.__validate__(line)


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
    def __validate__(cls, value: Any):
        MultiLineString.__validate__(value)
        for line in value:
            LineString.__validate__(line)
            if not (len(line) >= 4 and line[0] == line[-1]):
                raise ValueError("Polygon's must contain four unique points.")

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
    def __validate__(cls, value: Any):
        if not isinstance(value, list):
            raise TypeError(
                f"Expected type '{List[List[List[Tuple[float, float]]]]}' received type '{type(value)}'"
            )
        for poly in value:
            Polygon.__validate__(poly)

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
        value: Optional[Dict[str, Any]] = None,
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
        value: Optional[Dict[str, Any]] = None,
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
