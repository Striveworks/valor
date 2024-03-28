import datetime
from typing import Any, List, Optional, Tuple, Union

import numpy as np

from valor.schemas.symbolic.functions import (
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
    """
    A symbol contains no value and is defined by the tuple (owner, name, key, attribute).
    """

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

    def __eq__(self, other):
        if not isinstance(other, Symbol):
            return False
        return (
            self._owner == other._owner
            and self._name == other._name
            and self._key == other._key
            and self._attribute == other._attribute
        )

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self) -> int:
        return hash(self.__repr__())

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
    """
    Base class for constructing variables types.

    Contains either a value or a symbol.
    """

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
        """Decode object from JSON compatible dictionary."""
        return cls(value=value)

    def encode_value(self) -> Any:
        """Encode object to JSON compatible dictionary."""
        return self.get_value()

    @classmethod
    def from_dict(cls, value: dict):
        """Decode a JSON-compatible dictionary into an instance of the variable."""
        if set(value.keys()) != {"type", "value"}:
            raise KeyError
        elif value["type"] != cls.__name__.lower():
            raise TypeError
        return cls.decode_value(value["value"])

    def to_dict(self) -> dict:
        """Encode variable to a JSON-compatible dictionary."""
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

    def __eq__(self, value: Any) -> Union["Bool", Eq]:  # type: ignore - overriding __eq__
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '__eq__'"
        )

    def __ne__(self, value: Any) -> Union["Bool", Ne]:  # type: ignore - overriding __ne__
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '__ne__'"
        )

    def __gt__(self, value: Any) -> Union["Bool", Gt]:
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '__gt__'"
        )

    def __ge__(self, value: Any) -> Union["Bool", Ge]:
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '__ge__'"
        )

    def __lt__(self, value: Any) -> Union["Bool", Lt]:
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '__lt__'"
        )

    def __le__(self, value: Any) -> Union["Bool", Le]:
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '__le__'"
        )


class Bool(Variable):
    """
    Implementation of the built-in type 'bool' as a Variable.

    Examples
    --------
    >>> Bool(True)
    """

    @classmethod
    def __validate__(cls, value: Any):
        """Validates typing."""
        if not isinstance(value, bool):
            raise TypeError(
                f"Expected type '{bool}' received type '{type(value)}'"
            )

    def __eq__(self, value: Any) -> Union["Bool", Eq]:
        other = self.preprocess(value)
        if self.is_value and other.is_value:
            return type(self)(self.get_value() is other.get_value())
        return Eq(self, other)

    def __ne__(self, value: Any) -> Union["Bool", Ne]:
        other = self.preprocess(value)
        if self.is_value and other.is_value:
            return type(self)(self.get_value() is not other.get_value())
        return Ne(self, other)

    def __and__(self, value: Any) -> Union["Bool", And]:
        other = self.preprocess(value)
        if self.is_value and other.is_value:
            return type(self)(self.get_value() and other.get_value())
        return And(self, other)

    def __or__(self, value: Any) -> Union["Bool", Or]:
        other = self.preprocess(value)
        if self.is_value and other.is_value:
            return type(self)(self.get_value() or other.get_value())
        return Or(self, other)

    def __xor__(self, value: Any) -> Union["Bool", Xor]:
        other = self.preprocess(value)
        if self.is_value and other.is_value:
            return self != value
        return Xor(self, other)

    def __invert__(self) -> Union["Bool", Negate]:
        if self.is_value:
            return type(self)(not self.get_value())
        return Negate(self)


class Equatable(Variable):
    """
    Variable modifier to handle equatable values.
    """

    def __eq__(self, value: Any) -> Union["Bool", Eq]:
        other = self.preprocess(value)
        if self.is_value and other.is_value:
            lhs = self.encode_value()
            rhs = other.encode_value()
            if lhs is None:
                return Bool(rhs is None)
            elif rhs is None:
                return Bool(lhs is None)
            else:
                return Bool(lhs == rhs)
        return Eq(self, other)

    def __ne__(self, value: Any) -> Union["Bool", Ne]:
        other = self.preprocess(value)
        if self.is_value and other.is_value:
            lhs = self.encode_value()
            rhs = other.encode_value()
            if lhs is None:
                return Bool(rhs is not None)
            elif rhs is None:
                return Bool(lhs is not None)
            else:
                return Bool(lhs != rhs)
        return Ne(self, other)

    def in_(self, vlist: List[Any]) -> Or:
        """Returns Or(*[(self == v) for v in vlist])"""
        return Or(*[(self == v) for v in vlist])

    def __hash__(self):
        if self.is_symbolic:
            return hash(str(self))
        return hash(str(self.encode_value()))


class Quantifiable(Equatable):
    """
    Variable modifier to handle quantifiable values.
    """

    def __gt__(self, value: Any) -> Union["Bool", Gt]:
        other = self.preprocess(value)
        if self.is_value and other.is_value:
            return Bool(self.get_value() > other.get_value())
        return Gt(self, other)

    def __ge__(self, value: Any) -> Union["Bool", Ge]:
        other = self.preprocess(value)
        if self.is_value and other.is_value:
            return Bool(self.get_value() >= other.get_value())
        return Ge(self, other)

    def __lt__(self, value: Any) -> Union["Bool", Lt]:
        other = self.preprocess(value)
        if self.is_value and other.is_value:
            return Bool(self.get_value() < other.get_value())
        return Lt(self, other)

    def __le__(self, value: Any) -> Union["Bool", Le]:
        other = self.preprocess(value)
        if self.is_value and other.is_value:
            return Bool(self.get_value() <= other.get_value())
        return Le(self, other)


class Nullable(Variable):
    """
    Variable modifier to handle null values.
    """

    def is_none(self) -> Union["Bool", IsNull]:
        """Conditional whether variable is 'None'"""
        if self.is_value:
            return Bool(self.get_value() is None)
        return IsNull(self)

    def is_not_none(self) -> Union["Bool", IsNotNull]:
        """Conditional whether variable is not 'None'"""
        if self.is_value:
            return Bool(self.get_value() is not None)
        return IsNotNull(self)

    def get_value(self) -> Optional[Any]:
        """Re-typed to output 'Optional[Any]'"""
        return super().get_value()


class Spatial(Variable):
    """
    Variable modifier to handle spatial values.
    """

    def intersects(self, other: Any) -> Intersects:
        """Conditional whether lhs intersects rhs."""
        return Intersects(self, self.preprocess(other))

    def inside(self, other: Any) -> Inside:
        """Conditional whether lhs is fully inside of rhs."""
        return Inside(self, self.preprocess(other))

    def outside(self, other: Any) -> Outside:
        """Conditional whether lhs is outside of rhs."""
        return Outside(self, self.preprocess(other))


class Integer(Quantifiable):
    """
    Implementation of the built-in type 'int' as a Variable.

    Examples
    --------
    >>> Integer(123)
    """

    @classmethod
    def __validate__(cls, value: Any):
        if not isinstance(value, (int, np.integer)):
            raise TypeError(
                f"Expected type '{int}' received type '{type(value)}'"
            )


class Float(Quantifiable):
    """
    Implementation of the built-in type 'float' as a Variable.

    Examples
    --------
    >>> Float(3.14)
    """

    @classmethod
    def __validate__(cls, value: Any):
        if not isinstance(value, (int, float, np.floating)):
            raise TypeError(
                f"Expected type '{float}' received type '{type(value)}'"
            )


class String(Equatable):
    """
    Implementation of the built-in type 'str' as a Variable.

    Examples
    --------
    >>> String("hello world")
    """

    @classmethod
    def __validate__(cls, value: Any):
        if not isinstance(value, str):
            raise TypeError(
                f"Expected type '{str}' received type '{type(value)}'"
            )


class DateTime(Quantifiable):
    """
    Implementation of the type 'datetime.datetime' as a Variable.

    Examples
    --------
    >>> import datetime
    >>> DateTime(datetime.datetime(year=2024, month=1, day=1))
    """

    @classmethod
    def __validate__(cls, value: Any):
        if not isinstance(value, datetime.datetime):
            raise TypeError(
                f"Expected type '{datetime.datetime}' received type '{type(value)}'"
            )

    @classmethod
    def decode_value(cls, value: str):
        """Decode object from JSON compatible dictionary."""
        return cls(value=datetime.datetime.fromisoformat(value))

    def encode_value(self):
        """Encode object to JSON compatible dictionary."""
        return self.get_value().isoformat()


class Date(Quantifiable):
    """
    Implementation of the type 'datetime.date' as a Variable.

    Examples
    --------
    >>> import datetime
    >>> Date(datetime.date(year=2024, month=1, day=1))
    """

    @classmethod
    def __validate__(cls, value: Any):
        if not isinstance(value, datetime.date):
            raise TypeError(
                f"Expected type '{datetime.date}' received type '{type(value)}'"
            )

    @classmethod
    def decode_value(cls, value: str):
        """Decode object from JSON compatible dictionary."""
        return cls(value=datetime.date.fromisoformat(value))

    def encode_value(self):
        """Encode object to JSON compatible dictionary."""
        return self.get_value().isoformat()


class Time(Quantifiable):
    """
    Implementation of the type 'datetime.time' as a Variable.

    Examples
    --------
    >>> import datetime
    >>> Time(datetime.time(hour=1, minute=1))
    """

    @classmethod
    def __validate__(cls, value: Any):
        if not isinstance(value, datetime.time):
            raise TypeError(
                f"Expected type '{datetime.time}' received type '{type(value)}'"
            )

    @classmethod
    def decode_value(cls, value: str):
        """Decode object from JSON compatible dictionary."""
        return cls(value=datetime.time.fromisoformat(value))

    def encode_value(self):
        """Encode object to JSON compatible dictionary."""
        return self.get_value().isoformat()


class Duration(Quantifiable):
    """
    Implementation of the type 'datetime.timedelta' as a Variable.

    Examples
    --------
    >>> import datetime
    >>> Duration(datetime.timedelta(seconds=100))
    """

    @classmethod
    def __validate__(cls, value: Any):
        if not isinstance(value, datetime.timedelta):
            raise TypeError(
                f"Expected type '{datetime.timedelta}' received type '{type(value)}'"
            )

    @classmethod
    def decode_value(cls, value: int):
        """Decode object from JSON compatible dictionary."""
        return cls(value=datetime.timedelta(seconds=value))

    def encode_value(self):
        """Encode object to JSON compatible dictionary."""
        return self.get_value().total_seconds()


class Point(Spatial, Equatable):
    """
    Represents a point in 2D space.

    Follows the GeoJSON specification (RFC 7946).

    Examples
    --------
    >>> Point((1,2))
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
                f"Expected type 'Tuple[float, float]' received type '{type(value).__name__}'"
            )
        elif len(value) != 2:
            raise ValueError("")
        for item in value:
            if not isinstance(item, (int, float, np.floating)):
                raise TypeError(
                    f"Expected type '{float.__name__}' received type '{type(item).__name__}'"
                )

    @classmethod
    def decode_value(cls, value: List[float]):
        """Decode object from JSON compatible dictionary."""
        return cls((value[0], value[1]))

    def encode_value(self) -> Any:
        """Encode object to JSON compatible dictionary."""
        value = self.get_value()
        return (float(value[0]), float(value[1]))

    def tuple(self):
        return self.get_value()

    def resize(
        self,
        og_img_h=10,
        og_img_w=10,
        new_img_h=100,
        new_img_w=100,
    ):
        value = self.get_value()
        h_ratio = new_img_h / og_img_h
        w_ratio = new_img_w / og_img_w
        return Point((value[0] * h_ratio, value[1] * w_ratio))

    @property
    def x(self):
        return self.get_value()[0]

    @property
    def y(self):
        return self.get_value()[1]


class MultiPoint(Spatial):
    """
    Represents a list of points.

    Follows the GeoJSON specification (RFC 7946).

    Examples
    --------
    >>> MultiPoint([(0,0), (0,1), (1,1)])
    """

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
                f"Expected 'List[Tuple[float, float]]' received type '{type(value).__name__}'"
            )
        for point in value:
            Point.__validate__(point)

    @classmethod
    def decode_value(cls, value: List[List[float]]):
        """Decode object from JSON compatible dictionary."""
        return cls([(point[0], point[1]) for point in value])


class LineString(Spatial):
    """
    Represents a line.

    Follows the GeoJSON specification (RFC 7946).

    Examples
    --------
    Create a line.
    >>> LineString([(0,0), (0,1), (1,1)])
    """

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

    @classmethod
    def decode_value(cls, value: List[List[float]]):
        """Decode object from JSON compatible dictionary."""
        return cls([(point[0], point[1]) for point in value])


class MultiLineString(Spatial):
    """
    Represents a list of lines.

    Follows the GeoJSON specification (RFC 7946).

    Examples
    --------
    Create a single line.
    >>> MultiLineString([[(0,0), (0,1), (1,1), (0,0)]])

    Create 3 lines.
    >>> MultiLineString(
    ...     [
    ...         [(0,0), (0,1), (1,1)],
    ...         [(0.1, 0.1), (0.1, 0.2), (0.2, 0.2)],
    ...         [(0.6, 0.6), (0.6, 0.7), (0.7, 0.7)],
    ...     ]
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
        if not isinstance(value, list):
            raise TypeError(
                f"Expected type 'List[List[Tuple[float, float]]]' received type '{type(value).__name__}'"
            )
        for line in value:
            LineString.__validate__(line)

    @classmethod
    def decode_value(cls, value: List[List[List[float]]]):
        """Decode object from JSON compatible dictionary."""
        return cls(
            [[(point[0], point[1]) for point in line] for line in value]
        )


class Polygon(Spatial):
    """
    Represents a polygon with a boundary and optional holes.

    Follows the GeoJSON specification (RFC 7946).

    Examples
    --------
    Create a polygon without any holes.
    >>> Polygon([[(0,0), (0,1), (1,1), (0,0)]])

    Create a polygon with 2 holes.
    >>> Polygon(
    ...     [
    ...         [(0,0), (0,1), (1,1), (0,0)],
    ...         [(0.1, 0.1), (0.1, 0.2), (0.2, 0.2), (0.1, 0.1)],
    ...         [(0.6, 0.6), (0.6, 0.7), (0.7, 0.7), (0.6, 0.6)],
    ...     ]
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
                raise ValueError(
                    "Polygons are defined by at least 4 points with the first point being repeated at the end."
                )

    @classmethod
    def decode_value(cls, value: List[List[List[float]]]):
        """Decode object from JSON compatible dictionary."""
        return cls(
            [
                [(point[0], point[1]) for point in subpolygon]
                for subpolygon in value
            ]
        )

    @property
    def area(self):
        if not isinstance(self._value, Symbol):
            raise ValueError
        return Float.symbolic(
            owner=self._value._owner,
            name=self._value._name,
            key=self._value._key,
            attribute="area",
        )

    @property
    def boundary(self):
        """"""
        return self.get_value()[0]

    @property
    def holes(self):
        return self.get_value()[1:]

    @property
    def xmin(self):
        return min([p[0] for p in self.boundary])

    @property
    def xmax(self):
        return max([p[0] for p in self.boundary])

    @property
    def ymin(self):
        return min([p[1] for p in self.boundary])

    @property
    def ymax(self):
        return max([p[1] for p in self.boundary])


class MultiPolygon(Spatial):
    """
    Represents a collection of polygons.

    Follows the GeoJSON specification (RFC 7946).

    Examples
    --------
    >>> MultiPolygon(
    ...     [
    ...         [
    ...             [(0,0), (0,1), (1,1), (0,0)]
    ...         ],
    ...         [
    ...             [(0,0), (0,1), (1,1), (0,0)],
    ...             [(0.1, 0.1), (0.1, 0.2), (0.2, 0.2), (0.1, 0.1)],
    ...             [(0.6, 0.6), (0.6, 0.7), (0.7, 0.7), (0.6, 0.6)],
    ...         ],
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
                f"Expected type 'List[List[List[Tuple[float, float]]]]' received type '{type(value).__name__}'"
            )
        for poly in value:
            Polygon.__validate__(poly)

    @classmethod
    def decode_value(cls, value: List[List[List[List[float]]]]):
        """Decode object from JSON compatible dictionary."""
        return cls(
            [
                [
                    [(point[0], point[1]) for point in subpolygon]
                    for subpolygon in polygon
                ]
                for polygon in value
            ]
        )

    @property
    def area(self):
        if not isinstance(self._value, Symbol):
            raise ValueError(
                "attribute 'area' is reserved for symbolic variables."
            )
        return Float.symbolic(
            owner=self._value._owner,
            name=self._value._name,
            key=self._value._key,
            attribute="area",
        )

    @property
    def polygons(self) -> List[Polygon]:
        return [Polygon(poly) for poly in self.get_value()]
