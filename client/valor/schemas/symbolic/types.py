import datetime
import io
import typing
import warnings
from base64 import b64decode, b64encode
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import PIL.Image

from valor.enums import TaskType
from valor.schemas.symbolic.operators import (
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

    Examples
    --------
    >>> Symbol(name="a")
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

    Examples
    --------
    Creating a valued variable.
    >>> Variable(value=...)
    or
    >>> Variable.definite(...)

    Creating a symbolic variable.
    >>> Variable(symbol=Symbol(...))
    or
    >>> Variable.symbolic(name=...)
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
        """
        Initialize variable with a value.

        Parameters
        ----------
        value : Any
            The intended value of the variable.
        """
        return cls(value=value)

    @classmethod
    def symbolic(
        cls,
        name: Optional[str] = None,
        key: Optional[str] = None,
        attribute: Optional[str] = None,
        owner: Optional[str] = None,
    ):
        """
        Initialize variable as a symbol.

        Parameters
        ----------
        name: str, optional
            The name of the symbol. Defaults to the name of the parent class.
        key: str, optional
            An optional dictionary key.
        attribute: str, optional
            An optional attribute name.
        owner: str, optional
            An optional name describing the class that owns this symbol.
        """
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
        """
        Validates typing.

        Parameters
        ----------
        value : Any
            The value to validate.

        Raises
        ------
        NotImplementedError
            This function is not implemented in the base class.
        """
        raise NotImplementedError(
            f"Variable of type `{cls.__name__}` cannot be assigned a value."
        )

    @classmethod
    def supports(cls, value: Any) -> bool:
        """
        Checks if value is a supported type.

        Returns
        -------
        bool
        """
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
        """
        Retrieve value, if it exists.

        Raises
        ------
        TypeError
            If the variable is symbolic.
        """
        if isinstance(self._value, Symbol):
            raise TypeError(
                f"{type(self).__name__} is symbolic and does not contain a value."
            )
        return self._value

    def get_symbol(self) -> Symbol:
        """
        Retrieve symbol, if it exists.

        Raises
        ------
        TypeError
            If the variable is a valued object.

        Returns
        -------
        Symbol
        """
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

    Parameters
    ----------
    value : bool, optional
        A boolean value.
    symbol : Symbol, optional
        A symbolic representation.

    Examples
    --------
    >>> Bool(True)
    """

    @classmethod
    def __validate__(cls, value: Any):
        """
        Validates typing.

        Parameters
        ----------
        value : Any
            The value to validate.

        Raises
        ------
        TypeError
            If the value type is not supported.
        """
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

    Parameters
    ----------
    value : int, optional
        A integer value.
    symbol : Symbol, optional
        A symbolic representation.

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

    Parameters
    ----------
    value : float, optional
        A float value.
    symbol : Symbol, optional
        A symbolic representation.

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

    Parameters
    ----------
    value : str, optional
        A string value.
    symbol : Symbol, optional
        A symbolic representation.

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

    Parameters
    ----------
    value : datetime.datetime, optional
        A datetime value.
    symbol : Symbol, optional
        A symbolic representation.

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

    Parameters
    ----------
    value : datetime.date, optional
        A date value.
    symbol : Symbol, optional
        A symbolic representation.

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

    Parameters
    ----------
    value : datetime.time, optional
        A time value.
    symbol : Symbol, optional
        A symbolic representation.

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

    Parameters
    ----------
    value : datetime.timedelta, optional
        A time duration.
    symbol : Symbol, optional
        A symbolic representation.

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

    Parameters
    ----------
    value : Tuple[float, float], optional
        A point.
    symbol : Symbol, optional
        A symbolic representation.

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

    Parameters
    ----------
    value : List[Tuple[float, float]], optional
        A multipoint.
    symbol : Symbol, optional
        A symbolic representation.

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

    Parameters
    ----------
    value : List[Tuple[float, float]], optional
        A linestring.
    symbol : Symbol, optional
        A symbolic representation.

    Methods
    -------
    colorspace(c='rgb')
        Represent the photo in the given colorspace.
    gamma(n=1.0)
        Change the photo's gamma exposure.

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

    Parameters
    ----------
    value : List[List[Tuple[float, float]]], optional
        A multilinestring.
    symbol : Symbol, optional
        A symbolic representation.

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

    Parameters
    ----------
    value : List[List[Tuple[float, float]]], optional
        A polygon.
    symbol : Symbol, optional
        A symbolic representation.

    Attributes
    ----------
    area
    boundary
    holes
    xmin
    xmax
    ymin
    ymax

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
    def area(self) -> Float:
        """
        Symbolic representation of area.
        """
        if not isinstance(self._value, Symbol):
            raise ValueError
        return Float.symbolic(
            owner=self._value._owner,
            name=self._value._name,
            key=self._value._key,
            attribute="area",
        )

    @property
    def boundary(self) -> List[Tuple[float, float]]:
        """
        The boundary of the polygon.

        Returns
        -------
        List[Tuple(float, float)]
            A list of points.
        """
        value = self.get_value()
        if value is None:
            raise ValueError("Polygon is 'None'")
        return value[0]

    @property
    def holes(self) -> List[List[Tuple[float, float]]]:
        """
        Any holes in the polygon.

        Returns
        -------
        List[List[Tuple(float, float)]]
            A list of holes.
        """
        value = self.get_value()
        if value is None:
            raise ValueError("Polygon is 'None'")
        return value[1:]

    @property
    def xmin(self) -> float:
        """
        Minimum x-value.

        Returns
        -------
        float
        """
        return min([p[0] for p in self.boundary])

    @property
    def xmax(self) -> float:
        """
        Maximum x-value.

        Returns
        -------
        float
        """
        return max([p[0] for p in self.boundary])

    @property
    def ymin(self) -> float:
        """
        Minimum y-value.

        Returns
        -------
        float
        """
        return min([p[1] for p in self.boundary])

    @property
    def ymax(self) -> float:
        """
        Maximum y-value.

        Returns
        -------
        float
        """
        return max([p[1] for p in self.boundary])


class MultiPolygon(Spatial):
    """
    Represents a collection of polygons.

    Follows the GeoJSON specification (RFC 7946).

    Parameters
    ----------
    value : List[List[List[Tuple[float, float]]]], optional
        A list of polygons.
    symbol : Symbol, optional
        A symbolic representation.

    Attributes
    ----------
    area
    polygons

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
        """
        Symbolic representation of area.
        """
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
        """
        List of the component polygons.

        Returns
        -------
        List[Polygon]
        """
        return [Polygon(poly) for poly in self.get_value()]


class Score(Float, Nullable):
    """
    Implements a score annotation as a nullable floating-point variable.

    Parameters
    ----------
    value : float, optional
        A score value between 0.0 and 1.0.
    symbol : Symbol, optional
        A symbolic representation.

    Examples
    --------
    >>> Score(0.9)
    >>> Score(None)
    """

    @classmethod
    def __validate__(cls, value: Any):
        """
        Validates typing.

        Parameters
        ----------
        value : Any
            The value to validate.

        Raises
        ------
        TypeError
            If the value type is not supported.
        """
        if value is not None:
            Float.__validate__(value)
            if value < 0.0:
                raise ValueError("score must be non-negative")
            elif value > 1.0:
                raise ValueError("score must not exceed 1.0")

    @classmethod
    def decode_value(cls, value: Optional[Union[float, np.floating]]):
        """Decode object from JSON compatible dictionary."""
        if value is None:
            return cls(None)
        return super().decode_value(value)


class TaskTypeEnum(String):
    """
    Variable wrapper for 'valor.enums.TaskType'.

    Parameters
    ----------
    value : Union[str, valor.enums.TaskType], optional
        A task type enum value.
    symbol : Symbol, optional
        A symbolic representation.

    Examples
    --------
    >>> from valor.enums import TaskType
    >>> TaskTypeEnum(TaskType.CLASSIFICATION)
    >>> TaskTypeEnum("classification")
    """

    def __init__(
        self,
        value: Optional[Any] = None,
        symbol: Optional[Symbol] = None,
    ):
        if isinstance(value, str):
            value = TaskType(value)
        super().__init__(value=value, symbol=symbol)

    @classmethod
    def __validate__(cls, value: Any):
        """
        Validates typing.

        Parameters
        ----------
        value : Any
            The value to validate.

        Raises
        ------
        TypeError
            If the value type is not supported.
        """
        if not isinstance(value, TaskType):
            raise TypeError(
                f"Expected value with type '{TaskType.__name__}' received type '{type(value).__name__}'"
            )

    @classmethod
    def decode_value(cls, value: str):
        """Decode object from JSON compatible dictionary."""
        return cls(TaskType(value))

    def encode_value(self) -> Any:
        """Encode object to JSON compatible dictionary."""
        return self.get_value().value


class BoundingBox(Polygon, Nullable):
    """
    Bounding Box Annotation.

    Implements a nullable polygon with 4 unique points. Note that this does not need to be axis-aligned.

    Parameters
    ----------
    value : List[List[Tuple[float, float]]], optional
        An polygon value representing a box.
    symbol : Symbol, optional
        A symbolic representation.

    Attributes
    ----------
    area
    polygon
    boundary
    holes
    xmin
    xmax
    ymin
    ymax

    Examples
    --------
    >>> BoundingBox([[(0,0), (0,1), (1,1), (1,0), (0,0)]])

    Create a BoundingBox using extrema.
    >>> BoundingBox.from_extrema(
    ...     xmin=0, xmax=1,
    ...     ymin=0, ymax=1,
    ... )
    """

    @classmethod
    def __validate__(cls, value: Any):
        """
        Validates typing.

        Parameters
        ----------
        value : Any
            The value to validate.

        Raises
        ------
        TypeError
            If the value type is not supported.
        """
        if value is not None:
            Polygon.__validate__(value)
            if len(value) != 1:
                raise ValueError("Bounding Box should not contain holes.")
            elif len(value[0]) != 5:
                raise ValueError(
                    "Bounding Box should consist of four unique points."
                )

    @classmethod
    def decode_value(cls, value: Optional[List[List[List[float]]]]):
        """Decode object from JSON compatible dictionary."""
        if value is None:
            return cls(None)
        return super().decode_value(value)

    @classmethod
    def from_extrema(
        cls,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
    ):
        """
        Create a BoundingBox from extrema values.

        Parameters
        ----------
        xmin : float
            Minimum x-coordinate of the bounding box.
        xmax : float
            Maximum x-coordinate of the bounding box.
        ymin : float
            Minimum y-coordinate of the bounding box.
        ymax : float
            Maximum y-coordinate of the bounding box.

        Returns
        -------
        BoundingBox
            A BoundingBox created from the provided extrema values.
        """
        points = [
            [
                (xmin, ymin),
                (xmax, ymin),
                (xmax, ymax),
                (xmin, ymax),
                (xmin, ymin),
            ]
        ]
        return cls(value=points)

    @property
    def polygon(self) -> Optional[Polygon]:
        """
        The box defined as a 'Polygon'.

        Returns
        -------
        Polygon | None
            The component polygon or 'None' if it doesn't exist.
        """
        value = self.get_value()
        return Polygon(value) if value else None


class BoundingPolygon(Polygon, Nullable):
    """
    Represents a polygon with a boundary and optional holes.

    Parameters
    ----------
    value : List[List[Tuple[float, float]]], optional
        An polygon value.
    symbol : Symbol, optional
        A symbolic representation.

    Attributes
    ----------
    area
    polygon
    boundary
    holes
    xmin
    xmax
    ymin
    ymax

    Raises
    ------
    TypeError
        If `boundary` is not a `BasicPolygon` or cannot be converted to one.
        If `holes` is not a list, or an element in `holes` is not a `BasicPolygon`.

    Examples
    --------
    Create a polygon.
    >>> polygon1 = Polygon(
    ...     value = [[
    ...         (0, 1),
    ...         (1, 1),
    ...         (1, 0),
    ...         (0, 1),
    ...     ]],
    ... )

    Create a polygon with holes.
    >>> polygon2 = Polygon(
    ...     value = [[
    ...         (0, 1),
    ...         (1, 1),
    ...         (1, 0),
    ...         (0, 1),
    ...     ],[
    ...         (0, 0.5),
    ...         (0.5, 0.5),
    ...         (0.5, 0),
    ...         (0, 0.5),
    ...     ]],
    ... )
    """

    @classmethod
    def __validate__(cls, value: Any):
        """
        Validates typing.

        Parameters
        ----------
        value : Any
            The value to validate.

        Raises
        ------
        TypeError
            If the value type is not supported.
        """
        if value is not None:
            Polygon.__validate__(value)

    @classmethod
    def decode_value(cls, value: Optional[List[List[List[float]]]]):
        """Decode object from JSON compatible dictionary."""
        if value is None:
            return cls(None)
        return super().decode_value(value)

    @property
    def polygon(self) -> Optional[Polygon]:
        """
        The bounding polygon defined as a 'Polygon'.

        Returns
        -------
        Polygon | None
            The component polygon or 'None' if it doesn't exist.
        """
        value = self.get_value()
        return Polygon(value) if value else None


class Raster(Spatial, Nullable):
    """
    Represents a binary mask.

    Parameters
    ----------
    value : Dict[str, Union[np.ndarray, str, None]], optional
        An raster value.
    symbol : Symbol, optional
        A symbolic representation.

    Parameters
    ----------
    area
    array
    geometry
    height
    width

    Raises
    ------
    TypeError
        If `encoding` is not a string.

    Examples
    --------
    Generate a random mask.
    >>> import numpy.random
    >>> height = 640
    >>> width = 480
    >>> array = numpy.random.rand(height, width)

    Convert to binary mask.
    >>> mask = (array > 0.5)

    Create Raster.
    >>> Raster.from_numpy(mask)
    """

    @classmethod
    def __validate__(cls, value: Any):
        """
        Validates typing.

        Parameters
        ----------
        value : Any
            The value to validate.

        Raises
        ------
        TypeError
            If the value type is not supported.
        """
        if value is not None:
            if not isinstance(value, dict):
                raise TypeError(
                    "Raster should contain a dictionary describing a mask and optionally a geometry."
                )
            elif set(value.keys()) != {"mask", "geometry"}:
                raise ValueError(
                    "Raster should be described by a dictionary with keys 'mask' and 'geometry'"
                )
            elif not isinstance(value["mask"], np.ndarray):
                raise TypeError(
                    f"Expected mask to have type '{np.ndarray}' receieved type '{value['mask']}'"
                )
            elif len(value["mask"].shape) != 2:
                raise ValueError("raster only supports 2d arrays")
            elif value["mask"].dtype != bool:
                raise ValueError(
                    f"Expecting a binary mask (i.e. of dtype bool) but got dtype {value['mask'].dtype}"
                )
            elif (
                value["geometry"] is not None
                and not Polygon.supports(value["geometry"])
                and not MultiPolygon.supports(value["geometry"])
            ):
                raise TypeError(
                    "Expected geometry to conform to either Polygon or MultiPolygon or be 'None'"
                )

    def encode_value(self) -> Any:
        """Encode object to JSON compatible dictionary."""
        value = self.get_value()
        if value is not None:
            f = io.BytesIO()
            PIL.Image.fromarray(value["mask"]).save(f, format="PNG")
            f.seek(0)
            mask_bytes = f.read()
            f.close()
            value = {
                "mask": b64encode(mask_bytes).decode(),
                "geometry": value["geometry"],
            }
        return value

    @classmethod
    def decode_value(cls, value: Any):
        """Decode object from JSON compatible dictionary."""
        if value is not None:
            mask_bytes = b64decode(value["mask"])
            with io.BytesIO(mask_bytes) as f:
                img = PIL.Image.open(f)
                value = {
                    "mask": np.array(img),
                    "geometry": value["geometry"],
                }
        return cls(value=value)

    @classmethod
    def from_numpy(cls, mask: np.ndarray):
        """
        Create a Raster object from a NumPy array.

        Parameters
        ----------
        mask : np.ndarray
            The 2D binary array representing the mask.

        Returns
        -------
        Raster

        Raises
        ------
        ValueError
            If the input array is not 2D or not of dtype bool.
        """
        return cls(value={"mask": mask, "geometry": None})

    @classmethod
    def from_geometry(
        cls,
        geometry: Union[Polygon, MultiPolygon],
        height: int,
        width: int,
    ):
        """
        Create a Raster object from a geometric mask.

        Parameters
        ----------
        geometry : Union[Polygon, MultiPolygon]
            Defines the bitmask as a geometry. Overrides any existing mask.
        height : int
            The intended height of the binary mask.
        width : int
            The intended width of the binary mask.

        Returns
        -------
        Raster
        """
        bitmask = np.full((int(height), int(width)), False)
        return cls(value={"mask": bitmask, "geometry": geometry.get_value()})

    @property
    def area(self) -> Float:
        """
        Symbolic representation of area.
        """
        if not isinstance(self._value, Symbol):
            raise ValueError
        return Float.symbolic(
            owner=self._value._owner,
            name=self._value._name,
            key=self._value._key,
            attribute="area",
        )

    @property
    def array(self) -> Optional[np.ndarray]:
        """
        The bitmask as a numpy array.

        Returns
        -------
        Optional[np.ndarray]
            A 2D binary array representing the mask if it exists.
        """
        value = self.get_value()
        if value is not None:
            if value["geometry"] is not None:
                warnings.warn(
                    "array does not hold information as this is a geometry-based raster",
                    RuntimeWarning,
                )
            return value["mask"]
        warnings.warn("raster has no value", RuntimeWarning)
        return None

    @property
    def geometry(self) -> Optional[Union[Polygon, MultiPolygon]]:
        """
        The geometric mask if it exists.

        Returns
        -------
        Polygon | MultiPolygon | None
            The geometry if it exists.
        """
        value = self.get_value()
        if value is not None:
            return value["geometry"]
        warnings.warn("raster has no value", RuntimeWarning)
        return None

    @property
    def height(self) -> Optional[int]:
        """Returns the height of the raster if it exists."""
        array = self.array
        if array is not None:
            return array.shape[0]
        return None

    @property
    def width(self) -> Optional[int]:
        """Returns the width of the raster if it exists."""
        array = self.array
        if array is not None:
            return array.shape[1]
        return None


class Embedding(Spatial, Nullable):
    """
    Represents a model embedding.

    Parameters
    ----------
    value : List[float], optional
        An embedding value.
    symbol : Symbol, optional
        A symbolic representation.
    """

    @classmethod
    def __validate__(cls, value: Any):
        """
        Validates typing.

        Parameters
        ----------
        value : Any
            The value to validate.

        Raises
        ------
        TypeError
            If the value type is not supported.
        """
        if value is not None:
            if not isinstance(value, list):
                raise TypeError(
                    f"Expected type '{Optional[typing.List[float]]}' received type '{type(value)}'"
                )
            elif len(value) < 1:
                raise ValueError(
                    "embedding should have at least one dimension"
                )

    @classmethod
    def decode_value(cls, value: Optional[List[float]]):
        """Decode object from JSON compatible dictionary."""
        if value is None:
            return cls(None)
        return super().decode_value(value)
