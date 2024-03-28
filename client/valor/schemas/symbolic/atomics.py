import datetime
from typing import Any, List, Optional, Tuple

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
        """Decode object value from JSON compatible dictionary."""
        return cls(value=value)

    def encode_value(self) -> Any:
        """Encode object value to JSON compatible dictionary."""
        return self.get_value()

    @classmethod
    def from_dict(cls, value: dict):
        if set(value.keys()) != {"type", "value"}:
            raise KeyError
        elif value["type"] != cls.__name__.lower():
            raise TypeError
        return cls.decode_value(value["value"])

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
            lhs = self.encode_value()
            rhs = other.encode_value()
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
            lhs = self.encode_value()
            rhs = other.encode_value()
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
        if self.is_symbolic:
            return hash(str(self))
        return hash(str(self.encode_value()))


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
        if not isinstance(value, (int, float, np.floating)):
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


class Point(Spatial, Equatable):
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
        return cls((value[0], value[1]))

    def encode_value(self) -> Any:
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
        return cls([(point[0], point[1]) for point in value])


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

    @classmethod
    def decode_value(cls, value: List[List[float]]):
        return cls([(point[0], point[1]) for point in value])


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
                f"Expected type 'List[List[Tuple[float, float]]]' received type '{type(value).__name__}'"
            )
        for line in value:
            LineString.__validate__(line)

    @classmethod
    def decode_value(cls, value: List[List[List[float]]]):
        return cls(
            [[(point[0], point[1]) for point in line] for line in value]
        )


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
                raise ValueError(
                    "Polygons are defined by at least 4 points with the first point being repeated at the end."
                )

    @classmethod
    def decode_value(cls, value: List[List[List[float]]]):
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
                f"Expected type 'List[List[List[Tuple[float, float]]]]' received type '{type(value).__name__}'"
            )
        for poly in value:
            Polygon.__validate__(poly)

    @classmethod
    def decode_value(cls, value: List[List[List[List[float]]]]):
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
