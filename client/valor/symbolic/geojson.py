from typing import Any, List, Optional, Tuple

import numpy as np

from valor.symbolic.atomics import Float
from valor.symbolic.modifiers import Spatial, Symbol, Variable


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


class GeoJSON(Variable):
    @classmethod
    def supports(cls, value: Any) -> bool:
        match value["geometry"]["type"]:
            case "Point":
                geometry_type = Point
            case "MultiPoint":
                geometry_type = MultiPoint
            case "LineString":
                geometry_type = LineString
            case "MultiLineString":
                geometry_type = MultiLineString
            case "Polygon":
                geometry_type = Polygon
            case "MultiPolygon":
                geometry_type = MultiPolygon
            case _:
                return False
        if not geometry_type.supports(value["geometry"]["coordinates"]):
            return False
        return True
