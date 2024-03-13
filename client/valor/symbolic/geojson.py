import numpy as np
from typing import Any, List, Tuple, Optional

from valor.symbolic.modifiers import Variable, Spatial
from valor.symbolic.attributes import Area


class Point(Spatial):

    def __init__(
        self,
        value: Optional[Tuple[float, float]] = None,
        **kwargs,
    ):
        super().__init__(value=value, **kwargs)

    @staticmethod
    def supports(value: Any) -> bool:
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
        **kwargs,
    ):
        super().__init__(value=value, **kwargs)

    @staticmethod
    def supports(value: Any) -> bool:
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
        **kwargs,
    ):
        super().__init__(value=value, **kwargs)

    @staticmethod
    def supports(value: Any) -> bool:
        if MultiPoint.supports(value):
            return len(value) >= 2
        else:
            return issubclass(type(value), LineString)
        

class MultiLineString(Spatial):

    def __init__(
        self,
        value: Optional[List[List[Tuple[float, float]]]] = None,
        **kwargs,
    ):
        super().__init__(value=value, **kwargs)

    @staticmethod
    def supports(value: Any) -> bool:
        if isinstance(value, list):
            for line in value:
                if not LineString.supports(line):
                    return False
            return True
        else:
            return issubclass(type(value), MultiLineString)


class Polygon(Spatial, Area):

    def __init__(
        self,
        value: Optional[List[List[Tuple[float, float]]]] = None,
        **kwargs,
    ):
        super().__init__(value=value, **kwargs)

    @staticmethod
    def supports(value: Any) -> bool:
        if MultiLineString.supports(value):
            for line in value:
                if not (
                    len(line) >= 4
                    and line[0] == line[-1]
                ):
                    return False
            return True
        else:
            return issubclass(type(value), Polygon)


class MultiPolygon(Spatial, Area):

    def __init__(
        self,
        value: Optional[List[List[List[Tuple[float, float]]]]] = None,
        **kwargs,
    ):
        super().__init__(value=value, **kwargs)

    @staticmethod
    def supports(value: Any) -> bool:
        if isinstance(value, list):
            for poly in value:
                if not Polygon.supports(poly):
                    return False
            return True
        else:
            return issubclass(type(value), MultiPolygon)


class GeoJSON(Variable):

    @staticmethod
    def supports(value: Any) -> bool:
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
    