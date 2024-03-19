import numpy as np

from typing import Any

from valor.symbolic.functions import (
    AppendableFunction,
    OneArgumentFunction,
    TwoArgumentFunction,
)
from valor.symbolic.modifiers import Variable
from valor.symbolic.atomics import (
    Bool,
    String,
    Integer,
    Float,
    DateTime,
    Date,
    Time,
    Duration,
)
from valor.symbolic.geojson import (
    MultiPolygon,
    Polygon,
    MultiLineString,
    LineString,
    MultiPoint,
    Point,
)


def get_type_by_value(other: Any):
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
        raise NotImplementedError(str(other))


def get_type_by_name(name: str):
    match name:
        case "bool":
            return Bool
        case "string":
            return String
        case "integer":
            return Integer
        case "float":
            return Float
        case "datetime":
            return DateTime
        case "date":
            return Date
        case "time":
            return Time
        case "duration":
            return Duration
        case "multipolygon":
            return MultiPolygon
        case "polygon":
            return Polygon
        case "multilinestring":
            return MultiLineString
        case "linestring":
            return LineString
        case "multipoint":
            return MultiPoint
        case "point":
            return Point
        case _:
            raise NotImplementedError(name)


def jsonify(expr):
    type_ = type(expr)
    type_name = type_.__name__.lower()
    if issubclass(type_, Variable):
        return expr.to_dict()
    elif issubclass(type_, OneArgumentFunction):
        return {"op": type_name, "arg": jsonify(expr.arg)}
    elif issubclass(type_, TwoArgumentFunction):
        return {
            "op": type_name,
            "lhs": jsonify(expr.lhs),
            "rhs": jsonify(expr.rhs),
        }
    elif issubclass(type_, AppendableFunction):
        return {"op": type_name, "args": [jsonify(arg) for arg in expr._args]}
    else:
        raise TypeError(expr, type_.__name__)
