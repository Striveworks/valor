import numpy as np

from valor.symbolic.functions import (
    AppendableFunction,
    OneArgumentFunction,
    TwoArgumentFunction,
)
from valor.symbolic.modifiers import Variable


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
    elif isinstance(expr, (int, float, np.floating, str)):
        if isinstance(expr, np.floating):
            expr = float(expr)
        return {type(expr).__name__.lower(): expr}
    else:
        raise TypeError(type_.__name__)
