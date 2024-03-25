import datetime

import pytest

from valor.symbolic import (
    Float,
    LineString,
    List,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    Symbol,
    Variable,
)
from valor.symbolic.functions import AppendableFunction, TwoArgumentFunction
from valor.symbolic.structures import Dictionary, DictionaryValue


def get_function_name(fn: str) -> str:
    fns = {
        "__eq__": "eq",
        "__ne__": "ne",
        "__and__": "and",
        "__or__": "or",
        "__xor__": "xor",
        "__gt__": "gt",
        "__ge__": "ge",
        "__lt__": "lt",
        "__le__": "le",
        "is_none": "isnull",
        "is_not_none": "isnotnull",
        "intersects": "intersects",
        "inside": "inside",
        "outside": "outside",
    }
    return fns[fn]


def _test_to_dict(objcls, value):
    # test __init__
    assert objcls(value).to_dict() == {
        "type": objcls.__name__.lower(),
        "value": objcls(value).encode_value(),
    }
    # test definite
    assert objcls.definite(value).to_dict() == {
        "type": objcls.__name__.lower(),
        "value": objcls(value).encode_value(),
    }
    # test symbolic
    assert objcls.symbolic().to_dict() == {
        "type": "symbol",
        "value": {
            "owner": None,
            "name": objcls.__name__.lower(),
            "key": None,
            "attribute": None,
        },
    }


def _test_generic(objcls, permutations, op):
    """Tests expressions that can only be resolved to JSON."""
    for a, _ in permutations:
        A = objcls(a)
        C = objcls.symbolic()
        # test variable -> builtin against variable -> variable
        assert (
            C.__getattribute__(op)(a).to_dict()
            == C.__getattribute__(op)(A).to_dict()
        )
        # test commutative propery (this will fail)
        with pytest.raises(AssertionError):
            try:
                # function does not exist in left-operand
                a.__getattribute__(op)(C)
                # function exists, but is not commutative
                if type(a.__getattribute__(op)(A)) not in {objcls, type(a)}:
                    raise AssertionError("NotImplementedType")
            except AttributeError as e:
                raise AssertionError(e)
        # test instance dictionary generation
        _test_to_dict(objcls, a)
        # test functional dictionary generation
        expr = C.__getattribute__(op)(a)
        expr_dict = expr.to_dict()
        if issubclass(type(expr), AppendableFunction):
            assert len(expr_dict) == 2
            assert expr_dict["op"] == get_function_name(op)
            assert expr_dict["args"] == [
                C.to_dict(),
                A.to_dict(),
            ]
        elif issubclass(type(expr), TwoArgumentFunction):
            assert len(expr_dict) == 3
            assert expr_dict["op"] == get_function_name(op)
            assert expr_dict["lhs"] == C.to_dict()
            assert expr_dict["rhs"] == A.to_dict()
        else:
            raise AssertionError


def _test_unsupported(objcls, permutations, op):
    for a, b in permutations:
        with pytest.raises(AttributeError):
            objcls(a).__getattribute__(op)(b)


def test_list():
    # interoperable with built-in 'list'

    assert isinstance(List[Float], type)
    assert issubclass(List[Float], Variable)

    # test creating symbolic lists
    symbol = List[Float].symbolic()
    assert symbol.__str__() == "list[float]"
    assert symbol.to_dict() == {
        "type": "symbol",
        "value": {
            "owner": None,
            "name": "list[float]",
            "key": None,
            "attribute": None,
        },
    }

    # test creating valued lists
    variable = List[Float].definite([0.1, 0.2, 0.3])
    assert variable.__str__() == "[0.1, 0.2, 0.3]"
    assert variable.to_dict() == {
        "type": "list[float]",
        "value": [0.1, 0.2, 0.3],
    }

    # test setting value in list by index
    assert variable[1].get_value() == 0.2
    variable[1] = 3.14
    assert variable[1].get_value() == 3.14
    variable[1] = Float(0.2)

    # test nested typing
    assert variable[0].get_value() == 0.1

    # test comparison symbol -> value
    assert (symbol == [0.1, 0.2, 0.3]).to_dict() == {
        "op": "eq",
        "lhs": {
            "type": "symbol",
            "value": {
                "owner": None,
                "name": "list[float]",
                "key": None,
                "attribute": None,
            },
        },
        "rhs": {"type": "list[float]", "value": [0.1, 0.2, 0.3]},
    }

    # test comparison symbol -> valued variable
    assert (symbol == variable).to_dict() == {
        "op": "eq",
        "lhs": {
            "type": "symbol",
            "value": {
                "owner": None,
                "name": "list[float]",
                "key": None,
                "attribute": None,
            },
        },
        "rhs": {"type": "list[float]", "value": [0.1, 0.2, 0.3]},
    }

    # test comparison between valued variable and value
    assert (variable == [0.1, 0.2, 0.3]).to_dict() == {
        "type": "bool",
        "value": True,
    }


def test_dictionary_value():
    # test symbol cannot already have key or attribute
    with pytest.raises(ValueError) as e:
        DictionaryValue(
            symbol=Symbol(name="a", key="b", attribute="c", owner="d"),
            key="k",
        )
    assert "key" in str(e)
    with pytest.raises(ValueError) as e:
        DictionaryValue(
            symbol=Symbol(name="a", attribute="c", owner="d"),
            key="k",
        )
    assert "attribute" in str(e)


def test_dictionary():
    # interoperable with built-in 'dict'
    x = {
        "k0": True,
        "k1": "v1",
        "k2": 123,
        "k3": 1.24,
        "k4": datetime.datetime(year=2024, month=1, day=1),
        "k5": datetime.date(year=2024, month=1, day=1),
        "k6": datetime.time(hour=1),
        "k7": datetime.timedelta(seconds=100),
        "k8": Point((1, -1)),
        "k9": MultiPoint([(0, 0), (1, 1)]),
        "k10": LineString([(0, 0), (1, 1)]),
        "k11": MultiLineString([[(0, 0), (1, 1)]]),
        "k12": Polygon([[(0, 0), (1, 1), (0, 1), (0, 0)]]),
        "k13": MultiPolygon([[[(0, 0), (1, 1), (0, 1), (0, 0)]]]),
    }
    y = {
        "k0": False,
        "k1": "v2",
        "k2": 321,
        "k3": 1.24,
    }

    objcls = Dictionary
    permutations = [
        (x, x),
        (x, y),
        (y, y),
        (y, x),
    ]

    # test supported methods
    for op in ["__eq__", "__ne__"]:
        _test_generic(objcls, permutations, op)

    # test unsupported methods
    for op in [
        "__gt__",
        "__ge__",
        "__lt__",
        "__le__",
        "__and__",
        "__or__",
        "__xor__",
        "is_none",
        "is_not_none",
        "intersects",
        "inside",
        "outside",
    ]:
        _test_unsupported(objcls, permutations, op)

    # test encoding
    assert {
        "k0": {"type": "bool", "value": True},
        "k1": {"type": "string", "value": "v1"},
        "k2": {"type": "integer", "value": 123},
        "k3": {"type": "float", "value": 1.24},
        "k4": {"type": "datetime", "value": "2024-01-01T00:00:00"},
        "k5": {"type": "date", "value": "2024-01-01"},
        "k6": {"type": "time", "value": "01:00:00"},
        "k7": {"type": "duration", "value": 100.0},
        "k8": {"type": "point", "value": (1, -1)},
        "k9": {"type": "multipoint", "value": [(0, 0), (1, 1)]},
        "k10": {"type": "linestring", "value": [(0, 0), (1, 1)]},
        "k11": {"type": "multilinestring", "value": [[(0, 0), (1, 1)]]},
        "k12": {
            "type": "polygon",
            "value": [[(0, 0), (1, 1), (0, 1), (0, 0)]],
        },
        "k13": {
            "type": "multipolygon",
            "value": [[[(0, 0), (1, 1), (0, 1), (0, 0)]]],
        },
    } == Dictionary(x).encode_value()
