import datetime

import pytest

from valor.schemas import (
    Bool,
    Date,
    DateTime,
    Duration,
    Float,
    Integer,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    String,
    Symbol,
    Time,
    TypedList,
    Variable,
)
from valor.schemas.symbolic.operators import (
    AppendableFunction,
    Contains,
    TwoArgumentFunction,
)
from valor.schemas.symbolic.types import (
    Dictionary,
    DictionaryValue,
    _get_type_by_value,
    get_type_by_name,
)


def test__get_type_by_value():
    assert _get_type_by_value(True) is Bool
    assert _get_type_by_value("hello world") is String
    assert _get_type_by_value(int(1)) is Integer
    assert _get_type_by_value(float(3.14)) is Float
    assert (
        _get_type_by_value(datetime.datetime(year=2024, month=1, day=1))
        is DateTime
    )
    assert _get_type_by_value(datetime.date(year=2024, month=1, day=1)) is Date
    assert (
        _get_type_by_value(datetime.time(hour=1, minute=1, second=1)) is Time
    )
    assert _get_type_by_value(datetime.timedelta(seconds=100)) is Duration
    assert _get_type_by_value((1, 1)) is Point
    assert _get_type_by_value([(1, 1)]) is MultiPoint
    assert _get_type_by_value([(1, 1), (2, 2)]) is LineString
    assert _get_type_by_value([[(1, 1), (2, 2)]]) is MultiLineString
    assert _get_type_by_value([[(1, 1), (2, 2), (0, 1), (1, 1)]]) is Polygon
    assert (
        _get_type_by_value([[[(1, 1), (2, 2), (0, 1), (1, 1)]]])
        is MultiPolygon
    )
    assert _get_type_by_value({"randomvalue": "idk"}) is Dictionary
    with pytest.raises(NotImplementedError):
        assert _get_type_by_value(set()).__name__


def test_get_type_by_name():
    types_ = [
        Bool,
        String,
        Integer,
        Float,
        DateTime,
        Date,
        Time,
        Duration,
        Point,
        MultiPoint,
        LineString,
        MultiLineString,
        Polygon,
        MultiPolygon,
    ]
    for type_ in types_:
        type_name = type_.__name__
        assert issubclass(type_, Variable)
        assert isinstance(type_name, str)
        assert get_type_by_name(type_name) is type_
        assert get_type_by_name(f"list[{type_name}]") is TypedList[type_]
    with pytest.raises(NotImplementedError):
        assert get_type_by_name("some_nonexistent_type")


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
    # test value
    assert objcls(value).to_dict() == {
        "type": objcls.__name__.lower(),
        "value": objcls(value).encode_value(),
    }
    # test symbolic
    assert objcls.symbolic().to_dict() == {
        "type": "symbol",
        "value": {
            "name": objcls.__name__.lower(),
            "key": None,
            "attribute": None,
            "dtype": objcls.__name__.lower(),
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

    assert isinstance(TypedList[Float], type)
    assert issubclass(TypedList[Float], Variable)

    # test creating symbolic lists
    symbol = TypedList[Float].symbolic()
    assert (
        symbol.__str__() == "Symbol(name='list[float]', dtype='list[float]')"
    )
    assert symbol.to_dict() == {
        "type": "symbol",
        "value": {
            "name": "list[float]",
            "key": None,
            "attribute": None,
            "dtype": "list[float]",
        },
    }

    # test creating valued lists
    variable = TypedList[Float]([0.1, 0.2, 0.3])
    assert variable.__str__() == "[Float(0.1), Float(0.2), Float(0.3)]"
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
                "name": "list[float]",
                "key": None,
                "attribute": None,
                "dtype": "list[float]",
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
                "name": "list[float]",
                "key": None,
                "attribute": None,
                "dtype": "list[float]",
            },
        },
        "rhs": {"type": "list[float]", "value": [0.1, 0.2, 0.3]},
    }

    # test decode from json dict
    assert TypedList[Float].decode_value([0.1, 0.2, 0.3]).get_value() == [  # type: ignore
        0.1,
        0.2,
        0.3,
    ]

    # test comparison between valued variable and value
    assert (variable == [0.1, 0.2, 0.3]).to_dict() == {
        "type": "bool",
        "value": True,
    }

    # test setting list to non-list type
    with pytest.raises(TypeError):
        assert TypedList[String](String("hello"))

    # test setting list item to unsupported type
    with pytest.raises(TypeError):
        assert TypedList[Integer]([String("hello")])

    # test that untyped wrapper is not implemented
    with pytest.raises(TypeError):
        TypedList()  # type: ignore - intentionally missing args

    # test 'contains' operator
    op = TypedList[Float].symbolic().contains(Float(1.234), 3.14)
    assert isinstance(op, Contains)
    assert op.to_dict() == {
        "op": "contains",
        "lhs": {
            "type": "symbol",
            "value": {
                "name": "list[float]",
                "key": None,
                "attribute": None,
                "dtype": "list[float]",
            },
        },
        "rhs": {
            "type": "list[float]",
            "value": [
                1.234,
                3.14,
            ],
        },
    }


def test_dictionary_value():
    # test cannot hold a value
    with pytest.raises(ValueError):
        DictionaryValue(1)  # type: ignore - intentionally incorrect

    # test symbol must have key
    with pytest.raises(ValueError) as e:
        DictionaryValue(
            symbol=Symbol(name="a"),
        )
    assert "key" in str(e)

    # test router
    assert (DictionaryValue.symbolic(name="a", key="b") == 0).to_dict()[
        "op"
    ] == "eq"
    assert (DictionaryValue.symbolic(name="a", key="b") != 0).to_dict()[
        "op"
    ] == "ne"
    assert (DictionaryValue.symbolic(name="a", key="b") >= 0).to_dict()[
        "op"
    ] == "ge"
    assert (DictionaryValue.symbolic(name="a", key="b") <= 0).to_dict()[
        "op"
    ] == "le"
    assert (DictionaryValue.symbolic(name="a", key="b") > 0).to_dict()[
        "op"
    ] == "gt"
    assert (DictionaryValue.symbolic(name="a", key="b") < 0).to_dict()[
        "op"
    ] == "lt"
    assert (
        DictionaryValue.symbolic(name="a", key="b").intersects((0, 0))
    ).to_dict()["op"] == "intersects"
    assert (
        DictionaryValue.symbolic(name="a", key="b").inside((0, 0))
    ).to_dict()["op"] == "inside"
    assert (
        DictionaryValue.symbolic(name="a", key="b").outside((0, 0))
    ).to_dict()["op"] == "outside"
    assert (DictionaryValue.symbolic(name="a", key="b").is_none()).to_dict()[
        "op"
    ] == "isnull"
    assert (
        DictionaryValue.symbolic(name="a", key="b").is_not_none()
    ).to_dict()["op"] == "isnotnull"
    assert (DictionaryValue.symbolic(name="a", key="b").area == 0).to_dict()[
        "op"
    ] == "eq"

    # test router with Variable type
    assert (DictionaryValue.symbolic(name="a", key="b") == Float(0)).to_dict()[
        "op"
    ] == "eq"


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
        "intersects",
        "inside",
        "outside",
    ]:
        _test_unsupported(objcls, permutations, op)

    # test nullable
    v1 = objcls.nullable(None)
    assert v1.get_value() is None
    assert v1.is_none().get_value() is True  # type: ignore - always a bool
    assert v1.is_not_none().get_value() is False  # type: ignore - always a bool
    v2 = objcls.nullable(permutations[0][0])
    assert v2.get_value() is not None
    assert v2.is_none().get_value() is False  # type: ignore - always a bool
    assert v2.is_not_none().get_value() is True  # type: ignore - always a bool

    # test encoding
    assert {
        "k0": True,
        "k1": "v1",
        "k2": 123,
        "k3": 1.24,
        "k4": {"type": "datetime", "value": "2024-01-01T00:00:00"},
        "k5": {"type": "date", "value": "2024-01-01"},
        "k6": {"type": "time", "value": "01:00:00"},
        "k7": {"type": "duration", "value": 100.0},
        "k8": {
            "type": "geojson",
            "value": {"type": "Point", "coordinates": (1, -1)},
        },
        "k9": {
            "type": "geojson",
            "value": {"type": "MultiPoint", "coordinates": [(0, 0), (1, 1)]},
        },
        "k10": {
            "type": "geojson",
            "value": {"type": "LineString", "coordinates": [(0, 0), (1, 1)]},
        },
        "k11": {
            "type": "geojson",
            "value": {
                "type": "MultiLineString",
                "coordinates": [[(0, 0), (1, 1)]],
            },
        },
        "k12": {
            "type": "geojson",
            "value": {
                "type": "Polygon",
                "coordinates": [[(0, 0), (1, 1), (0, 1), (0, 0)]],
            },
        },
        "k13": {
            "type": "geojson",
            "value": {
                "type": "MultiPolygon",
                "coordinates": [[[(0, 0), (1, 1), (0, 1), (0, 0)]]],
            },
        },
    } == Dictionary(x).encode_value()
