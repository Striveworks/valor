import datetime
import typing

import pytest

from valor.schemas.symbolic.operators import (
    Condition,
    Eq,
    Function,
    IsNotNull,
    IsNull,
    Ne,
)
from valor.schemas.symbolic.types import (
    Boolean,
    Date,
    DateTime,
    Duration,
    Equatable,
    Float,
    Integer,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    Quantifiable,
    Spatial,
    String,
    Symbol,
    Time,
    Variable,
)


def test_symbol():
    s = Symbol(name="some_symbol")
    assert s.__repr__() == "Symbol(name='some_symbol')"
    assert s.__str__() == "some_symbol"
    assert s.to_dict() == {
        "name": "some_symbol",
        "key": None,
    }

    s = Symbol(
        name="some_name",
        key="some_key",
    )
    assert s.__repr__() == "Symbol(name='some_name', key='some_key')"
    assert s.__str__() == "some_name['some_key']"
    assert s.to_dict() == {
        "name": "some_name",
        "key": "some_key",
    }

    # test '__eq__'
    assert s == Symbol(
        name="some_name",
        key="some_key",
    )
    assert not (s == "symbol")

    # test '__ne__'
    assert not (
        s
        != Symbol(
            name="some_name",
            key="some_key",
        )
    )
    assert s != "symbol"


def _test_symbolic_outputs(v, s=Symbol(name="test")):
    assert s.to_dict() == v.to_dict()
    assert s.to_dict() == v.get_symbol().to_dict()
    assert f"Variable({s.__repr__()})" == v.__repr__()
    assert s.__str__() == v.__str__()
    assert v.is_symbolic and not v.is_value

    with pytest.raises(TypeError):
        v.get_value()


def test_variable():
    # test symbolic variables

    var_method1 = Variable.symbolic(name="test")
    var_method2 = Variable.preprocess(value=Symbol(name="test"))
    _test_symbolic_outputs(var_method1)
    _test_symbolic_outputs(var_method2)

    # test is_none
    is_none = Variable.symbolic().is_none()
    assert isinstance(is_none, IsNull)
    assert is_none.to_dict() == {
        "op": "isnull",
        "lhs": {
            "name": "variable",
            "key": None,
        },
        "rhs": None,
    }
    assert Variable.symbolic().get_symbol() == Symbol(name="variable")
    assert Variable(None).is_none() is True
    assert Variable(1234).is_none() is False
    with pytest.raises(TypeError):
        Variable(1234).get_symbol()

    # test is_not_none
    is_not_none = Variable.symbolic().is_not_none()
    assert isinstance(is_not_none, IsNotNull)
    assert is_not_none.to_dict() == {
        "op": "isnotnull",
        "lhs": {
            "name": "variable",
            "key": None,
        },
        "rhs": None,
    }
    assert Variable(None).is_not_none() is False
    assert Variable(1234).is_not_none() is True


def _test_equatable(varA, varB, varC):

    # equal
    assert (varA == varB).to_dict() == {
        "op": "eq",
        "lhs": {
            "name": "a",
            "key": None,
        },
        "rhs": {
            "name": "b",
            "key": None,
        },
    }
    assert (varA == varB).to_dict() == (varA == Symbol("B")).to_dict()

    # not equal
    assert (varA != varB).to_dict() == {
        "op": "not",
        "args": {
            "op": "eq",
            "lhs": {
                "name": "a",
                "key": None,
            },
            "rhs": {
                "name": "b",
                "key": None,
            },
        },
    }
    assert (varA != varB).to_dict() == (varA != Symbol("B")).to_dict()

    # in (exists within list)
    assert varA.in_([varB, varC]).to_dict() == {
        "op": "or",
        "args": [
            {
                "op": "eq",
                "lhs": {
                    "name": "a",
                    "key": None,
                },
                "rhs": {
                    "name": "b",
                    "key": None,
                },
            },
            {
                "op": "eq",
                "lhs": {
                    "name": "a",
                    "key": None,
                },
                "rhs": {
                    "name": "c",
                    "key": None,
                },
            },
        ],
    }
    assert (
        varA.in_([varB, varC]).to_dict()
        == varA.in_([Symbol("B"), Symbol("C")]).to_dict()
    )

    # hashable
    assert {varA, varB} == {varB, varA}


def _test_quantifiable(varA, varB, varC):

    # greater-than
    assert (varA > varB).to_dict() == {
        "lhs": {
            "key": None,
            "name": "a",
        },
        "op": "gt",
        "rhs": {
            "name": "b",
            "key": None,
        },
    }

    # greater-than or equal
    assert (varA >= varB).to_dict() == {
        "lhs": {
            "key": None,
            "name": "a",
        },
        "op": "gte",
        "rhs": {
            "key": None,
            "name": "b",
        },
    }

    # less-than
    assert (varA < varB).to_dict() == {
        "lhs": {
            "key": None,
            "name": "a",
        },
        "op": "lt",
        "rhs": {
            "key": None,
            "name": "b",
        },
    }

    # less-than or equal
    assert (varA <= varB).to_dict() == {
        "lhs": {
            "key": None,
            "name": "a",
        },
        "op": "lte",
        "rhs": {
            "key": None,
            "name": "b",
        },
    }


def _test_nullable(varA, varB, varC):
    # is none
    assert varA.is_none().to_dict() == {
        "lhs": {
            "key": None,
            "name": "a",
        },
        "rhs": None,
        "op": "isnull",
    }

    # is not none
    assert varA.is_not_none().to_dict() == {
        "lhs": {
            "key": None,
            "name": "a",
        },
        "rhs": None,
        "op": "isnotnull",
    }


def _test_spatial(varA, varB, varC):
    # intersects
    assert varA.intersects(varB).to_dict() == {
        "lhs": {
            "key": None,
            "name": "a",
        },
        "op": "intersects",
        "rhs": {
            "key": None,
            "name": "b",
        },
    }

    # inside
    assert varA.inside(varB).to_dict() == {
        "lhs": {
            "key": None,
            "name": "a",
        },
        "op": "inside",
        "rhs": {
            "key": None,
            "name": "b",
        },
    }

    # outside
    assert varA.outside(varB).to_dict() == {
        "lhs": {
            "key": None,
            "name": "a",
        },
        "op": "outside",
        "rhs": {
            "key": None,
            "name": "b",
        },
    }


def test_modifiers():

    # equatable
    A = Equatable.symbolic("A")
    B = Equatable.symbolic("B")
    C = Equatable.symbolic("C")
    _test_equatable(A, B, C)
    _test_nullable(A, B, C)
    with pytest.raises(AttributeError):
        _test_quantifiable(A, B, C)
    with pytest.raises(AttributeError):
        _test_spatial(A, B, C)

    # quantifiable
    A = Quantifiable.symbolic("A")
    B = Quantifiable.symbolic("B")
    C = Quantifiable.symbolic("C")
    _test_equatable(A, B, C)
    _test_quantifiable(A, B, C)
    _test_nullable(A, B, C)
    with pytest.raises(AttributeError):
        _test_spatial(A, B, C)

    # spatial
    A = Spatial.symbolic("A")
    B = Spatial.symbolic("B")
    C = Spatial.symbolic("C")
    _test_spatial(A, B, C)
    _test_nullable(A, B, C)
    with pytest.raises(AttributeError):
        _test_equatable(A, B, C)
    with pytest.raises(AttributeError):
        _test_quantifiable(A, B, C)


def get_function_name(fn: str) -> str:
    fns = {
        "__eq__": "eq",
        "__ne__": "ne",
        "__and__": "and",
        "__or__": "or",
        "__xor__": "xor",
        "__gt__": "gt",
        "__ge__": "gte",
        "__lt__": "lt",
        "__le__": "lte",
        "is_none": "isnull",
        "is_not_none": "isnotnull",
        "intersects": "intersects",
        "inside": "inside",
        "outside": "outside",
    }
    return fns[fn]


def _test_encoding(objcls, value, encoded_value):
    assert (
        objcls(value).to_dict() == objcls.decode_value(encoded_value).to_dict()
    )
    assert encoded_value == objcls(value).encode_value()
    assert objcls.decode_value(None) is None
    assert objcls.nullable(None).encode_value() is None


def _test_to_dict(objcls, value, type_name: typing.Optional[str] = None):
    type_name = type_name if type_name else objcls.__name__.lower()
    # test __init__
    assert objcls(value).to_dict() == {
        "type": type_name,
        "value": objcls(value).encode_value(),
    }
    # test valued
    assert objcls(value).to_dict() == {
        "type": type_name,
        "value": objcls(value).encode_value(),
    }
    # test symbolic
    assert objcls.symbolic().to_dict() == {
        "name": type_name,
        "key": None,
    }


def _test_generic(
    objcls, permutations, op, type_name: typing.Optional[str] = None
):
    """Tests expressions that can only be resolved to JSON."""
    type_name = type_name if type_name else objcls.__name__.lower()
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
        _test_to_dict(objcls, a, type_name=type_name)
        # test functional dictionary generation
        expr = C.__getattribute__(op)(a)
        expr_dict = expr.to_dict()
        if isinstance(expr, Ne):
            # this is an edge case as the Ne operator is currently set to Not(Equal(A, B))
            assert len(expr_dict) == 2
            assert expr_dict["op"] == "not"
            assert expr_dict["args"] == Eq(C, A).to_dict()
        elif issubclass(type(expr), Function):
            assert len(expr_dict) == 2
            assert expr_dict["op"] == get_function_name(op)
            assert expr_dict["args"] == [
                C.to_dict(),
                A.to_dict(),
            ]
        elif issubclass(type(expr), Condition):
            assert len(expr_dict) == 3
            assert expr_dict["op"] == get_function_name(op)
            assert expr_dict["lhs"] == C.to_dict()
            assert expr_dict["rhs"] == A.to_dict()
        else:
            raise AssertionError


def _test_resolvable(
    objcls, permutations, op, type_name: typing.Optional[str] = None
):
    """Test expressions that can be simplified to 'Boolean'"""
    type_name = type_name if type_name else objcls.__name__.lower()
    for a, b in permutations:
        A = objcls(a)
        B = objcls(b)
        # test variable -> builtin against truth
        assert A.__getattribute__(op)(b) is a.__getattribute__(op)(b)
        # test variable -> variable against truth
        assert A.__getattribute__(op)(B) is a.__getattribute__(op)(b)
        # test dictionary generation
        dictA = A.to_dict()
        assert A.get_value() == a
        assert len(dictA) == 2
        assert dictA["type"] == type_name
        assert dictA["value"] == A.encode_value()
    # test expressions that cannot be simplified
    _test_generic(objcls, permutations, op, type_name=type_name)


def _test_unsupported(objcls, permutations, op):
    for a, b in permutations:
        with pytest.raises(AttributeError):
            objcls(a).__getattribute__(op)(b)


def test_bool():
    # interoperable with builtin 'bool'
    objcls = Boolean
    permutations = [
        (True, True),
        (True, False),
        (False, False),
        (False, True),
    ]

    # test supported methods
    for op in ["__eq__", "__ne__", "__and__", "__or__"]:
        _test_resolvable(objcls, permutations, op)
    assert (~Boolean(True)) is False
    assert (~Boolean(False)) is True

    # test unsupported methods
    for op in [
        "__gt__",
        "__ge__",
        "__lt__",
        "__le__",
        "intersects",
        "inside",
        "outside",
    ]:
        _test_unsupported(objcls, permutations, op)

    # test nullable
    v1 = objcls.nullable(None)
    assert v1.get_value() is None
    assert v1.is_none() is True
    assert v1.is_not_none() is False
    v2 = objcls.nullable(permutations[0][0])
    assert v2.get_value() is not None
    assert v2.is_none() is False
    assert v2.is_not_none() is True

    # test encoding
    _test_encoding(objcls, True, True)
    _test_encoding(objcls, False, False)

    # test and operation
    assert (Boolean(True) & Boolean(True)) is True
    assert (Boolean(True) & Boolean(False)) is False
    assert (Boolean(False) & Boolean(True)) is False
    assert (Boolean(False) & Boolean(False)) is False

    # test or operation
    assert (Boolean(True) | Boolean(True)) is True
    assert (Boolean(True) | Boolean(False)) is True
    assert (Boolean(False) | Boolean(True)) is True
    assert (Boolean(False) | Boolean(False)) is False

    # test negation operation
    assert (~Boolean(True)) is False
    assert (~Boolean(False)) is True
    assert (~Boolean.symbolic()).to_dict() == {  # type: ignore
        "op": "not",
        "args": {
            "name": "boolean",
            "key": None,
        },
    }


def test_integer():
    # interoperable with builtin 'int'
    objcls = Integer
    permutations = [
        (100, 100),
        (100, -100),
        (-100, -100),
        (-100, 100),
    ]

    # test supported methods
    for op in ["__eq__", "__ne__", "__gt__", "__ge__", "__lt__", "__le__"]:
        _test_resolvable(objcls, permutations, op)

    # test unsupported methods
    for op in [
        "__and__",
        "__or__",
        "intersects",
        "inside",
        "outside",
    ]:
        _test_unsupported(objcls, permutations, op)

    # test equatable
    assert (Integer.nullable(None) == Integer(1)) is False
    assert (Integer(1) == Integer.nullable(None)) is False
    assert (Integer.nullable(None) != Integer(1)) is True
    assert (Integer(1) != Integer.nullable(None)) is True

    # test nullable
    v1 = objcls.nullable(None)
    assert v1.get_value() is None
    assert v1.is_none() is True
    assert v1.is_not_none() is False
    v2 = objcls.nullable(permutations[0][0])
    assert v2.get_value() is not None
    assert v2.is_none() is False
    assert v2.is_not_none() is True

    # test encoding
    _test_encoding(objcls, 10, 10)


def test_float():
    # interoperable with builtin 'float'
    objcls = Float
    permutations = [
        (1.23, 3.21),
        (1.23, -3.21),
        (-1.23, -3.21),
        (-1.23, 3.21),
    ]

    # test supported methods
    for op in ["__eq__", "__ne__", "__gt__", "__ge__", "__lt__", "__le__"]:
        _test_resolvable(objcls, permutations, op)

    # test unsupported methods
    for op in [
        "__and__",
        "__or__",
        "intersects",
        "inside",
        "outside",
    ]:
        _test_unsupported(objcls, permutations, op)

    # test nullable
    v1 = objcls.nullable(None)
    assert v1.get_value() is None
    assert v1.is_none() is True
    assert v1.is_not_none() is False
    v2 = objcls.nullable(permutations[0][0])
    assert v2.get_value() is not None
    assert v2.is_none() is False
    assert v2.is_not_none() is True

    # test encoding
    _test_encoding(objcls, 1.23, 1.23)


def test_string():
    # interoperable with builtin 'str'
    objcls = String
    permutations = [
        ("hello", "hello"),
        ("hello", "world"),
        ("world", "hello"),
        ("world", "world"),
    ]

    # test supported methods
    for op in ["__eq__", "__ne__"]:
        _test_resolvable(objcls, permutations, op)

    # test unsupported methods
    for op in [
        "__gt__",
        "__ge__",
        "__lt__",
        "__le__",
        "__and__",
        "__or__",
        "intersects",
        "inside",
        "outside",
    ]:
        _test_unsupported(objcls, permutations, op)

    # test nullable
    v1 = objcls.nullable(None)
    assert v1.get_value() is None
    assert v1.is_none() is True
    assert v1.is_not_none() is False
    v2 = objcls.nullable(permutations[0][0])
    assert v2.get_value() is not None
    assert v2.is_none() is False
    assert v2.is_not_none() is True

    # test encoding
    _test_encoding(objcls, "hello", "hello")


def test_datetime():
    # interoperable with 'datetime.datetime'
    objcls = DateTime
    permutations = [
        (
            datetime.datetime(year=2024, month=1, day=1),
            datetime.datetime(year=2024, month=1, day=1),
        ),
        (
            datetime.datetime(year=2024, month=1, day=1),
            datetime.datetime(year=2024, month=1, day=2),
        ),
        (
            datetime.datetime(year=2024, month=1, day=2),
            datetime.datetime(year=2024, month=1, day=2),
        ),
        (
            datetime.datetime(year=2024, month=1, day=2),
            datetime.datetime(year=2024, month=1, day=1),
        ),
    ]

    # test supported methods
    for op in ["__eq__", "__ne__", "__gt__", "__ge__", "__lt__", "__le__"]:
        _test_resolvable(objcls, permutations, op)

    # test unsupported methods
    for op in [
        "__and__",
        "__or__",
        "intersects",
        "inside",
        "outside",
    ]:
        _test_unsupported(objcls, permutations, op)

    # test nullable
    v1 = objcls.nullable(None)
    assert v1.get_value() is None
    assert v1.is_none() is True
    assert v1.is_not_none() is False
    v2 = objcls.nullable(permutations[0][0])
    assert v2.get_value() is not None
    assert v2.is_none() is False
    assert v2.is_not_none() is True

    # test encoding
    _test_encoding(
        objcls,
        datetime.datetime(year=2024, month=1, day=1),
        "2024-01-01T00:00:00",
    )


def test_date():
    # interoperable with 'datetime.date'
    objcls = Date
    permutations = [
        (
            datetime.date(year=2024, month=1, day=1),
            datetime.date(year=2024, month=1, day=1),
        ),
        (
            datetime.date(year=2024, month=1, day=1),
            datetime.date(year=2024, month=1, day=2),
        ),
        (
            datetime.date(year=2024, month=1, day=2),
            datetime.date(year=2024, month=1, day=2),
        ),
        (
            datetime.date(year=2024, month=1, day=2),
            datetime.date(year=2024, month=1, day=1),
        ),
    ]

    # test supported methods
    for op in ["__eq__", "__ne__", "__gt__", "__ge__", "__lt__", "__le__"]:
        _test_resolvable(objcls, permutations, op)

    # test unsupported methods
    for op in [
        "__and__",
        "__or__",
        "intersects",
        "inside",
        "outside",
    ]:
        _test_unsupported(objcls, permutations, op)

    # test nullable
    v1 = objcls.nullable(None)
    assert v1.get_value() is None
    assert v1.is_none() is True
    assert v1.is_not_none() is False
    v2 = objcls.nullable(permutations[0][0])
    assert v2.get_value() is not None
    assert v2.is_none() is False
    assert v2.is_not_none() is True

    # test encoding
    _test_encoding(
        objcls, datetime.date(year=2024, month=1, day=1), "2024-01-01"
    )


def test_time():
    # interoperable with 'datetime.time'
    objcls = Time
    permutations = [
        (datetime.time(hour=1), datetime.time(hour=1)),
        (datetime.time(hour=1), datetime.time(hour=2)),
        (datetime.time(hour=2), datetime.time(hour=2)),
        (datetime.time(hour=2), datetime.time(hour=1)),
    ]

    # test supported methods
    for op in ["__eq__", "__ne__", "__gt__", "__ge__", "__lt__", "__le__"]:
        _test_resolvable(objcls, permutations, op)

    # test unsupported methods
    for op in [
        "__and__",
        "__or__",
        "intersects",
        "inside",
        "outside",
    ]:
        _test_unsupported(objcls, permutations, op)

    # test nullable
    v1 = objcls.nullable(None)
    assert v1.get_value() is None
    assert v1.is_none() is True
    assert v1.is_not_none() is False
    v2 = objcls.nullable(permutations[0][0])
    assert v2.get_value() is not None
    assert v2.is_none() is False
    assert v2.is_not_none() is True

    # test encoding
    _test_encoding(objcls, datetime.time(hour=1), "01:00:00")


def test_duration():
    # interoperable with 'datetime.timedelta'
    objcls = Duration
    permutations = [
        (datetime.timedelta(seconds=1), datetime.timedelta(seconds=1)),
        (datetime.timedelta(seconds=1), datetime.timedelta(seconds=2)),
        (datetime.timedelta(seconds=2), datetime.timedelta(seconds=2)),
        (datetime.timedelta(seconds=2), datetime.timedelta(seconds=1)),
    ]

    # test supported methods
    for op in ["__eq__", "__ne__", "__gt__", "__ge__", "__lt__", "__le__"]:
        _test_resolvable(objcls, permutations, op)

    # test unsupported methods
    for op in [
        "__and__",
        "__or__",
        "intersects",
        "inside",
        "outside",
    ]:
        _test_unsupported(objcls, permutations, op)

    # test nullable
    v1 = objcls.nullable(None)
    assert v1.get_value() is None
    assert v1.is_none() is True
    assert v1.is_not_none() is False
    v2 = objcls.nullable(permutations[0][0])
    assert v2 is not None
    assert v2.is_none() is False
    assert v2.is_not_none() is True

    # test encoding
    _test_encoding(objcls, datetime.timedelta(seconds=1), 1.0)


def test_point():
    # interoperable with GeoJSON-style 'point' geometry
    objcls = Point
    permutations = [((0, 0), (1, 1))]

    # test supported methods
    for op in ["intersects", "inside", "outside", "__eq__", "__ne__"]:
        _test_generic(objcls, permutations, op)

    # test unsupported methods
    for op in [
        "__gt__",
        "__ge__",
        "__lt__",
        "__le__",
        "__and__",
        "__or__",
    ]:
        _test_unsupported(objcls, permutations, op)

    # test nullable
    v1 = objcls.nullable(None)
    assert v1.get_value() is None
    assert v1.is_none() is True
    assert v1.is_not_none() is False
    v2 = objcls.nullable(permutations[0][0])
    assert v2.get_value() is not None
    assert v2.is_none() is False
    assert v2.is_not_none() is True

    # test encoding
    _test_encoding(objcls, (1, -1), (1, -1))


def test_multipoint():
    # interoperable with GeoJSON-style 'multipoint' geometry
    objcls = MultiPoint
    permutations = [([(0, 0), (1, 1)], [(1, 0), (0, 1)])]

    # test supported methods
    for op in ["intersects", "inside", "outside"]:
        _test_generic(objcls, permutations, op)

    # test unsupported methods
    for op in [
        "__eq__",
        "__ne__",
        "__gt__",
        "__ge__",
        "__lt__",
        "__le__",
        "__and__",
        "__or__",
    ]:
        _test_unsupported(objcls, permutations, op)

    # test nullable
    v1 = objcls.nullable(None)
    assert v1.get_value() is None
    assert v1.is_none() is True
    assert v1.is_not_none() is False
    v2 = objcls.nullable(permutations[0][0])
    assert v2.get_value() is not None
    assert v2.is_none() is False
    assert v2.is_not_none() is True

    # test encoding
    _test_encoding(objcls, [(0, 0), (1, 1)], [(0, 0), (1, 1)])


def test_linestring():
    # interoperable with GeoJSON-style 'linestring' geometry
    objcls = LineString
    permutations = [([(0, 0), (1, 1)], [(1, 0), (0, 1)])]

    # test supported methods
    for op in ["intersects", "inside", "outside"]:
        _test_generic(objcls, permutations, op)

    # test unsupported methods
    for op in [
        "__eq__",
        "__ne__",
        "__gt__",
        "__ge__",
        "__lt__",
        "__le__",
        "__and__",
        "__or__",
    ]:
        _test_unsupported(objcls, permutations, op)

    # test nullable
    v1 = objcls.nullable(None)
    assert v1.get_value() is None
    assert v1.is_none() is True
    assert v1.is_not_none() is False
    v2 = objcls.nullable(permutations[0][0])
    assert v2.get_value() is not None
    assert v2.is_none() is False
    assert v2.is_not_none() is True

    # test encoding
    _test_encoding(objcls, [(0, 0), (1, 1)], [(0, 0), (1, 1)])


def test_multilinestring():
    # interoperable with GeoJSON-style 'multilinestring' geometry
    objcls = MultiLineString
    permutations = [
        (
            [[(0, 0), (1, 1)], [(1, 0), (0, 1)]],
            [[(-1, 1), (1, 1)], [(1, 0), (0, 1)]],
        )
    ]

    # test supported methods
    for op in ["intersects", "inside", "outside"]:
        _test_generic(objcls, permutations, op)

    # test unsupported methods
    for op in [
        "__eq__",
        "__ne__",
        "__gt__",
        "__ge__",
        "__lt__",
        "__le__",
        "__and__",
        "__or__",
    ]:
        _test_unsupported(objcls, permutations, op)

    # test nullable
    v1 = objcls.nullable(None)
    assert v1.get_value() is None
    assert v1.is_none() is True
    assert v1.is_not_none() is False
    v2 = objcls.nullable(permutations[0][0])
    assert v2.get_value() is not None
    assert v2.is_none() is False
    assert v2.is_not_none() is True

    # test encoding
    _test_encoding(objcls, [[(0, 0), (1, 1)]], [[(0, 0), (1, 1)]])


def test_polygon():
    # interoperable with GeoJSON-style 'polygon' geometry
    objcls = Polygon
    permutations = [
        (
            [[(0, 0), (1, 1), (0, 1), (0, 0)]],  # regular polygon
            [
                [(0, 0), (1, 1), (0, 1), (0, 0)],
                [(0.1, 0.1), (0.9, 0.9), (0.1, 0.9), (0.1, 0.1)],
            ],  # polygon w/ hole
        )
    ]

    # test supported methods
    for op in ["intersects", "inside", "outside"]:
        _test_generic(objcls, permutations, op)

    # test unsupported methods
    for op in [
        "__eq__",
        "__ne__",
        "__gt__",
        "__ge__",
        "__lt__",
        "__le__",
        "__and__",
        "__or__",
    ]:
        _test_unsupported(objcls, permutations, op)

    # test nullable
    v1 = objcls.nullable(None)
    assert v1.get_value() is None
    assert v1.is_none() is True
    assert v1.is_not_none() is False
    v2 = objcls.nullable(permutations[0][0])
    assert v2.get_value() is not None
    assert v2.is_none() is False
    assert v2.is_not_none() is True

    # test encoding
    _test_encoding(
        objcls,
        [[(0, 0), (1, 1), (0, 1), (0, 0)]],
        [[(0, 0), (1, 1), (0, 1), (0, 0)]],
    )

    # test property 'area'
    assert objcls.symbolic().area.is_symbolic
    assert objcls.symbolic().area.to_dict() == {
        "name": f"{objcls.__name__.lower()}.area",
        "key": None,
    }
    # test that property 'area' is not accessible when object is a value
    with pytest.raises(ValueError):
        objcls(permutations[0][0]).area


def test_multipolygon():
    # interoperable with GeoJSON-style 'multipolygon' geometry
    objcls = MultiPolygon
    permutations = [
        (
            [
                [[(0, 0), (1, 1), (0, 1), (0, 0)]],
                [
                    [(0, 0), (1, 1), (0, 1), (0, 0)],
                    [(0.1, 0.1), (0.9, 0.9), (0.1, 0.9), (0.1, 0.1)],
                ],
            ],
            [
                [[(0, 0), (1, 1), (0, 1), (0, 0)]],
                [
                    [(0, 0), (1, 1), (0, 1), (0, 0)],
                    [(0.1, 0.1), (0.9, 0.9), (0.1, 0.9), (0.1, 0.1)],
                ],
            ],
        )
    ]

    # test supported methods
    for op in ["intersects", "inside", "outside"]:
        _test_generic(objcls, permutations, op)

    # test unsupported methods
    for op in [
        "__eq__",
        "__ne__",
        "__gt__",
        "__ge__",
        "__lt__",
        "__le__",
        "__and__",
        "__or__",
    ]:
        _test_unsupported(objcls, permutations, op)

    # test nullable
    v1 = objcls.nullable(None)
    assert v1.get_value() is None
    assert v1.is_none() is True
    assert v1.is_not_none() is False
    v2 = objcls.nullable(permutations[0][0])
    assert v2.get_value() is not None
    assert v2.is_none() is False
    assert v2.is_not_none() is True

    # test encoding
    _test_encoding(
        objcls,
        [[[(0, 0), (1, 1), (0, 1), (0, 0)]]],
        [[[(0, 0), (1, 1), (0, 1), (0, 0)]]],
    )

    # test property 'area'
    assert objcls.symbolic().area.is_symbolic
    assert objcls.symbolic().area.to_dict() == {
        "name": f"{objcls.__name__.lower()}.area",
        "key": None,
    }
    # test that property 'area' is not accessible when object is a value
    with pytest.raises(ValueError):
        objcls(permutations[0][0]).area

    # test `from_polygons` class method

    poly1_boundary = [(0, 0), (1, 1), (0, 1), (0, 0)]
    poly2_boundary = [(0, 10), (5, 5), (0, 5), (0, 10)]
    poly2_hole = [(0.1, 0.1), (0.9, 0.9), (0.1, 0.9), (0.1, 0.1)]
    polys = [Polygon([poly1_boundary]), Polygon([poly2_boundary, poly2_hole])]
    multi_poly = MultiPolygon.from_polygons(polys)
    assert multi_poly.get_value() == [
        [poly1_boundary],
        [poly2_boundary, poly2_hole],
    ]


def test_nullable():

    # test usage
    assert Float.nullable(0.6).get_value() == 0.6
    assert Float.nullable(0.6).to_dict() == {
        "type": "float",
        "value": 0.6,
    }
    assert Float.nullable(None).get_value() is None
    assert Float.nullable(None).to_dict() == {
        "type": "float",
        "value": None,
    }
