import datetime
import typing

import pytest

from valor.schemas.symbolic.operators import (
    AppendableFunction,
    TwoArgumentFunction,
)
from valor.schemas.symbolic.types import (
    Bool,
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
        "type": "symbol",
        "value": {
            "owner": None,
            "name": "some_symbol",
            "key": None,
            "attribute": None,
        },
    }

    s = Symbol(
        owner="some_owner",
        name="some_name",
        attribute="some_attribute",
        key="some_key",
    )
    assert (
        s.__repr__()
        == "Symbol(owner='some_owner', name='some_name', key='some_key', attribute='some_attribute')"
    )
    assert s.__str__() == "some_owner.some_name['some_key'].some_attribute"
    assert s.to_dict() == {
        "type": "symbol",
        "value": {
            "owner": "some_owner",
            "name": "some_name",
            "key": "some_key",
            "attribute": "some_attribute",
        },
    }

    # test '__eq__'
    assert s == Symbol(
        owner="some_owner",
        name="some_name",
        attribute="some_attribute",
        key="some_key",
    )
    assert not (s == "symbol")

    # test '__ne__'
    assert not (
        s
        != Symbol(
            owner="some_owner",
            name="some_name",
            attribute="some_attribute",
            key="some_key",
        )
    )
    assert s != "symbol"


def _test_symbolic_outputs(v, s=Symbol(name="test")):
    assert s.to_dict() == v.to_dict()
    assert s.to_dict() == v.get_symbol().to_dict()
    assert s.__repr__() == v.__repr__()
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
    assert Variable.symbolic().is_none().to_dict() == {
        "op": "isnull",
        "arg": {
            "type": "symbol",
            "value": {
                "name": "variable",
                "owner": None,
                "key": None,
                "attribute": None,
            },
        },
    }
    assert Variable.symbolic().get_symbol() == Symbol(name="variable")
    assert Variable(None).is_none().get_value() is True  # type: ignore - known output
    assert Variable(1234).is_none().get_value() is False  # type: ignore - known output
    with pytest.raises(TypeError):
        Variable(1234).get_symbol()

    # test is_not_none
    assert Variable.symbolic().is_not_none().to_dict() == {
        "op": "isnotnull",
        "arg": {
            "type": "symbol",
            "value": {
                "name": "variable",
                "owner": None,
                "key": None,
                "attribute": None,
            },
        },
    }
    assert Variable(None).is_not_none().get_value() is False  # type: ignore - known output
    assert Variable(1234).is_not_none().get_value() is True  # type: ignore - known output


def _test_equatable(varA, varB, varC):

    # equal
    assert (varA == varB).to_dict() == {
        "op": "eq",
        "lhs": {
            "type": "symbol",
            "value": {
                "owner": None,
                "name": "a",
                "key": None,
                "attribute": None,
            },
        },
        "rhs": {
            "type": "symbol",
            "value": {
                "owner": None,
                "name": "b",
                "key": None,
                "attribute": None,
            },
        },
    }
    assert (varA == varB).to_dict() == (varA == Symbol("B")).to_dict()

    # not equal
    assert (varA != varB).to_dict() == {
        "op": "ne",
        "lhs": {
            "type": "symbol",
            "value": {
                "owner": None,
                "name": "a",
                "key": None,
                "attribute": None,
            },
        },
        "rhs": {
            "type": "symbol",
            "value": {
                "owner": None,
                "name": "b",
                "key": None,
                "attribute": None,
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
                    "type": "symbol",
                    "value": {
                        "owner": None,
                        "name": "a",
                        "key": None,
                        "attribute": None,
                    },
                },
                "rhs": {
                    "type": "symbol",
                    "value": {
                        "owner": None,
                        "name": "b",
                        "key": None,
                        "attribute": None,
                    },
                },
            },
            {
                "op": "eq",
                "lhs": {
                    "type": "symbol",
                    "value": {
                        "owner": None,
                        "name": "a",
                        "key": None,
                        "attribute": None,
                    },
                },
                "rhs": {
                    "type": "symbol",
                    "value": {
                        "owner": None,
                        "name": "c",
                        "key": None,
                        "attribute": None,
                    },
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
            "type": "symbol",
            "value": {
                "attribute": None,
                "key": None,
                "name": "a",
                "owner": None,
            },
        },
        "op": "gt",
        "rhs": {
            "type": "symbol",
            "value": {
                "attribute": None,
                "key": None,
                "name": "b",
                "owner": None,
            },
        },
    }

    # greater-than or equal
    assert (varA >= varB).to_dict() == {
        "lhs": {
            "type": "symbol",
            "value": {
                "attribute": None,
                "key": None,
                "name": "a",
                "owner": None,
            },
        },
        "op": "ge",
        "rhs": {
            "type": "symbol",
            "value": {
                "attribute": None,
                "key": None,
                "name": "b",
                "owner": None,
            },
        },
    }

    # less-than
    assert (varA < varB).to_dict() == {
        "lhs": {
            "type": "symbol",
            "value": {
                "attribute": None,
                "key": None,
                "name": "a",
                "owner": None,
            },
        },
        "op": "lt",
        "rhs": {
            "type": "symbol",
            "value": {
                "attribute": None,
                "key": None,
                "name": "b",
                "owner": None,
            },
        },
    }

    # less-than or equal
    assert (varA <= varB).to_dict() == {
        "lhs": {
            "type": "symbol",
            "value": {
                "attribute": None,
                "key": None,
                "name": "a",
                "owner": None,
            },
        },
        "op": "le",
        "rhs": {
            "type": "symbol",
            "value": {
                "attribute": None,
                "key": None,
                "name": "b",
                "owner": None,
            },
        },
    }


def _test_nullable(varA, varB, varC):
    # is none
    assert varA.is_none().to_dict() == {
        "arg": {
            "type": "symbol",
            "value": {
                "attribute": None,
                "key": None,
                "name": "a",
                "owner": None,
            },
        },
        "op": "isnull",
    }

    # is not none
    assert varA.is_not_none().to_dict() == {
        "arg": {
            "type": "symbol",
            "value": {
                "attribute": None,
                "key": None,
                "name": "a",
                "owner": None,
            },
        },
        "op": "isnotnull",
    }


def _test_spatial(varA, varB, varC):
    # intersects
    assert varA.intersects(varB).to_dict() == {
        "lhs": {
            "type": "symbol",
            "value": {
                "attribute": None,
                "key": None,
                "name": "a",
                "owner": None,
            },
        },
        "op": "intersects",
        "rhs": {
            "type": "symbol",
            "value": {
                "attribute": None,
                "key": None,
                "name": "b",
                "owner": None,
            },
        },
    }

    # inside
    assert varA.inside(varB).to_dict() == {
        "lhs": {
            "type": "symbol",
            "value": {
                "attribute": None,
                "key": None,
                "name": "a",
                "owner": None,
            },
        },
        "op": "inside",
        "rhs": {
            "type": "symbol",
            "value": {
                "attribute": None,
                "key": None,
                "name": "b",
                "owner": None,
            },
        },
    }

    # outside
    assert varA.outside(varB).to_dict() == {
        "lhs": {
            "type": "symbol",
            "value": {
                "attribute": None,
                "key": None,
                "name": "a",
                "owner": None,
            },
        },
        "op": "outside",
        "rhs": {
            "type": "symbol",
            "value": {
                "attribute": None,
                "key": None,
                "name": "b",
                "owner": None,
            },
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
        "type": "symbol",
        "value": {
            "owner": None,
            "name": type_name,
            "key": None,
            "attribute": None,
        },
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


def _test_resolvable(
    objcls, permutations, op, type_name: typing.Optional[str] = None
):
    """Test expressions that can be simplified to 'Bool'"""
    type_name = type_name if type_name else objcls.__name__.lower()
    for a, b in permutations:
        A = objcls(a)
        B = objcls(b)
        # test variable -> builtin against truth
        assert A.__getattribute__(op)(b).get_value() is a.__getattribute__(op)(
            b
        )
        # test variable -> variable against truth
        assert A.__getattribute__(op)(B).get_value() is a.__getattribute__(op)(
            b
        )
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
    objcls = Bool
    permutations = [
        (True, True),
        (True, False),
        (False, False),
        (False, True),
    ]

    # test supported methods
    for op in ["__eq__", "__ne__", "__and__", "__or__", "__xor__"]:
        _test_resolvable(objcls, permutations, op)
    assert (~Bool(True)).get_value() is False  # type: ignore - this will always return a bool
    assert (~Bool(False)).get_value() is True  # type: ignore - this will always return a bool

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
    assert v1.is_none().get_value() is True  # type: ignore - always a bool
    assert v1.is_not_none().get_value() is False  # type: ignore - always a bool
    v2 = objcls.nullable(permutations[0][0])
    assert v2.get_value() is not None
    assert v2.is_none().get_value() is False  # type: ignore - always a bool
    assert v2.is_not_none().get_value() is True  # type: ignore - always a bool

    # test encoding
    _test_encoding(objcls, True, True)
    _test_encoding(objcls, False, False)

    # test and operation
    assert (Bool(True) & Bool(True)).get_value() is True  # type: ignore - known output
    assert (Bool(True) & Bool(False)).get_value() is False  # type: ignore - known output
    assert (Bool(False) & Bool(True)).get_value() is False  # type: ignore - known output
    assert (Bool(False) & Bool(False)).get_value() is False  # type: ignore - known output

    # test or operation
    assert (Bool(True) | Bool(True)).get_value() is True  # type: ignore - known output
    assert (Bool(True) | Bool(False)).get_value() is True  # type: ignore - known output
    assert (Bool(False) | Bool(True)).get_value() is True  # type: ignore - known output
    assert (Bool(False) | Bool(False)).get_value() is False  # type: ignore - known output

    # test xor operation
    assert (Bool(True) ^ Bool(True)).get_value() is False  # type: ignore - known output
    assert (Bool(True) ^ Bool(False)).get_value() is True  # type: ignore - known output
    assert (Bool(False) ^ Bool(True)).get_value() is True  # type: ignore - known output
    assert (Bool(False) ^ Bool(False)).get_value() is False  # type: ignore - known output

    # test negation operation
    assert (~Bool(True)).get_value() is False  # type: ignore - known output
    assert (~Bool(False)).get_value() is True  # type: ignore - known output
    assert (~Bool.symbolic()).to_dict() == {
        "op": "negate",
        "arg": {
            "type": "symbol",
            "value": {
                "owner": None,
                "name": "bool",
                "key": None,
                "attribute": None,
            },
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
        "__xor__",
        "intersects",
        "inside",
        "outside",
    ]:
        _test_unsupported(objcls, permutations, op)

    # test equatable
    assert (Integer.nullable(None) == Integer(1)).get_value() is False  # type: ignore - always a bool
    assert (Integer(1) == Integer.nullable(None)).get_value() is False  # type: ignore - always a bool
    assert (Integer.nullable(None) != Integer(1)).get_value() is True  # type: ignore - always a bool
    assert (Integer(1) != Integer.nullable(None)).get_value() is True  # type: ignore - always a bool

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
        "__xor__",
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
    _test_encoding(objcls, (1, -1), (1, -1))

    # test geojson rules
    pass  # TODO


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
        "__xor__",
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
    _test_encoding(objcls, [(0, 0), (1, 1)], [(0, 0), (1, 1)])

    # test geojson rules
    pass  # TODO


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
        "__xor__",
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
    _test_encoding(objcls, [(0, 0), (1, 1)], [(0, 0), (1, 1)])

    # test geojson rules
    pass  # TODO


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
        "__xor__",
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
    _test_encoding(objcls, [[(0, 0), (1, 1)]], [[(0, 0), (1, 1)]])

    # test geojson rules
    pass  # TODO


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
        "__xor__",
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
    _test_encoding(
        objcls,
        [[(0, 0), (1, 1), (0, 1), (0, 0)]],
        [[(0, 0), (1, 1), (0, 1), (0, 0)]],
    )

    # test property 'area'
    assert objcls.symbolic().area.is_symbolic
    assert objcls.symbolic().area.to_dict() == {
        "type": "symbol",
        "value": {
            "owner": None,
            "name": objcls.__name__.lower(),
            "key": None,
            "attribute": "area",
        },
    }
    # test that property 'area' is not accessible when object is a value
    with pytest.raises(ValueError):
        objcls(permutations[0][0]).area

    # test geojson rules
    pass  # TODO


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
        "__xor__",
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
    _test_encoding(
        objcls,
        [[[(0, 0), (1, 1), (0, 1), (0, 0)]]],
        [[[(0, 0), (1, 1), (0, 1), (0, 0)]]],
    )

    # test property 'area'
    assert objcls.symbolic().area.is_symbolic
    assert objcls.symbolic().area.to_dict() == {
        "type": "symbol",
        "value": {
            "owner": None,
            "name": objcls.__name__.lower(),
            "key": None,
            "attribute": "area",
        },
    }
    # test that property 'area' is not accessible when object is a value
    with pytest.raises(ValueError):
        objcls(permutations[0][0]).area

    # test geojson rules
    pass  # TODO


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
