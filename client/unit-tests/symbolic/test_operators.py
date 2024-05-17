from typing import Tuple

import pytest

from valor.schemas.symbolic.operators import (
    And,
    AppendableFunction,
    Function,
    Not,
    OneArgumentFunction,
    Or,
    TwoArgumentFunction,
    Xor,
)
from valor.schemas.symbolic.types import Float, Integer, String


@pytest.fixture
def variables() -> Tuple[Integer, String, Float]:
    x = Integer(1)
    y = String("2")
    z = Float(0.3)
    return (x, y, z)


def test_function(variables):
    x, y, z = variables

    # test stringify
    assert (
        Function(x, y, z).__repr__()
        == "Function(Integer(1), String('2'), Float(0.3))"
    )
    assert (
        Function(x, y, z).__str__()
        == "Function(Integer(1), String('2'), Float(0.3))"
    )

    # test dictionary generation
    assert Function(x, y, z).to_dict() == {
        "op": "function",
        "args": [
            {"type": "integer", "value": 1},
            {"type": "string", "value": "2"},
            {"type": "float", "value": 0.3},
        ],
    }

    # test stringify w/ operator
    assert issubclass(And, Function)
    assert And._operator is not None
    assert (
        And(x, y, z).__repr__() == "And(Integer(1), String('2'), Float(0.3))"
    )
    assert And(x, y, z).__str__() == "(Integer(1) & String('2') & Float(0.3))"

    # test logical operators
    assert type(Function(x) & Function(y)) is And
    assert type(Function(x) | Function(y)) is Or
    assert type(Function(x) ^ Function(y)) is Xor
    assert type(~Function(x)) is Not

    # test requirement that args must have a 'to_dict' method.
    with pytest.raises(ValueError):
        Function(1)
    with pytest.raises(ValueError):
        Function("2")
    with pytest.raises(ValueError):
        Function(0.3)


def test_appendable_function(variables):
    assert issubclass(AppendableFunction, Function)

    x, y, z = variables

    # test case where too few args
    with pytest.raises(TypeError):
        AppendableFunction(x)  # type: ignore - edge case test

    # test that all appendable functions define a overloadable function
    assert issubclass(And, AppendableFunction)
    assert issubclass(Or, AppendableFunction)
    assert issubclass(Xor, AppendableFunction)

    # test append
    f = AppendableFunction(x, y)
    f.append(z)
    assert f.to_dict() == {
        "op": "appendablefunction",
        "args": [
            {"type": "integer", "value": 1},
            {"type": "string", "value": "2"},
            {"type": "float", "value": 0.3},
        ],
    }

    # continue append on the subclass 'And'
    f1 = And(x, y)
    f1.append(z)
    assert f1.to_dict() == {
        "op": "and",
        "args": [
            {"type": "integer", "value": 1},
            {"type": "string", "value": "2"},
            {"type": "float", "value": 0.3},
        ],
    }
    assert f1.to_dict() == And(x, y, z).to_dict()
    assert f1.__repr__() == And(x, y, z).__repr__()
    assert f1.__str__() == And(x, y, z).__str__()

    # test that nested AND's collapse into one
    f2 = And(And(x, y), z)
    assert f1.to_dict() == f2.to_dict()

    # test '&' operator overload
    e1 = Integer.symbolic() == x
    e2 = String.symbolic() == y
    e3 = Float.symbolic() == z
    f3 = e1 & e2 & e3
    f4 = e1 & (e2 & e3)
    f5 = (e1 & e2) & e3
    assert f3.to_dict() == f4.to_dict()
    assert f3.to_dict() == f5.to_dict()
    assert f3.to_dict() == {
        "op": "and",
        "args": [
            {
                "op": "eq",
                "lhs": {
                    "type": "symbol",
                    "value": {
                        "name": "integer",
                        "key": None,
                        "attribute": None,
                        "dtype": "integer",
                    },
                },
                "rhs": {"type": "integer", "value": 1},
            },
            {
                "op": "eq",
                "lhs": {
                    "type": "symbol",
                    "value": {
                        "name": "string",
                        "key": None,
                        "attribute": None,
                        "dtype": "string",
                    },
                },
                "rhs": {"type": "string", "value": "2"},
            },
            {
                "op": "eq",
                "lhs": {
                    "type": "symbol",
                    "value": {
                        "name": "float",
                        "key": None,
                        "attribute": None,
                        "dtype": "float",
                    },
                },
                "rhs": {"type": "float", "value": 0.3},
            },
        ],
    }


def test_one_arg_function(variables):
    assert issubclass(OneArgumentFunction, Function)

    x, _, _ = variables
    f = OneArgumentFunction(x)

    # test dictionary generation
    assert f.to_dict() == {
        "op": "oneargumentfunction",
        "arg": {"type": "integer", "value": 1},
    }


def test_two_arg_function(variables):
    assert issubclass(TwoArgumentFunction, Function)

    x, y, z = variables
    f = TwoArgumentFunction(x, y)

    # test memebers
    assert f.lhs == x
    assert f.rhs == y

    # test dictionary generation
    assert f.to_dict() == {
        "op": "twoargumentfunction",
        "lhs": {"type": "integer", "value": 1},
        "rhs": {"type": "string", "value": "2"},
    }

    # test cases where too few args are provided
    with pytest.raises(TypeError):
        TwoArgumentFunction(x)  # type: ignore - edge case test
    # test case where too many args are provided
    with pytest.raises(TypeError):
        TwoArgumentFunction(x, y, z)  # type: ignore - edge case test
