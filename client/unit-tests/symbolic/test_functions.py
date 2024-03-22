from typing import Tuple

import pytest

from valor.symbolic.atomics import Float, Integer, String
from valor.symbolic.functions import (
    And,
    AppendableFunction,
    Function,
    Negate,
    OneArgumentFunction,
    Or,
    TwoArgumentFunction,
    Xor,
)


@pytest.fixture
def variables() -> Tuple[Integer, String, Float]:
    x = Integer(1)
    y = String("2")
    z = Float(0.3)
    return (x, y, z)


def test_function(variables):
    x, y, z = variables

    # test stringify
    assert Function(x, y, z).__repr__() == "Function(1, '2', 0.3)"
    assert Function(x, y, z).__str__() == "Function(1, '2', 0.3)"

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
    assert And(x, y, z).__repr__() == "And(1, '2', 0.3)"
    assert And(x, y, z).__str__() == "(1 & '2' & 0.3)"

    # test logical operators
    assert type(Function(x) & Function(y)) is And
    assert type(Function(x) | Function(y)) is Or
    assert type(Function(x) ^ Function(y)) is Xor
    assert type(~Function(x)) is Negate

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
    # test base class is not fully implemented
    assert AppendableFunction._function is None
    with pytest.raises(NotImplementedError):
        AppendableFunction(x, y)

    # test that all appendable functions define a overloadable function
    assert issubclass(And, AppendableFunction)
    assert issubclass(Or, AppendableFunction)
    assert issubclass(Xor, AppendableFunction)
    assert And._function == "__and__"
    assert Or._function == "__or__"
    assert Xor._function == "__xor__"

    # continue tests using And
    f = And(x, y)
    f.append(z)

    # test dictionary generation
    assert f.to_dict() == {
        "op": "and",
        "args": [
            {"type": "integer", "value": 1},
            {"type": "string", "value": "2"},
            {"type": "float", "value": 0.3},
        ],
    }

    # test that all args can be defined at initialization to yield same result
    assert f.to_dict() == And(x, y, z).to_dict()
    assert f.__repr__() == And(x, y, z).__repr__()
    assert f.__str__() == And(x, y, z).__str__()


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
