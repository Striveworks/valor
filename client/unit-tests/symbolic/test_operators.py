from typing import Tuple

import pytest

from valor.schemas.symbolic.operators import (
    Condition,
    Function,
    and_,
    not_,
    or_,
)
from valor.schemas.symbolic.types import Float, Integer, String


@pytest.fixture
def variables() -> Tuple[Integer, String, Float]:
    x = Integer(1)
    y = String("2")
    z = Float(0.3)
    return (x, y, z)


# def test_function(variables):
#     x, y, z = variables

#     # test stringify
#     assert (
#         Function(x, y, z).__repr__()
#         == "Function(Integer(1), String('2'), Float(0.3))"
#     )
#     assert (
#         Function(x, y, z).__str__()
#         == "Function(Integer(1), String('2'), Float(0.3))"
#     )

#     # test dictionary generation
#     assert Function(x, y, z).to_dict() == {
#         "op": "function",
#         "args": [
#             {"type": "integer", "value": 1},
#             {"type": "string", "value": "2"},
#             {"type": "float", "value": 0.3},
#         ],
#     }

#     # test stringify w/ operator
#     assert issubclass(And, Function)
#     assert (
#         And(x, y, z).__repr__() == "And(Integer(1), String('2'), Float(0.3))"
#     )
#     assert And(x, y, z).__str__() == "And(Integer(1), String('2'), Float(0.3))"

#     # test logical operators
#     assert type(Function(x) & Function(y)) is And
#     assert type(Function(x) | Function(y)) is Or
#     assert type(~Function(x)) is Not

#     # test requirement that args must have a 'to_dict' method.
#     with pytest.raises(ValueError):
#         Function(1)
#     with pytest.raises(ValueError):
#         Function("2")
#     with pytest.raises(ValueError):
#         Function(0.3)


# def test_appendable_function(variables):

#     x, y, z = variables

#     # test that all appendable functions define a overloadable function
#     assert issubclass(And, Function)
#     assert issubclass(Or, Function)

#     # test append
#     f = Function(x, y)
#     f._args.append(z)
#     assert f.to_dict() == {
#         "op": "function",
#         "args": [
#             {"type": "integer", "value": 1},
#             {"type": "string", "value": "2"},
#             {"type": "float", "value": 0.3},
#         ],
#     }

#     # continue append on the subclass 'And'
#     f1 = And(x, y)
#     f1 &= z
#     assert f1.to_dict() == {
#         "op": "and",
#         "args": [
#             {"type": "integer", "value": 1},
#             {"type": "string", "value": "2"},
#             {"type": "float", "value": 0.3},
#         ],
#     }
#     assert f1.to_dict() == And(x, y, z).to_dict()
#     assert f1.__repr__() == And(x, y, z).__repr__()
#     assert f1.__str__() == And(x, y, z).__str__()

#     # test that nested AND's collapse into one
#     f2 = And(And(x, y), z)
#     assert f1.to_dict() == f2.to_dict()

#     # test '&' operator overload
#     e1 = Integer.symbolic() == x
#     e2 = String.symbolic() == y
#     e3 = Float.symbolic() == z
#     f3 = e1 & e2 & e3
#     f4 = e1 & (e2 & e3)
#     f5 = (e1 & e2) & e3
#     assert f3.to_dict() == f4.to_dict()
#     assert f3.to_dict() == f5.to_dict()
#     assert f3.to_dict() == {
#         "op": "and",
#         "args": [
#             {
#                 "op": "eq",
#                 "lhs": {
#                     "name": "integer",
#                     "key": None,
#                 },
#                 "rhs": {"type": "integer", "value": 1},
#             },
#             {
#                 "op": "eq",
#                 "lhs": {
#                     "name": "string",
#                     "key": None,
#                 },
#                 "rhs": {"type": "string", "value": "2"},
#             },
#             {
#                 "op": "eq",
#                 "lhs": {
#                     "name": "float",
#                     "key": None,
#                 },
#                 "rhs": {"type": "float", "value": 0.3},
#             },
#         ],
#     }


# def test_one_arg_function(variables):

#     x, _, _ = variables
#     f = Function(x)

#     # test dictionary generation
#     assert f.to_dict() == {
#         "op": "function",
#         "args": {"type": "integer", "value": 1},
#     }


def test_and_function(variables):

    conditional = Integer.symbolic() == 1

    # test required number of args > 2
    with pytest.raises(ValueError):
        and_()
    with pytest.raises(ValueError):
        and_(True)

    # test bool evaluation
    assert and_(True, True) is True
    assert and_(False, True) is False
    assert and_(True, False) is False
    assert and_(False, False) is False
    assert and_(True, True, True) is True
    assert and_(True, True, False) is False
    assert and_(False, False, False) is False

    # test mixed evaluation
    assert and_(conditional, True) == conditional
    assert and_(conditional, False) is False
    assert (
        and_(conditional, conditional, True).to_dict()
        == and_(conditional, conditional).to_dict()
    )
    assert and_(conditional, conditional, False) is False

    # test function output
    assert and_(conditional, conditional, True, conditional).to_dict() == {
        "args": [
            {
                "lhs": {
                    "key": None,
                    "name": "integer",
                },
                "op": "eq",
                "rhs": {
                    "type": "integer",
                    "value": 1,
                },
            },
            {
                "lhs": {
                    "key": None,
                    "name": "integer",
                },
                "op": "eq",
                "rhs": {
                    "type": "integer",
                    "value": 1,
                },
            },
            {
                "lhs": {
                    "key": None,
                    "name": "integer",
                },
                "op": "eq",
                "rhs": {
                    "type": "integer",
                    "value": 1,
                },
            },
        ],
        "op": "and",
    }


def test_or_function(variables):

    conditional = Integer.symbolic() == 1

    # test required number of args > 2
    with pytest.raises(ValueError):
        and_()
    with pytest.raises(ValueError):
        and_(True)

    # test bool evaluation
    assert and_(True, True) is True
    assert and_(False, True) is True
    assert and_(True, False) is False
    assert and_(False, False) is False
    assert and_(True, True, True) is True
    assert and_(True, True, False) is False
    assert and_(False, False, False) is False

    # test mixed evaluation
    assert and_(conditional, True) == conditional
    assert and_(conditional, False) is False
    assert (
        and_(conditional, conditional, True).to_dict()
        == and_(conditional, conditional).to_dict()
    )
    assert and_(conditional, conditional, False) is False

    # test function output
    assert and_(conditional, conditional, True, conditional).to_dict() == {
        "args": [
            {
                "lhs": {
                    "key": None,
                    "name": "integer",
                },
                "op": "eq",
                "rhs": {
                    "type": "integer",
                    "value": 1,
                },
            },
            {
                "lhs": {
                    "key": None,
                    "name": "integer",
                },
                "op": "eq",
                "rhs": {
                    "type": "integer",
                    "value": 1,
                },
            },
            {
                "lhs": {
                    "key": None,
                    "name": "integer",
                },
                "op": "eq",
                "rhs": {
                    "type": "integer",
                    "value": 1,
                },
            },
        ],
        "op": "and",
    }


def test_condition(variables):
    x, y, z = variables

    with pytest.warns(RuntimeWarning):
        f = Condition(x, y)

    # test memebers
    assert f.lhs == x
    assert f.rhs == y

    # test dictionary generation
    assert f.to_dict() == {
        "op": "condition",
        "lhs": {"type": "integer", "value": 1},
        "rhs": {"type": "string", "value": "2"},
    }

    with pytest.warns(RuntimeWarning):
        f = Condition(x)

    assert f.to_dict() == {
        "op": "condition",
        "lhs": {"type": "integer", "value": 1},
        "rhs": None,
    }

    # test case where too many args are provided
    with pytest.raises(TypeError):
        Condition(x, y, z)  # type: ignore - testing
