import pytest

from valor.schemas import (
    And,
    Label,
    Polygon,
    String,
    TypedList,
    unpack_function,
    unpack_variable,
)


@pytest.fixture
def polygon() -> Polygon:
    coordinates = [
        [
            (125.2750725, 38.760525),
            (125.3902365, 38.775069),
            (125.5054005, 38.789613),
            (125.5051935, 38.71402425),
            (125.5049865, 38.6384355),
            (125.3902005, 38.6244225),
            (125.2754145, 38.6104095),
            (125.2752435, 38.68546725),
            (125.2750725, 38.760525),
        ]
    ]
    return Polygon(coordinates)


def test_unpack_variable():

    # String
    str_dict = String("hello world").to_dict()
    assert str_dict == {
        "type": "string",
        "value": "hello world",
    }
    str_var = unpack_variable(str_dict)
    assert isinstance(str_var, String)
    assert str_var.is_value
    assert str_var.to_dict() == str_dict

    # Symbolic string
    sym_dict = String.symbolic("x").to_dict()
    assert sym_dict == {
        "type": "symbol",
        "value": {
            "name": "x",
            "attribute": None,
            "key": None,
            "dtype": "string",
        },
    }
    sym_var = unpack_variable(sym_dict)
    assert isinstance(sym_var, String)
    assert sym_var.is_symbolic
    assert sym_var.to_dict() == sym_dict

    # String list
    strlist_dict = TypedList[String](["hello", "world"]).to_dict()
    assert strlist_dict == {
        "type": "list[string]",
        "value": ["hello", "world"],
    }
    strlist_var = unpack_variable(strlist_dict)
    assert isinstance(strlist_var, TypedList[String])  # type: ignore - List[String] returns a class type
    assert strlist_var.is_value
    assert strlist_var.to_dict() == strlist_dict

    # Symbolic string list
    strlist_dict = TypedList[String].symbolic("y").to_dict()
    assert strlist_dict == {
        "type": "symbol",
        "value": {
            "name": "y",
            "attribute": None,
            "key": None,
            "dtype": "list[string]",
        },
    }
    strlist_var = unpack_variable(strlist_dict)
    assert isinstance(strlist_var, TypedList[String])  # type: ignore - List[String] returns a class type
    assert strlist_var.is_symbolic
    assert strlist_var.to_dict() == strlist_dict


def test_unpack_function(polygon):
    expr_dict = (
        (String.symbolic("x") == "foobar")
        & (
            TypedList[Label].symbolic("y")
            == [Label(key="k1", value="v1"), Label(key="k2", value="v2")]
        )
    ).to_dict()
    assert expr_dict == {
        "op": "and",
        "args": [
            {
                "op": "eq",
                "lhs": {
                    "type": "symbol",
                    "value": {
                        "attribute": None,
                        "key": None,
                        "name": "x",
                        "dtype": "string",
                    },
                },
                "rhs": {
                    "type": "string",
                    "value": "foobar",
                },
            },
            {
                "op": "eq",
                "lhs": {
                    "type": "symbol",
                    "value": {
                        "attribute": None,
                        "key": None,
                        "name": "y",
                        "dtype": "list[label]",
                    },
                },
                "rhs": {
                    "type": "list[label]",
                    "value": [
                        {
                            "key": "k1",
                            "score": None,
                            "value": "v1",
                        },
                        {
                            "key": "k2",
                            "score": None,
                            "value": "v2",
                        },
                    ],
                },
            },
        ],
    }
    expr_func = unpack_function(expr_dict, additional_types=[Label])
    assert isinstance(expr_func, And)
    assert expr_func.to_dict() == expr_dict
