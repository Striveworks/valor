import pytest

from valor.symbolic.modifiers import (
    Equatable,
    Listable,
    Nullable,
    Quantifiable,
    Spatial,
    Symbol,
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
    var_method2 = Variable(symbol=Symbol(name="test"))
    var_method3 = Variable.preprocess(value=Symbol(name="test"))
    _test_symbolic_outputs(var_method1)
    _test_symbolic_outputs(var_method2)
    _test_symbolic_outputs(var_method3)

    # valued variables are not supported in the base class
    with pytest.raises(NotImplementedError):
        Variable(value="hello")
    with pytest.raises(NotImplementedError):
        Variable.definite("hello")


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


def _test_listable(varA, varB, varC):
    listA = varA.list().symbolic("list_of_A")
    assert listA.to_dict() == {
        "type": "symbol",
        "value": {
            "attribute": None,
            "key": None,
            "name": "list_of_a",
            "owner": None,
        },
    }


def test_modifiers():

    # equatable
    A = Equatable.symbolic("A")
    B = Equatable.symbolic("B")
    C = Equatable.symbolic("C")
    _test_equatable(A, B, C)
    with pytest.raises(AttributeError):
        _test_quantifiable(A, B, C)
    with pytest.raises(AttributeError):
        _test_nullable(A, B, C)
    with pytest.raises(AttributeError):
        _test_spatial(A, B, C)
    with pytest.raises(AttributeError):
        _test_listable(A, B, C)

    # quantifiable
    A = Quantifiable.symbolic("A")
    B = Quantifiable.symbolic("B")
    C = Quantifiable.symbolic("C")
    _test_equatable(A, B, C)
    _test_quantifiable(A, B, C)
    with pytest.raises(AttributeError):
        _test_nullable(A, B, C)
    with pytest.raises(AttributeError):
        _test_spatial(A, B, C)
    with pytest.raises(AttributeError):
        _test_listable(A, B, C)

    # nullable
    A = Nullable.symbolic("A")
    B = Nullable.symbolic("B")
    C = Nullable.symbolic("C")
    _test_nullable(A, B, C)
    with pytest.raises(AttributeError):
        _test_equatable(A, B, C)
    with pytest.raises(AttributeError):
        _test_quantifiable(A, B, C)
    with pytest.raises(AttributeError):
        _test_spatial(A, B, C)
    with pytest.raises(AttributeError):
        _test_listable(A, B, C)

    # spatial
    A = Spatial.symbolic("A")
    B = Spatial.symbolic("B")
    C = Spatial.symbolic("C")
    _test_spatial(A, B, C)
    with pytest.raises(AttributeError):
        _test_equatable(A, B, C)
    with pytest.raises(AttributeError):
        _test_quantifiable(A, B, C)
    with pytest.raises(AttributeError):
        _test_nullable(A, B, C)
    with pytest.raises(AttributeError):
        _test_listable(A, B, C)

    # listable
    A = Listable.symbolic("A")
    B = Listable.symbolic("B")
    C = Listable.symbolic("C")
    _test_listable(A, B, C)
    with pytest.raises(AttributeError):
        _test_equatable(A, B, C)
    with pytest.raises(AttributeError):
        _test_quantifiable(A, B, C)
    with pytest.raises(AttributeError):
        _test_nullable(A, B, C)
    with pytest.raises(AttributeError):
        _test_spatial(A, B, C)
