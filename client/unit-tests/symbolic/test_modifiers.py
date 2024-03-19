import pytest

from valor.symbolic.modifiers import (
    Symbol,
    Variable,
    Equatable,
    Quantifiable,
    Nullable,
    Spatial,
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


def test_equatable():
    varA = Equatable.symbolic(name="A")
    varB = Equatable.symbolic(name="B")
    varC = Equatable.symbolic(name="C")

    # Equals
    assert (varA == varB).to_dict() == {
        "op": "eq",
        "lhs": {
            "type": "symbol",
            "value": {
                "owner": None,
                "name": "A",
                "key": None,
                "attribute": None,
            }
        },
        "rhs": {
            "type": "symbol",
            "value": {
                "owner": None,
                "name": "B",
                "key": None,
                "attribute": None,
            }
        }
    }
    assert (varA == varB).to_dict() == (varA == Symbol("B")).to_dict()

    # Not Equals
    assert (varA != varB).to_dict() == {
        "op": "ne",
        "lhs": {
            "type": "symbol",
            "value": {
                "owner": None,
                "name": "A",
                "key": None,
                "attribute": None,
            }
        },
        "rhs": {
            "type": "symbol",
            "value": {
                "owner": None,
                "name": "B",
                "key": None,
                "attribute": None,
            }
        }
    }
    assert (varA != varB).to_dict() == (varA != Symbol("B")).to_dict()

    # In (exists within list)
    assert varA.in_([varB, varC]).to_dict() == {
        "op": "or",
        "args": [
            {
                "op": "eq",
                "lhs": {
                    "type": "symbol",
                    "value": {
                        "owner": None,
                        "name": "A",
                        "key": None,
                        "attribute": None,
                    }
                },
                "rhs": {
                    "type": "symbol",
                    "value": {
                        "owner": None,
                        "name": "B",
                        "key": None,
                        "attribute": None,
                    }
                }
            },
            {
                "op": "eq",
                "lhs": {
                    "type": "symbol",
                    "value": {
                        "owner": None,
                        "name": "A",
                        "key": None,
                        "attribute": None,
                    }
                },
                "rhs": {
                    "type": "symbol",
                    "value": {
                        "owner": None,
                        "name": "C",
                        "key": None,
                        "attribute": None,
                    }
                }
            }
        ]
    }
    assert (
        varA.in_([varB, varC]).to_dict() 
        == varA.in_([Symbol("B"), Symbol("C")]).to_dict()
    )


def test_quantifiable():
    pass


def test_nullable():
    pass


def test_spatial():
    pass
