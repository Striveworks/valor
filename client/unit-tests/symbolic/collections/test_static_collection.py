import pytest

from valor.schemas import List as SymbolicList
from valor.schemas.symbolic.collections import StaticCollection
from valor.schemas.symbolic.types import Bool, Float, Integer, String


def test_static_collection_init():
    class A(StaticCollection):
        w: Integer
        x: Float
        y: String
        z: Bool

    # test that kwargs are required
    with pytest.raises(ValueError):
        A()


def test_static_collection_symbol():
    class A(StaticCollection):
        w: Integer
        x: Float
        y: String
        z: Bool

    # test that the 'symbolic' classmethod is the same as passing a symbol
    symA = A.symbolic()
    assert symA.to_dict() == A.symbolic(name="a").to_dict()

    # test symbolic usage
    assert symA.to_dict() == {
        "type": "symbol",
        "value": {
            "owner": None,
            "name": "a",
            "key": None,
            "attribute": None,
        },
    }

    # test that members are also symbolic
    assert symA.w.to_dict() == {
        "type": "symbol",
        "value": {
            "owner": "a",
            "name": "w",
            "key": None,
            "attribute": None,
        },
    }
    assert symA.x.to_dict() == {
        "type": "symbol",
        "value": {
            "owner": "a",
            "name": "x",
            "key": None,
            "attribute": None,
        },
    }
    assert symA.y.to_dict() == {
        "type": "symbol",
        "value": {
            "owner": "a",
            "name": "y",
            "key": None,
            "attribute": None,
        },
    }
    assert symA.z.to_dict() == {
        "type": "symbol",
        "value": {
            "owner": "a",
            "name": "z",
            "key": None,
            "attribute": None,
        },
    }


def test_static_collection_value():
    class A(StaticCollection):
        w: Integer
        x: Float
        y: String
        z: Bool

    encoding = {"w": 101, "x": 0.123, "y": "foobar", "z": True}

    # test that casting to symbolics is implicit
    v1 = A(w=101, x=0.123, y="foobar", z=True)
    v2 = A(w=Integer(101), x=Float(0.123), y=String("foobar"), z=Bool(True))
    v3 = A(w=101, x=Float(0.123), y=String("foobar"), z=True)
    assert v1.to_dict() == v2.to_dict()
    assert v1.to_dict() == v3.to_dict()

    # test that kwargs can be loaded by dictionary
    v4 = A(**encoding)
    v5 = A(**encoding)
    assert v1.to_dict() == v4.to_dict()
    assert v1.to_dict() == v5.to_dict()

    # test dictionary generation
    assert v1.to_dict() == {
        "type": "a",
        "value": {"w": 101, "x": 0.123, "y": "foobar", "z": True},
    }

    # test value members
    assert isinstance(v1.w, int)
    assert v1.w == 101

    assert isinstance(v1.x, float)
    assert v1.x == 0.123

    assert isinstance(v1.y, str)
    assert v1.y == "foobar"

    assert isinstance(v1.z, bool)
    assert v1.z is True


def test__get_static_types():
    class A(StaticCollection):
        w: Integer
        x: "Float"
        y: "String"
        z: Bool

    # test parsing of forward references
    assert A._get_static_types() == {
        "w": Integer,
        "x": Float,
        "y": String,
        "z": Bool,
    }

    # test lists of variables (note: these are not directly comparable)
    class B(StaticCollection):
        w: SymbolicList[Integer]
        x: SymbolicList[Float]
        y: SymbolicList[String]
        z: SymbolicList[Bool]

    types_ = B._get_static_types()
    assert types_["w"].get_element_type() == Integer
    assert types_["x"].get_element_type() == Float
    assert types_["y"].get_element_type() == String
    assert types_["z"].get_element_type() == Bool
