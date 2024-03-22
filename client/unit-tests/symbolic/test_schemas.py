import pytest

from valor.symbolic.functions import AppendableFunction, TwoArgumentFunction
from valor.symbolic.schemas import (
    Annotation,
    BoundingBox,
    BoundingPolygon,
    Datum,
    Embedding,
    Label,
    Raster,
    Score,
    StaticCollection,
    TaskTypeEnum,
)


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


def _test_to_dict(objcls, value):
    # test __init__
    assert objcls(value).to_dict() == {
        "type": objcls.__name__.lower(),
        "value": objcls(value).encode_value(),
    }
    # test definite
    assert objcls.definite(value).to_dict() == {
        "type": objcls.__name__.lower(),
        "value": objcls(value).encode_value(),
    }
    # test symbolic
    assert objcls.symbolic().to_dict() == {
        "type": "symbol",
        "value": {
            "owner": None,
            "name": objcls.__name__.lower(),
            "key": None,
            "attribute": None,
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


def _test_resolvable(objcls, permutations, op):
    # test expressions that can be simplified to a 'Bool'
    for a, b in permutations:
        A = objcls(a)
        B = objcls(b)

        # determine truth
        truth = a.__getattribute__(op)(b)

        # test variable -> builtin against truth
        assert A.__getattribute__(op)(b).get_value() is truth
        # test variable -> variable against truth
        assert A.__getattribute__(op)(B).get_value() is truth
        # test dictionary generation
        dictA = A.to_dict()
        assert A.get_value() == a
        assert len(dictA) == 2
        assert dictA["type"] == objcls.__name__.lower()
        assert dictA["value"] == A.encode_value()
    # test expressions that cannot be simplified
    _test_generic(objcls, permutations, op)


def _test_unsupported(objcls, permutations, op):
    for a, b in permutations:
        with pytest.raises(AttributeError):
            objcls(a).__getattribute__(op)(b)


def test_score():
    objcls = Score
    permutations = [
        (0.9, 0.1),
        (0.9, 0.9),
        (0.1, 0.9),
        (0.1, 0.1),
    ]
    unresolvable_permutations = [
        (0.9, None),
        (None, 0.9),
    ]
    for op in ["__eq__", "__ne__", "__gt__", "__ge__", "__lt__", "__le__"]:
        _test_resolvable(objcls, permutations, op)
        _test_generic(objcls, unresolvable_permutations, op)
        with pytest.raises((AssertionError, TypeError)):
            _test_resolvable(objcls, unresolvable_permutations, op)

    assert Score(1.0).is_none().get_value() is False  # type: ignore - always returns bool
    assert Score(1.0).is_not_none().get_value() is True  # type: ignore - always returns bool
    assert Score(None).is_none().get_value() is True  # type: ignore - always returns bool
    assert Score(None).is_not_none().get_value() is False  # type: ignore - always returns bool

    for op in [
        "__and__",
        "__or__",
        "__xor__",
        "intersects",
        "inside",
        "outside",
    ]:
        _test_unsupported(objcls, permutations, op)

    # test encoding
    _test_encoding(objcls, 0.2, 0.2)
    _test_encoding(objcls, None, None)

    # test that score is bounded to the range 0.0 <= score <= 1.0
    with pytest.raises(TypeError):
        Score(-0.01)
    with pytest.raises(TypeError):
        Score(1.01)


def test_tasktypeenum():
    pass


def test_bounding_box():
    pass


def test_bounding_polygon():
    pass


def test_raster():
    pass


def test_embedding():
    pass


def test_static_collection():
    pass


def test_label():
    pass


def test_annotation():
    pass


def test_datum():
    pass
