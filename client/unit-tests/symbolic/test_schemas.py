import numpy as np
import pytest

from valor.schemas import (
    BoundingBox,
    BoundingPolygon,
    Embedding,
    Raster,
    Score,
    TaskTypeEnum,
)
from valor.schemas.symbolic.functions import (
    AppendableFunction,
    TwoArgumentFunction,
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

    # test supported methods
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

    # test nullable
    assert Score(1.0).is_none().get_value() is False  # type: ignore - always returns bool
    assert Score(1.0).is_not_none().get_value() is True  # type: ignore - always returns bool
    assert Score(None).is_none().get_value() is True  # type: ignore - always returns bool
    assert Score(None).is_not_none().get_value() is False  # type: ignore - always returns bool

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

    # test encoding
    _test_encoding(objcls, 0.2, 0.2)
    _test_encoding(objcls, None, None)

    # test that score is bounded to the range 0.0 <= score <= 1.0
    with pytest.raises(ValueError):
        Score(-0.01)
    with pytest.raises(ValueError):
        Score(1.01)


def test_tasktypeenum():
    from valor.enums import TaskType

    objcls = TaskTypeEnum

    # test supported methods
    permutations = [
        (TaskType.CLASSIFICATION, TaskType.CLASSIFICATION),
        (TaskType.CLASSIFICATION, TaskType.OBJECT_DETECTION),
        (TaskType.OBJECT_DETECTION, TaskType.OBJECT_DETECTION),
        (TaskType.OBJECT_DETECTION, TaskType.CLASSIFICATION),
    ]
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
        "is_none",
        "is_not_none",
        "intersects",
        "inside",
        "outside",
    ]:
        _test_unsupported(objcls, permutations, op)

    # test encoding
    _test_encoding(
        objcls, TaskType.CLASSIFICATION, TaskType.CLASSIFICATION.value
    )
    _test_encoding(
        objcls, TaskType.OBJECT_DETECTION, TaskType.OBJECT_DETECTION.value
    )
    _test_encoding(
        objcls,
        TaskType.SEMANTIC_SEGMENTATION,
        TaskType.SEMANTIC_SEGMENTATION.value,
    )
    _test_encoding(objcls, TaskType.EMBEDDING, TaskType.EMBEDDING.value)


def test_bounding_box():
    objcls = BoundingBox
    value = [[(0, 2), (1, 2), (1, 3), (0, 3), (0, 2)]]
    other = [[(1, 2), (2, 2), (2, 3), (1, 3), (1, 2)]]

    # test __init__
    assert objcls(value).get_value() == value

    # test 'from_extrema' classmethod
    assert objcls.from_extrema(0, 1, 2, 3).get_value() == value

    # test dictionary generation
    assert objcls.from_extrema(0, 1, 2, 3).to_dict() == {
        "type": "boundingbox",
        "value": value,
    }

    # test permutations
    permutations = [
        (value, value),
        (value, other),
        (other, other),
        (other, value),
        (value, None),
        (None, value),
    ]
    for op in ["intersects", "inside", "outside"]:
        _test_generic(objcls, permutations, op)

    # test nullable
    assert objcls.from_extrema(0, 1, 2, 3).is_none().get_value() is False  # type: ignore - always returns bool
    assert objcls.from_extrema(0, 1, 2, 3).is_not_none().get_value() is True  # type: ignore - always returns bool
    assert objcls(None).is_none().get_value() is True  # type: ignore - always returns bool
    assert objcls(None).is_not_none().get_value() is False  # type: ignore - always returns bool

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

    # test encoding
    _test_encoding(objcls, value, value)
    _test_encoding(objcls, None, None)

    # test validate box must define 5 points with first == last
    with pytest.raises(ValueError):
        BoundingBox([[(0, 0)]])
    with pytest.raises(ValueError):
        BoundingBox(value[:-1])
    value[0][-1] = (10, 10)
    with pytest.raises(ValueError):
        BoundingBox(value)


def test_bounding_polygon():
    objcls = BoundingPolygon
    value = [[(0, 2), (1, 2), (1, 3), (0, 3), (0, 2)]]
    other = [[(1, 2), (2, 2), (2, 3), (1, 3), (1, 2)]]

    # test __init__
    assert objcls(value).get_value() == value

    # test dictionary generation
    assert objcls(value).to_dict() == {
        "type": "boundingpolygon",
        "value": value,
    }

    # test permutations
    permutations = [
        (value, value),
        (value, other),
        (other, other),
        (other, value),
        (value, None),
        (None, value),
    ]
    for op in ["intersects", "inside", "outside"]:
        _test_generic(objcls, permutations, op)

    # test nullable
    assert objcls(value).is_none().get_value() is False  # type: ignore - always returns bool
    assert objcls(value).is_not_none().get_value() is True  # type: ignore - always returns bool
    assert objcls(None).is_none().get_value() is True  # type: ignore - always returns bool
    assert objcls(None).is_not_none().get_value() is False  # type: ignore - always returns bool

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

    # test encoding
    _test_encoding(objcls, value, value)
    _test_encoding(objcls, None, None)


def test_raster():
    objcls = Raster

    bitmask1 = np.full((10, 10), True)
    bitmask2 = np.full((10, 10), False)
    geom = BoundingBox.from_extrema(0, 1, 2, 3)

    value = {"mask": bitmask1, "geometry": None}
    other = {"mask": bitmask2, "geometry": geom.get_value()}

    encoded_value = {
        "mask": "iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKAQAAAAClSfIQAAAAEElEQVR4nGP8f5CJgQEXAgAzSQHUW1CW8QAAAABJRU5ErkJggg==",
        "geometry": None,
    }

    # test encoding
    _test_encoding(objcls, value, encoded_value)
    _test_encoding(objcls, None, None)

    # test permutations
    permutations = [
        (value, value),
        (value, other),
        (other, other),
        (other, value),
        (value, None),
        (None, value),
    ]
    for op in ["intersects", "inside", "outside"]:
        _test_generic(objcls, permutations, op)

    # test nullable
    assert objcls(value).is_none().get_value() is False  # type: ignore - always returns bool
    assert objcls(value).is_not_none().get_value() is True  # type: ignore - always returns bool
    assert objcls(None).is_none().get_value() is True  # type: ignore - always returns bool
    assert objcls(None).is_not_none().get_value() is False  # type: ignore - always returns bool
    assert Raster(None)

    # test 'from_numpy' classmethod
    assert Raster.from_numpy(bitmask1).to_dict() == Raster(value).to_dict()

    # test 'from_geometry' classmethod
    assert (
        Raster.from_geometry(geom, 10, 10).to_dict() == Raster(other).to_dict()
    )

    # test type validation
    with pytest.raises(TypeError):
        Raster(123)
    with pytest.raises(ValueError):
        Raster({})
    with pytest.raises(TypeError):
        Raster({"mask": 123, "geometry": None})
    with pytest.raises(ValueError) as e:
        Raster({"mask": np.zeros((10,)), "geometry": None})
    assert "2d arrays" in str(e)
    with pytest.raises(ValueError) as e:
        Raster({"mask": np.zeros((10, 10, 10)), "geometry": None})
    assert "2d arrays" in str(e)
    with pytest.raises(ValueError) as e:
        Raster({"mask": np.zeros((10, 10)), "geometry": None})
    assert "bool" in str(e)
    with pytest.raises(TypeError):
        Raster({"mask": bitmask1, "geometry": 123})

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

    # test property 'area' is not available to values
    with pytest.raises(ValueError):
        objcls(value).area

    # test property 'array'
    assert (bitmask1 == Raster(value).array).all()
    with pytest.warns(RuntimeWarning):
        Raster(other).array
    with pytest.warns(RuntimeWarning):
        Raster(None).array

    # test property 'array' is not available to symbols
    with pytest.raises(TypeError):
        Raster.symbolic().array


def test_embedding():
    objcls = Embedding
    value = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    other = [5, 6, 6, 7, 8, 9, 0, 1, 2, 3]

    # test __init__
    assert objcls(value).get_value() == value

    # test dictionary generation
    assert objcls(value).to_dict() == {
        "type": "embedding",
        "value": value,
    }

    # test permutations
    permutations = [
        (value, value),
        (value, other),
        (other, other),
        (other, value),
        (value, None),
        (None, value),
    ]
    for op in ["intersects", "inside", "outside"]:
        _test_generic(objcls, permutations, op)

    # test nullable
    assert objcls(value).is_none().get_value() is False  # type: ignore - always returns bool
    assert objcls(value).is_not_none().get_value() is True  # type: ignore - always returns bool
    assert objcls(None).is_none().get_value() is True  # type: ignore - always returns bool
    assert objcls(None).is_not_none().get_value() is False  # type: ignore - always returns bool

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

    # test encoding
    _test_encoding(objcls, value, value)
    _test_encoding(objcls, None, None)


def test_label():
    pass


def test_annotation():
    pass


def test_datum():
    pass
