import pytest
from pydantic import ValidationError

from velour_api import schemas


def test_label_no_scores():
    # valid
    l1 = schemas.Label(key="k1", value="v1")
    l2 = schemas.Label(key="k2", value="v2")

    # test property `key`
    with pytest.raises(ValidationError):
        schemas.Label(key=("k1",), value="v1")

    # test property `value`
    with pytest.raises(ValidationError):
        schemas.Label(key="k1", value=("v1",))

    # test member fn `__eq__`
    assert l1 == l1
    assert not l1 == l2

    # test member fn `__hash__`
    assert l1.__hash__() == l1.__hash__()
    assert l1.__hash__() != l2.__hash__()


def test_label_with_scores():
    # test property `score`
    with pytest.raises(ValidationError):
        schemas.Label(key="k1", value="v1", score="score")

    l1 = schemas.Label(key="k1", value="v1", score=0.75)
    l2 = schemas.Label(key="k1", value="v1", score=0.5)
    l3 = schemas.Label(key="k1", value="v1")
    l4 = schemas.Label(key="k1", value="v1", score=0.75000000000000001)

    assert l1 != l2
    assert l2 != l3
    assert l1 == l4
