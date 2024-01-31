import numpy as np
import pytest

from velour import Label


def test_label():
    # valid
    l1 = Label(key="test", value="value")

    # test `__post_init__`
    with pytest.raises(TypeError) as e:
        Label(key=123, value="123")  # type: ignore

    assert "str" in str(e)
    with pytest.raises(TypeError) as e:
        Label(key="123", value=123)  # type: ignore
    assert "str" in str(e)

    # test member fn `tuple`
    assert l1.tuple() == ("test", "value", None)

    # test member fn `__eq__`
    l2 = Label(key="test", value="value")
    assert l1 == l2

    # test member fn `__hash__`
    assert l1.__hash__() == l2.__hash__()


def test_scored_label():
    l1 = Label(key="test", value="value")

    # valid
    s1 = Label(key="test", value="value", score=0.5)
    s2 = Label(key="test", value="value", score=0.5)
    s3 = Label(key="test", value="value", score=0.1)
    s4 = Label(key="test", value="other", score=0.5)
    s5 = Label(key="other", value="value", score=0.5)
    s6 = Label(key="test", value="value", score=np.float32(0.5))

    # test `__post_init__`

    with pytest.raises(TypeError) as e:
        Label(key="k", value="v", score="boo")  # type: ignore
    assert "float" in str(e)

    # test property `key`
    assert l1.key == "test"

    # test property `value`
    assert l1.value == "value"

    # test member fn `__eq__`
    assert s1 == s2
    assert s1 == s6
    assert not s1 == s3
    assert not s1 == s4
    assert not s1 == s5
    assert not s1 == 123
    assert not s1 == "123"

    # test member fn `__eq__`
    assert not s1 != s2
    assert s1 != s3
    assert s1 != s4
    assert s1 != s5
    assert s1 != 123
    assert s1 != "123"

    # test member fn `__hash__`
    assert s1.__hash__() == s2.__hash__()
    assert s1.__hash__() != s3.__hash__()
    assert s1.__hash__() != s4.__hash__()
    assert s1.__hash__() != s5.__hash__()
