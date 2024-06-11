import numpy as np
import pytest

from valor import Label


def test_label():
    # valid
    l1 = Label(key="test", value="value")

    # test validation
    with pytest.raises(TypeError):
        assert Label(key=123, value="123")  # type: ignore - testing
    with pytest.raises(TypeError):
        assert Label(key="123", value=123)  # type: ignore - testing

    # test member fn `tuple`
    assert l1.tuple() == ("test", "value", None)

    # test member fn `__eq__`
    l2 = Label(key="test", value="value")
    assert l1 == l2

    # test member fn `__ne__`
    l3 = Label(key="test", value="other")
    assert l1 != l3

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

    # test validation
    with pytest.raises(TypeError):
        assert Label(key="k", value="v", score="boo")  # type: ignore - testing

    # test property `key`
    assert l1.key == "test"

    # test property `value`
    assert l1.value == "value"

    # test member fn `__eq__`
    assert s1 == s2  # type: ignore - resolved to 'Bool' as both sides are values
    assert s1 == s6  # type: ignore - resolved to 'Bool' as both sides are values
    assert not (s1 == s3)  # type: ignore - resolved to 'Bool' as both sides are values
    assert not (s1 == s4)  # type: ignore - resolved to 'Bool' as both sides are values
    assert not (s1 == s5)  # type: ignore - resolved to 'Bool' as both sides are values
    with pytest.raises(TypeError):
        assert s1 == 123
    with pytest.raises(TypeError):
        assert s1 == "123"

    # test member fn `__ne__`
    assert not (s1 != s2)  # type: ignore - resolved to 'Bool' as both sides are values
    assert s1 != s3  # type: ignore - resolved to 'Bool' as both sides are values
    assert s1 != s4  # type: ignore - resolved to 'Bool' as both sides are values
    assert s1 != s5  # type: ignore - resolved to 'Bool' as both sides are values
    with pytest.raises(TypeError):
        assert s1 != 123
    with pytest.raises(TypeError):
        assert s1 != "123"

    # test member fn `__hash__`
    assert s1.__hash__() == s2.__hash__()
    assert s1.__hash__() != s3.__hash__()
    assert s1.__hash__() != s4.__hash__()
    assert s1.__hash__() != s5.__hash__()


def test_label_equality():
    label1 = Label(key="test", value="value")
    label2 = Label(key="test", value="value")
    label3 = Label(key="test", value="other")
    label4 = Label(key="other", value="value")

    eq1 = label1 == label2
    assert type(eq1) == bool
    assert eq1

    eq2 = label1 == label3
    assert type(eq2) == bool
    assert not eq2

    eq3 = label1 == label4
    assert type(eq3) == bool
    assert not eq3
