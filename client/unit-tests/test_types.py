import numpy as np

from velour.types import is_floating, is_numeric


def test_is_numeric():
    assert is_numeric(int(1))
    assert is_numeric(float(0.5))
    assert is_numeric(np.float32(0.5))
    assert not is_numeric(None)
    assert not is_numeric("hello world")


def test_is_floating():
    assert is_floating(float(0.5))
    assert is_floating(np.float32(0.5))
    assert not is_floating(int(1))
    assert not is_floating(None)
    assert not is_floating("hello world")
