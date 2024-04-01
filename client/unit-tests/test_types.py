import numpy as np

from valor.type_checks import is_float, is_numeric


def test_is_numeric():
    assert is_numeric(int(1))
    assert is_numeric(float(0.5))
    assert is_numeric(np.float32(0.5))
    assert not is_numeric(None)
    assert not is_numeric("hello world")


def test_is_float():
    assert is_float(float(0.5))
    assert is_float(np.float32(0.5))
    assert not is_float(int(1))
    assert not is_float(None)
    assert not is_float("hello world")
