import numpy as np
from numpy.typing import NDArray

try:
    _numpy_trapezoid = np.trapezoid  # numpy v2
except AttributeError:
    _numpy_trapezoid = np.trapz  # numpy v1


def trapezoid(
    x: NDArray[np.float64], y: NDArray[np.float64], axis: int
) -> NDArray[np.float64]:
    return _numpy_trapezoid(x=x, y=y, axis=axis)  # type: ignore - NumPy compatibility
