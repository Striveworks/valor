import numpy as np
from numpy.typing import NDArray

try:
    # numpy v2
    _numpy_trapezoid = np.trapezoid  # type: ignore[reportAttributeAccessIssue]
except AttributeError:
    # numpy v1
    _numpy_trapezoid = np.trapz  # type: ignore[reportAttributeAccessIssue]


def trapezoid(
    x: NDArray[np.float64], y: NDArray[np.float64], axis: int
) -> NDArray[np.float64]:
    return _numpy_trapezoid(x=x, y=y, axis=axis)  # type: ignore - NumPy compatibility
