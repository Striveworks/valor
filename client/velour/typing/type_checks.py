from typing import Any

import numpy as np


def is_numeric(value: Any) -> bool:
    """
    Checks whether the value input is a numeric type.

    Parameters
    ----------
    value : Any
        The value to check.

    Returns
    -------
    bool
        Whether the value is a number.
    """
    return (
        isinstance(value, int)
        or isinstance(value, float)
        or isinstance(value, np.floating)
    )


def is_floating(value: Any) -> bool:
    """
    Checks whether the value input is a floating point type.

    Parameters
    ----------
    value : Any
        The value to check.

    Returns
    -------
    bool
        Whether the value is a floating point number.
    """
    return isinstance(value, float) or isinstance(value, np.floating)
