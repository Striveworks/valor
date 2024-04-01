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
    return isinstance(value, (int, float, np.floating))


def is_float(value: Any) -> bool:
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
    return isinstance(value, (float, np.floating))


def is_geojson(value: Any) -> bool:
    """
    Checks if value is an instance of geojson.

    Parameters
    ----------
    value : Any
        The value to check.

    Returns
    -------
    bool
        Whether the value conforms to the geojson standard.
    """
    if not isinstance(value, dict):
        return False
    elif set(value.keys()) != {"type", "coordinates"}:
        return False
    elif value["type"] not in {"Point", "Polygon", "MultiPolygon"}:
        return False
    elif not isinstance(value["coordinates"], list):
        return False
    else:
        return True
