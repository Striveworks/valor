from typing import List, Tuple

import numpy as np
import pytest

from valor import Label
from valor.symbolic.annotations import BoundingBox, BoundingPolygon, Raster


@pytest.fixture
def labels() -> List[Label]:
    return [
        Label(key="k1", value="v1"),
        Label(key="k2", value="v2"),
    ]


@pytest.fixture
def box_points() -> List[Tuple[float, float]]:
    return [
        (0,0),
        (10,0),
        (10,10),
        (0,10),
    ]


@pytest.fixture
def bbox() -> BoundingBox:
    return BoundingBox.from_extrema(xmin=0, xmax=10, ymin=0, ymax=10)


@pytest.fixture
def polygon(box_points) -> BoundingPolygon:
    return BoundingPolygon(value=[box_points])


@pytest.fixture
def raster_raw_mask() -> np.ndarray:
    """
    Creates a 2d numpy of bools of shape:
    | T  F |
    | F  T |
    """
    ones = np.ones((10, 10))
    zeros = np.zeros((10, 10))
    top = np.concatenate((ones, zeros), axis=1)
    bottom = np.concatenate((zeros, ones), axis=1)
    return np.concatenate((top, bottom), axis=0) == 1


@pytest.fixture
def raster(raster_raw_mask) -> Raster:
    return Raster.from_numpy(raster_raw_mask)


@pytest.fixture
def metadata() -> dict:
    return {
        "a": 1234,
        "b": 1.234,
        "c": "1234",
    }
