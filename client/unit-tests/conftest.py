from typing import List

import numpy as np
import pytest

from valor import Label, schemas


@pytest.fixture
def labels() -> List[Label]:
    return [
        Label(key="k1", value="v1"),
        Label(key="k2", value="v2"),
    ]


@pytest.fixture
def box_points() -> List[schemas.Point]:
    return [
        schemas.Point(x=0, y=0),
        schemas.Point(x=10, y=0),
        schemas.Point(x=10, y=10),
        schemas.Point(x=0, y=10),
    ]


@pytest.fixture
def basic_polygon(box_points) -> schemas.BasicPolygon:
    return schemas.BasicPolygon(points=box_points)


@pytest.fixture
def bbox() -> schemas.BoundingBox:
    return schemas.BoundingBox.from_extrema(xmin=0, xmax=10, ymin=0, ymax=10)


@pytest.fixture
def polygon(basic_polygon) -> schemas.Polygon:
    return schemas.Polygon(boundary=basic_polygon)


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
def raster(raster_raw_mask) -> schemas.Raster:
    return schemas.Raster.from_numpy(raster_raw_mask)


@pytest.fixture
def metadata() -> dict:
    return {
        "a": 1234,
        "b": 1.234,
        "c": "1234",
    }
