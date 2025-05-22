from uuid import uuid4

import numpy as np
import pytest
from shapely.geometry import Polygon as ShapelyPolygon

from valor_lite.object_detection import (
    Bitmask,
    BoundingBox,
    Detection,
    Polygon,
)


def test_BoundingBox():
    # groundtruth
    gt = BoundingBox(
        uid=str(uuid4()), xmin=0, xmax=1, ymin=0, ymax=1, labels=["label"]
    )

    # prediction
    pd = BoundingBox(
        uid=str(uuid4()),
        xmin=-1,
        xmax=11,
        ymin=0,
        ymax=1,
        labels=["label"],
        scores=[0.7],
    )

    with pytest.raises(ValueError):
        BoundingBox(
            uid=str(uuid4()),
            xmin=0,
            xmax=1,
            ymin=0,
            ymax=1,
            labels=["label"],
            scores=[0.7, 0.1],
        )
    with pytest.raises(ValueError):
        BoundingBox(
            uid=str(uuid4()),
            xmin=0,
            xmax=1,
            ymin=0,
            ymax=1,
            labels=["label1", "label2"],
            scores=[0.7],
        )
    with pytest.raises(ValueError):
        BoundingBox(
            uid=str(uuid4()),
            xmin=0,
            xmax=1,
            ymin=0,
            ymax=1,
            labels=["label1", "label2"],
            scores=[],
        )

    # test `extrema` property
    assert gt.extrema == (0, 1, 0, 1)
    assert pd.extrema == (-1, 11, 0, 1)


def test_Bitmask():

    mask = mask = np.zeros((10, 10), dtype=np.bool_)
    mask[:5, :5] = True

    # groundtruth
    assert Bitmask(uid=str(uuid4()), mask=mask, labels=["label"])

    # prediction
    Bitmask(
        uid=str(uuid4()),
        mask=mask,
        labels=["label"],
        scores=[0.7],
    )

    # test score-label matching
    with pytest.raises(ValueError):
        Bitmask(
            uid=str(uuid4()),
            mask=np.zeros((10, 10), dtype=np.bool_),
            labels=["label"],
            scores=[0.7, 0.1],
        )
    with pytest.raises(ValueError):
        Bitmask(
            uid=str(uuid4()),
            mask=mask,
            labels=["label1", "label2"],
            scores=[0.7],
        )
    with pytest.raises(ValueError):
        Bitmask(
            uid=str(uuid4()),
            mask=mask,
            labels=["label1", "label2"],
            scores=[],
        )
    with pytest.raises(ValueError):
        Bitmask(
            uid=str(uuid4()),
            mask=[[0, 0], [0, 0]],  # type: ignore - testing
            labels=["label"],
            scores=[0.7],
        )

    with pytest.raises(ValueError):
        Bitmask(
            uid=str(uuid4()),
            mask=np.array([], dtype=np.bool_),
            labels=["label"],
        )


def test_Polygon(rect1_rotated_5_degrees_around_origin):

    shape = ShapelyPolygon(rect1_rotated_5_degrees_around_origin)

    # groundtruth
    assert Polygon(uid=str(uuid4()), shape=shape, labels=["label"])

    # prediction
    Polygon(
        uid=str(uuid4()),
        shape=shape,
        labels=["label"],
        scores=[0.7],
    )

    # test score-label matching
    with pytest.raises(ValueError):
        Polygon(
            uid=str(uuid4()),
            shape=shape,
            labels=["label"],
            scores=[0.7, 0.1],
        )
    with pytest.raises(ValueError):
        Polygon(
            uid=str(uuid4()),
            shape=shape,
            labels=["label1", "label2"],
            scores=[0.7],
        )
    with pytest.raises(ValueError):
        Polygon(
            uid=str(uuid4()),
            shape=shape,
            labels=["label1", "label2"],
            scores=[],
        )
    # test that we throw a type error if the shape isn't a shapely.geometry.Polygon
    with pytest.raises(TypeError):
        Polygon(
            uid="uid",
            shape=np.zeros((10, 10), dtype=np.bool_),  # type: ignore - purposefully throwing error
            labels=["label1", "label2"],
            scores=[0.7],
        )

    with pytest.raises(ValueError):
        Polygon(uid=str(uuid4()), shape=ShapelyPolygon([]), labels=["label"])


def test_Detection():

    # groundtruth
    gt = BoundingBox(
        uid=str(uuid4()), xmin=0, xmax=1, ymin=0, ymax=1, labels=["label"]
    )

    # prediction
    pd = BoundingBox(
        uid=str(uuid4()),
        xmin=-1,
        xmax=11,
        ymin=0,
        ymax=1,
        labels=["label"],
        scores=[0.7],
    )

    Detection(
        uid="uid",
        groundtruths=[gt],
        predictions=[pd],
    )

    # test that predictions must contain scores
    with pytest.raises(ValueError):
        Detection(
            uid="uid",
            groundtruths=[gt],
            predictions=[gt],
        )
