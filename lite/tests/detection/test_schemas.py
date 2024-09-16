import numpy as np
import pytest
from valor_lite.detection import Bitmask, BoundingBox, Detection


def test_BoundingBox():
    # groundtruth
    gt = BoundingBox(xmin=0, xmax=1, ymin=0, ymax=1, labels=[("k", "v")])

    # prediction
    pd = BoundingBox(
        xmin=-1,
        xmax=11,
        ymin=0,
        ymax=1,
        labels=[("k", "v")],
        scores=[0.7],
    )

    with pytest.raises(ValueError):
        BoundingBox(
            xmin=0,
            xmax=1,
            ymin=0,
            ymax=1,
            labels=[("k", "v")],
            scores=[0.7, 0.1],
        )
    with pytest.raises(ValueError):
        BoundingBox(
            xmin=0,
            xmax=1,
            ymin=0,
            ymax=1,
            labels=[("k", "v1"), ("k", "v2")],
            scores=[0.7],
        )

    # test `extrema` property
    assert gt.extrema == (0, 1, 0, 1)
    assert pd.extrema == (-1, 11, 0, 1)


def test_Bitmask():

    mask = mask = np.zeros((10, 10), dtype=np.bool_)
    mask[:5, :5] = True

    # groundtruth
    gt = Bitmask(mask=mask, labels=[("k", "v")])

    # prediction
    Bitmask(
        mask=mask,
        labels=[("k", "v")],
        scores=[0.7],
    )

    # test score-label matching
    with pytest.raises(ValueError):
        Bitmask(
            mask=np.zeros((10, 10), dtype=np.bool_),
            labels=[("k", "v")],
            scores=[0.7, 0.1],
        )
    with pytest.raises(ValueError):
        Bitmask(
            mask=np.zeros((10, 10), dtype=np.bool_),
            labels=[("k", "v1"), ("k", "v2")],
            scores=[0.7],
        )

    # test `to_box` function
    with pytest.raises(NotImplementedError):
        gt.to_box()


def test_Detection():

    # groundtruth
    gt = BoundingBox(xmin=0, xmax=1, ymin=0, ymax=1, labels=[("k", "v")])

    # prediction
    pd = BoundingBox(
        xmin=-1,
        xmax=11,
        ymin=0,
        ymax=1,
        labels=[("k", "v")],
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
