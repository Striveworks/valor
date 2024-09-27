import numpy as np
import pytest
from shapely.geometry import Polygon as ShapelyPolygon
from valor_lite.detection import Bitmask, BoundingBox, Detection, Polygon


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

    # test `to_box` method
    box = gt.to_box()
    assert box
    assert box.extrema == (0, 4, 0, 4)

    empty_box = Bitmask(mask=np.array([]), labels=[("k", "v")])

    assert empty_box.to_box() is None


def test_Polygon(rect1_rotated_5_degrees_around_origin):

    shape = ShapelyPolygon(rect1_rotated_5_degrees_around_origin)

    # groundtruth
    gt = Polygon(shape=shape, labels=[("k", "v")])

    # prediction
    Polygon(
        shape=shape,
        labels=[("k", "v")],
        scores=[0.7],
    )

    # test score-label matching
    with pytest.raises(ValueError):
        Polygon(
            shape=shape,
            labels=[("k", "v")],
            scores=[0.7, 0.1],
        )
    with pytest.raises(ValueError):
        Polygon(
            shape=shape,
            labels=[("k", "v1"), ("k", "v2")],
            scores=[0.7],
        )
    # test that we throw a type error if the shape isn't a shapely.geometry.Polygon
    with pytest.raises(TypeError):
        Polygon(
            shape=np.zeros((10, 10), dtype=np.bool_),  # type: ignore - purposefully throwing error
            labels=[("k", "v1"), ("k", "v2")],
            scores=[0.7],
        )

    # test `to_box` method
    box = gt.to_box()
    assert box
    assert box.extrema == (
        6.475717271011129,
        58.90012445802815,
        10.833504408394036,
        45.07713248852931,
    )

    empty_box = Polygon(shape=ShapelyPolygon([]), labels=[("k", "v")])

    assert empty_box.to_box() is None


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
