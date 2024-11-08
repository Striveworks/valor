import numpy as np
import pytest
from shapely.geometry import Polygon as ShapelyPolygon
from valor_lite.object_detection import (
    Bitmask,
    BoundingBox,
    DataLoader,
    Detection,
    Polygon,
)


def test_no_data():
    loader = DataLoader()
    with pytest.raises(ValueError):
        loader.finalize()


def test_iou_computation():

    detection = Detection(
        uid="uid",
        groundtruths=[
            BoundingBox(xmin=0, xmax=10, ymin=0, ymax=10, labels=["0"]),
            BoundingBox(xmin=100, xmax=110, ymin=100, ymax=110, labels=["0"]),
            BoundingBox(
                xmin=1000, xmax=1100, ymin=1000, ymax=1100, labels=["0"]
            ),
        ],
        predictions=[
            BoundingBox(
                xmin=1,
                xmax=11,
                ymin=1,
                ymax=11,
                labels=["0", "1", "2"],
                scores=[0.5, 0.25, 0.25],
            ),
            BoundingBox(
                xmin=105,
                xmax=116,
                ymin=105,
                ymax=116,
                labels=["0", "1", "2"],
                scores=[0.5, 0.25, 0.25],
            ),
        ],
    )

    loader = DataLoader()
    loader.add_bounding_boxes([detection])

    assert len(loader.pairs) == 1

    # show that three unique IOUs exist
    unique_ious = np.unique(loader.pairs[0][:, 3])
    assert np.isclose(
        unique_ious, np.array([0.0, 0.12755102, 0.68067227])
    ).all()


def test_mixed_annotations(
    rect1: tuple[float, float, float, float],
    rect1_rotated_5_degrees_around_origin: tuple[float, float, float, float],
):
    """Check that we throw an error if the user tries to mix annotation types."""

    # test add_bounding_box
    mixed_detections = [
        Detection(
            uid="uid1",
            groundtruths=[
                BoundingBox(
                    xmin=rect1[0],
                    xmax=rect1[1],
                    ymin=rect1[2],
                    ymax=rect1[3],
                    labels=["v1"],
                ),
            ],
            predictions=[
                Polygon(
                    shape=ShapelyPolygon(
                        rect1_rotated_5_degrees_around_origin
                    ),
                    labels=["v1"],
                    scores=[0.3],
                ),
            ],
        ),
        Detection(
            uid="uid1",
            groundtruths=[
                BoundingBox(
                    xmin=rect1[0],
                    xmax=rect1[1],
                    ymin=rect1[2],
                    ymax=rect1[3],
                    labels=["v1"],
                ),
            ],
            predictions=[
                Bitmask(
                    mask=np.ones((80, 32), dtype=bool),
                    labels=["v1"],
                    scores=[0.3],
                ),
            ],
        ),
        Detection(
            uid="uid1",
            groundtruths=[
                Bitmask(
                    mask=np.ones((80, 32), dtype=bool),
                    labels=["v1"],
                    scores=[0.3],
                ),
            ],
            predictions=[
                Polygon(
                    shape=ShapelyPolygon(
                        rect1_rotated_5_degrees_around_origin
                    ),
                    labels=["v1"],
                    scores=[0.3],
                ),
            ],
        ),
    ]

    loader = DataLoader()

    for detection in mixed_detections:

        # anything can be converted to a bbox
        loader.add_bounding_boxes([detection])

        with pytest.raises(AttributeError) as e:
            loader.add_polygons([detection])
        assert "no attribute 'shape'" in str(e)

        with pytest.raises(AttributeError) as e:
            loader.add_bitmasks([detection])
        assert "no attribute 'mask'" in str(e)
