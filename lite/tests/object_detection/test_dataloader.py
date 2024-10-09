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

    for input_ in mixed_detections:
        with pytest.raises(ValueError) as e:
            loader.add_bounding_boxes([input_])
        assert "but annotation is of type" in str(e)
