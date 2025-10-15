from uuid import uuid4

import numpy as np
import pytest
from shapely.geometry import Polygon as ShapelyPolygon

from valor_lite.exceptions import EmptyEvaluatorError
from valor_lite.object_detection import (
    Bitmask,
    BoundingBox,
    DataLoader,
    Detection,
    Polygon,
)


def test_no_data():
    loader = DataLoader()
    with pytest.raises(EmptyEvaluatorError):
        loader.finalize()


def test_iou_computation():

    detection = Detection(
        uid="uid",
        groundtruths=[
            BoundingBox(
                uid=str(uuid4()),
                xmin=0,
                xmax=10,
                ymin=0,
                ymax=10,
                labels=["0"],
            ),
            BoundingBox(
                uid=str(uuid4()),
                xmin=100,
                xmax=110,
                ymin=100,
                ymax=110,
                labels=["0"],
            ),
            BoundingBox(
                uid=str(uuid4()),
                xmin=1000,
                xmax=1100,
                ymin=1000,
                ymax=1100,
                labels=["0"],
            ),
        ],
        predictions=[
            BoundingBox(
                uid=str(uuid4()),
                xmin=1,
                xmax=11,
                ymin=1,
                ymax=11,
                labels=["0", "1", "2"],
                scores=[0.5, 0.25, 0.25],
            ),
            BoundingBox(
                uid=str(uuid4()),
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
    evaluator = loader.finalize()

    assert evaluator.info["number_of_datums"] == 1
    assert evaluator.info["number_of_labels"] == 3
    assert evaluator.info["number_of_groundtruth_annotations"] == 3
    assert evaluator.info["number_of_prediction_annotations"] == 2
    assert evaluator.info["number_of_rows"] == 7

    tbl = evaluator._dataset.to_table()
    assert tbl.shape == (7, 12)  # 7 rows, 12 columns

    # show that three unique IOUs exist
    unique_ious = np.unique(tbl["iou"].to_numpy())
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
                    uid=str(uuid4()),
                    xmin=rect1[0],
                    xmax=rect1[1],
                    ymin=rect1[2],
                    ymax=rect1[3],
                    labels=["v1"],
                ),
            ],
            predictions=[
                Polygon(
                    uid=str(uuid4()),
                    shape=ShapelyPolygon(
                        rect1_rotated_5_degrees_around_origin
                    ),
                    labels=["v1"],
                    scores=[0.3],
                ),  # type: ignore - testing
            ],
        ),
        Detection(
            uid="uid1",
            groundtruths=[
                BoundingBox(
                    uid=str(uuid4()),
                    xmin=rect1[0],
                    xmax=rect1[1],
                    ymin=rect1[2],
                    ymax=rect1[3],
                    labels=["v1"],
                ),
            ],
            predictions=[
                Bitmask(
                    uid=str(uuid4()),
                    mask=np.ones((80, 32), dtype=bool),
                    labels=["v1"],
                    scores=[0.3],
                ),  # type: ignore - testing
            ],
        ),
        Detection(
            uid="uid1",
            groundtruths=[
                Bitmask(
                    uid=str(uuid4()),
                    mask=np.ones((80, 32), dtype=bool),
                    labels=["v1"],
                    scores=[0.3],
                ),
            ],
            predictions=[
                Polygon(
                    uid=str(uuid4()),
                    shape=ShapelyPolygon(
                        rect1_rotated_5_degrees_around_origin
                    ),
                    labels=["v1"],
                    scores=[0.3],
                ),  # type: ignore - testing
            ],
        ),
    ]

    loader = DataLoader()

    for detection in mixed_detections:
        with pytest.raises(AttributeError) as e:
            loader.add_bounding_boxes([detection])
        assert "no attribute 'extrema'" in str(e)

        with pytest.raises(AttributeError) as e:
            loader.add_polygons([detection])
        assert "no attribute 'shape'" in str(e)

        with pytest.raises(AttributeError) as e:
            loader.add_bitmasks([detection])
        assert "no attribute 'mask'" in str(e)
