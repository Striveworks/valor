from uuid import uuid4

import numpy as np
import pytest
from shapely.geometry import Polygon as ShapelyPolygon

from valor_lite.exceptions import EmptyCacheError
from valor_lite.object_detection import (
    Bitmask,
    BoundingBox,
    Detection,
    Loader,
    Polygon,
)


def test_no_data(loader: Loader):
    with pytest.raises(EmptyCacheError):
        loader.finalize()


def test_unmatched_prediction_labels_emitted_once(loader: Loader):
    detection = Detection(
        uid="datum",
        groundtruths=[
            BoundingBox(
                uid=f"gt-{i}",
                xmin=0,
                xmax=10,
                ymin=0,
                ymax=10,
                labels=[label],
            )
            for i, label in enumerate(["first", "second", "third"])
        ],
        predictions=[
            BoundingBox(
                uid="unmatched",
                xmin=20,
                xmax=30,
                ymin=20,
                ymax=30,
                labels=["prediction-a", "prediction-b"],
                scores=[0.9, 0.2],
                metadata={"pd_rect": "unmatched-metadata"},
            ),
            BoundingBox(
                uid="matched",
                xmin=0,
                xmax=10,
                ymin=0,
                ymax=10,
                labels=["first"],
                scores=[0.8],
            ),
        ],
    )
    loader.add_bounding_boxes([detection])
    evaluator = loader.finalize()
    rows = [
        row
        for table in evaluator._detailed_reader.iterate_tables()
        for row in table.to_pylist()
    ]
    unmatched = [row for row in rows if row["pd_uid"] == "unmatched"]
    assert len(rows) == 5
    assert len(unmatched) == 2
    assert {(row["pd_label"], row["pd_score"]) for row in unmatched} == {
        ("prediction-a", 0.9),
        ("prediction-b", 0.2),
    }
    assert all(row["gt_id"] == -1 for row in unmatched)
    assert all(row["pd_rect"] == "unmatched-metadata" for row in unmatched)
    assert evaluator._index_to_label == {
        0: "first",
        1: "prediction-a",
        2: "prediction-b",
        3: "second",
        4: "third",
    }
    assert evaluator.info.number_of_groundtruth_annotations == 3
    assert evaluator.info.number_of_prediction_annotations == 2


def test_iou_computation(loader: Loader):

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

    loader.add_bounding_boxes([detection])
    evaluator = loader.finalize()

    assert evaluator._detailed_reader.count_rows() == 7

    # show that three unique IOUs exist
    for tbl in evaluator._detailed_reader.iterate_tables():
        unique_ious = np.unique(tbl["iou"].to_numpy())
        assert np.isclose(
            unique_ious, np.array([0.0, 0.12755102, 0.68067227])
        ).all()


def test_mixed_annotations(
    loader: Loader,
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


def test_add_data_metadata_handling(loader: Loader):
    loader.add_bounding_boxes(
        detections=[
            Detection(
                uid="0",
                metadata={"datum_uid": "a"},
                groundtruths=[
                    BoundingBox(
                        uid="0",
                        xmin=0,
                        xmax=10,
                        ymin=0,
                        ymax=10,
                        labels=["dog"],
                        metadata={"datum_uid": "b"},
                    )
                ],
                predictions=[
                    BoundingBox(
                        uid="0",
                        xmin=0,
                        xmax=10,
                        ymin=0,
                        ymax=10,
                        labels=["dog"],
                        scores=[1.0],
                        metadata={"datum_uid": "c"},
                    )
                ],
            ),
            Detection(
                uid="1",
                metadata={"datum_uid": "a"},
                groundtruths=[
                    BoundingBox(
                        uid="0",
                        xmin=0,
                        xmax=10,
                        ymin=0,
                        ymax=10,
                        labels=["dog"],
                        metadata={"datum_uid": "b"},
                    )
                ],
                predictions=[
                    BoundingBox(
                        uid="0",
                        xmin=0,
                        xmax=10,
                        ymin=0,
                        ymax=10,
                        labels=["dog"],
                        scores=[1.0],
                        metadata={"datum_uid": "c"},
                    )
                ],
            ),
        ]
    )
    loader._detailed_writer.flush()
    reader = loader._detailed_writer.to_reader()

    datum_uids = set()
    for tbl in reader.iterate_tables():
        assert set(tbl.column_names) == {
            "datum_id",
            "datum_uid",
            "gt_id",
            "gt_label",
            "gt_label_id",
            "gt_rect",
            "gt_uid",
            "iou",
            "pd_id",
            "pd_label",
            "pd_label_id",
            "pd_rect",
            "pd_score",
            "pd_uid",
        }
        for uid in tbl["datum_uid"].to_pylist():
            datum_uids.add(uid)
    assert datum_uids == {"0", "1"}
