import numpy as np
import pytest

from valor_lite.exceptions import EmptyCacheError
from valor_lite.semantic_segmentation import Loader, MetricType, Segmentation


def test_no_data(loader: Loader):
    with pytest.raises(EmptyCacheError):
        loader.finalize()


def test_zero_is_an_ordinary_class(loader: Loader):
    loader.add_data(
        [
            Segmentation(
                "0",
                np.zeros((10, 10), dtype=np.uint16),
                np.zeros((10, 10), dtype=np.uint16),
                ["sky"],
            )
        ]
    )
    evaluator = loader.finalize()
    assert evaluator.info.number_of_datums == 1
    assert evaluator.info.number_of_pixels == 100
    assert evaluator.info.number_of_groundtruth_pixels == 100
    assert evaluator.info.number_of_prediction_pixels == 100
    metrics = evaluator.compute_precision_recall_iou()
    assert metrics[MetricType.mIOU][0].value == 1.0
    assert metrics[MetricType.IOU][0].parameters == {"label": "sky"}
    assert metrics[MetricType.IOU][0].value == 1.0


def test_reordered_labels_across_batches(loader: Loader):
    loader.add_data(
        [
            Segmentation(
                "0",
                np.array([[0, 1]]),
                np.array([[1, 1]]),
                ["sky", "road", "absent"],
            )
        ]
    )
    loader.add_data(
        [
            Segmentation(
                "1", np.array([[1, 0]]), np.array([[0, 0]]), ["road", "sky"]
            )
        ]
    )
    evaluator = loader.finalize()
    assert evaluator._index_to_label == {0: "sky", 1: "road", 2: "absent"}
    np.testing.assert_array_equal(
        evaluator._compute_confusion_matrix_intermediate(),
        [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 2, 0], [0, 0, 0, 0]],
    )
    assert (
        evaluator.info.number_of_rows == 5
    )  # Four pairs and one absent-class row.
    assert (
        evaluator.compute_precision_recall_iou()[MetricType.mIOU][0].value
        == 1 / 6
    )


def test_add_data_metadata_handling(loader: Loader):
    loader.add_data(
        [
            Segmentation(
                uid="image",
                groundtruths=np.array([[0, 1]]),
                predictions=np.array([[1, 0]]),
                labels=["sky", "road"],
                metadata={
                    "datum_uid": "incorrect",
                    "count": 999,
                    "gt_xmin": -1,
                },
                groundtruth_metadata={0: {"gt_xmin": 10}, 1: {"gt_xmin": 20}},
                prediction_metadata={0: {"pd_xmin": 30}, 1: {"pd_xmin": 40}},
            )
        ]
    )
    evaluator = loader.finalize()
    rows = [
        row
        for table in evaluator._reader.iterate_tables()
        for row in table.to_pylist()
    ]
    assert len(rows) == 2
    assert all(
        row["datum_uid"] == "image" and row["count"] == 1 for row in rows
    )
    assert [
        (row["gt_label"], row["pd_label"], row["gt_xmin"], row["pd_xmin"])
        for row in rows
    ] == [("sky", "road", 10, 40), ("road", "sky", 20, 30)]


def test_high_ids_and_global_vocabulary_are_not_narrowed(loader: Loader):
    loader.add_data(
        [
            Segmentation(
                "0",
                np.array([[0, 65535]], dtype=np.uint16),
                np.array([[65535, 0]], dtype=np.uint16),
                [str(i) for i in range(65536)],
            )
        ]
    )
    loader.add_data(
        [Segmentation("1", np.array([[0]]), np.array([[0]]), ["extra"])]
    )
    evaluator = loader.finalize()
    assert evaluator.info.number_of_labels == 65537
    assert evaluator.info.number_of_pixels == 3
    rows = [
        row
        for table in evaluator._reader.iterate_tables()
        for row in table.to_pylist()
    ]
    assert [
        (row["gt_label_id"], row["pd_label_id"], row["count"])
        for row in rows
        if row["count"]
    ] == [(0, 65535, 1), (65535, 0, 1), (65536, 65536, 1)]


def test_pixel_counts_are_not_uint16(loader: Loader):
    array = np.zeros((257, 257), dtype=np.uint16)
    loader.add_data([Segmentation("image", array, array, ["sky"])])
    evaluator = loader.finalize()
    assert evaluator.info.number_of_pixels == 66049
    matrix = evaluator._compute_confusion_matrix_intermediate()
    assert matrix.dtype == np.uint64
    assert matrix[1, 1] == 66049
