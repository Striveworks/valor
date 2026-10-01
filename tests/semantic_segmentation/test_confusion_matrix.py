import numpy as np
import pytest

from valor_lite.semantic_segmentation import Loader, MetricType, Segmentation


def test_confusion_matrix_basic_segmentations(
    loader: Loader, basic_segmentations
):
    loader.add_data(basic_segmentations)
    evaluator = loader.finalize()
    result = evaluator.compute_precision_recall_iou()[
        MetricType.ConfusionMatrix
    ][0].value
    assert result == {
        "confusion_matrix": {
            "v1": {
                "v1": {"iou": 0.5},
                "v2": {"iou": 0.0},
                "other": {"iou": 0.5},
            },
            "v2": {
                "v1": {"iou": 0.0},
                "v2": {"iou": 0.5},
                "other": {"iou": 0.0},
            },
            "other": {
                "v1": {"iou": 0.0},
                "v2": {"iou": 0.5},
                "other": {"iou": 0.0},
            },
        },
        "unmatched_predictions": {
            label: {"ratio": 0.0} for label in ["v1", "v2", "other"]
        },
        "unmatched_ground_truths": {
            label: {"ratio": 0.0} for label in ["v1", "v2", "other"]
        },
    }


def test_confusion_matrix_segmentations_from_boxes(
    loader: Loader, segmentations_from_boxes
):
    loader.add_data(segmentations_from_boxes)
    result = (
        loader.finalize()
        .compute_precision_recall_iou()[MetricType.ConfusionMatrix][0]
        .value
    )
    expected = [
        [1 / 3, 0, 5000 / 530000],
        [0, 1 / 19999, 14999 / 525001],
        [5000 / 520000, 4999 / 515001, 505001 / 534999],
    ]
    for i, gt in enumerate(["v1", "v2", "other"]):
        for j, pd in enumerate(["v1", "v2", "other"]):
            assert result["confusion_matrix"][gt][pd]["iou"] == pytest.approx(
                expected[i][j]
            )
        assert result["unmatched_predictions"][gt]["ratio"] == 0
        assert result["unmatched_ground_truths"][gt]["ratio"] == 0


def test_confusion_matrix_intermediate_counting(loader: Loader):
    loader.add_data(
        [
            Segmentation(
                "image",
                np.array([[3, 4], [1, 2]]),
                np.full((2, 2), 3),
                ["a", "b", "c", "d"],
            )
        ]
    )
    matrix = loader.finalize()._compute_confusion_matrix_intermediate()
    np.testing.assert_array_equal(
        matrix,
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0],
        ],
    )
