import numpy as np

from valor_lite.semantic_segmentation import Loader, MetricType, Segmentation


def test_confusion_matrix_basic_segmentations(
    loader: Loader,
    basic_segmentations: list[Segmentation],
):
    loader.add_data(basic_segmentations)
    evaluator = loader.finalize()

    metrics = evaluator.compute_precision_recall_iou()

    actual_metrics = [m.to_dict() for m in metrics[MetricType.ConfusionMatrix]]
    expected_metrics = [
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {
                    "v1": {"v1": {"iou": 0.5}, "v2": {"iou": 0.0}},
                    "v2": {"v1": {"iou": 0.0}, "v2": {"iou": 0.5}},
                },
                "unmatched_predictions": {
                    "v1": {"ratio": 0.0},
                    "v2": {"ratio": 0.5},
                },
                "unmatched_ground_truths": {
                    "v1": {"ratio": 0.5},
                    "v2": {"ratio": 0.0},
                },
            },
            "parameters": {},
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_confusion_matrix_segmentations_from_boxes(
    loader: Loader,
    segmentations_from_boxes: list[Segmentation],
):
    loader.add_data(segmentations_from_boxes)
    evaluator = loader.finalize()

    metrics = evaluator.compute_precision_recall_iou()

    actual_metrics = [m.to_dict() for m in metrics[MetricType.ConfusionMatrix]]
    expected_metrics = [
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {
                    "v1": {
                        "v1": {
                            "iou": 5000 / (10000 + 10000 - 5000)
                        },  # 50% overlap
                        "v2": {"iou": 0.0},
                    },
                    "v2": {
                        "v1": {"iou": 0.0},
                        "v2": {
                            "iou": 1 / (14999 + 4999 + 1)  # overlaps 1 pixel
                        },
                    },
                },
                "unmatched_predictions": {
                    "v1": {"ratio": 5000 / 10000},  # 50% overlap
                    "v2": {
                        "ratio": 4999 / 5000
                    },  # overlaps 1 pixel out of 5000 predictions
                },
                "unmatched_ground_truths": {
                    "v1": {"ratio": 5000 / 10000},
                    "v2": {
                        "ratio": 14999 / 15000
                    },  # overlaps 1 pixel out of 15,000 groundtruths
                },
            },
            "parameters": {},
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_confusion_matrix_intermediate_counting(loader: Loader):

    segmentation = Segmentation(
        uid="uid1",
        groundtruths=np.array([[3, 4], [1, 2]], dtype=np.uint16),
        predictions=np.full((2, 2), 3, dtype=np.uint16),
        labels=["a", "b", "c", "d"],
    )

    loader.add_data([segmentation])
    evaluator = loader.finalize()

    confusion_matrix = evaluator._compute_confusion_matrix_intermediate()
    assert confusion_matrix.shape == (5, 5)
    assert (
        confusion_matrix
        == np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0],
            ]
        )
    ).all()
