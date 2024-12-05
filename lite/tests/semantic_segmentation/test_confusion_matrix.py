import numpy as np
from valor_lite.semantic_segmentation import (
    Bitmask,
    DataLoader,
    MetricType,
    Segmentation,
)


def test_confusion_matrix_basic_segmentations(
    basic_segmentations: list[Segmentation],
):
    loader = DataLoader()
    loader.add_data(basic_segmentations)
    evaluator = loader.finalize()

    metrics = evaluator.evaluate()

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
    segmentations_from_boxes: list[Segmentation],
):
    loader = DataLoader()
    loader.add_data(segmentations_from_boxes)
    evaluator = loader.finalize()

    metrics = evaluator.evaluate()

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


def test_confusion_matrix_intermediate_counting():

    segmentation = Segmentation(
        uid="uid1",
        groundtruths=[
            Bitmask(
                mask=np.array([[False, False], [True, False]]),
                label="a",
            ),
            Bitmask(
                mask=np.array([[False, False], [False, True]]),
                label="b",
            ),
            Bitmask(
                mask=np.array([[True, False], [False, False]]),
                label="c",
            ),
            Bitmask(
                mask=np.array([[False, True], [False, False]]),
                label="d",
            ),
        ],
        predictions=[
            Bitmask(
                mask=np.array([[False, False], [False, False]]),
                label="a",
            ),
            Bitmask(
                mask=np.array([[False, False], [False, False]]),
                label="b",
            ),
            Bitmask(
                mask=np.array([[True, True], [True, True]]),
                label="c",
            ),
            Bitmask(
                mask=np.array([[False, False], [False, False]]),
                label="d",
            ),
        ],
    )

    loader = DataLoader()
    loader.add_data([segmentation])

    assert len(loader.matrices) == 1
    assert (
        loader.matrices[0]
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
