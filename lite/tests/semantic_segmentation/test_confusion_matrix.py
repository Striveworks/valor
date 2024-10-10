from valor_lite.semantic_segmentation import (
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

    metrics = evaluator.evaluate(as_dict=True)

    actual_metrics = [m for m in metrics[MetricType.ConfusionMatrix]]
    expected_metrics = [
        {
            "type": "ConfusionMatrix",
            "value": {
                "confusion_matrix": {
                    "v1": {"v1": {"iou": 0.5}, "v2": {"iou": 0.0}},
                    "v2": {"v1": {"iou": 0.0}, "v2": {"iou": 0.5}},
                },
                "hallucinations": {
                    "v1": {"ratio": 0.0},
                    "v2": {"ratio": 0.5},
                },
                "missing_predictions": {
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

    metrics = evaluator.evaluate(as_dict=True)

    actual_metrics = [m for m in metrics[MetricType.ConfusionMatrix]]
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
                "hallucinations": {
                    "v1": {"ratio": 5000 / 10000},  # 50% overlap
                    "v2": {
                        "ratio": 4999 / 5000
                    },  # overlaps 1 pixel out of 5000 predictions
                },
                "missing_predictions": {
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
