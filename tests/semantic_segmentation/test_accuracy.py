from valor_lite.semantic_segmentation import (
    Loader,
    Metric,
    MetricType,
    Segmentation,
)


def test_accuracy_basic_segmentations(
    loader: Loader,
    basic_segmentations: list[Segmentation],
):
    loader.add_data(basic_segmentations)
    evaluator = loader.finalize()

    metrics = evaluator.compute_precision_recall_iou()

    actual_metrics = [m.to_dict() for m in metrics[MetricType.Accuracy]]
    expected_metrics = [
        {
            "type": "Accuracy",
            "value": 0.5,
            "parameters": {},
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_accuracy_segmentations_from_boxes(
    loader: Loader,
    segmentations_from_boxes: list[Segmentation],
):
    loader.add_data(segmentations_from_boxes)
    evaluator = loader.finalize()

    metrics = evaluator.compute_precision_recall_iou()

    actual_metrics = [m.to_dict() for m in metrics[MetricType.Accuracy]]
    expected_metrics = [
        {
            "type": "Accuracy",
            "value": 0.9444481481481481,
            "parameters": {},
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_accuracy_large_random_segmentations(
    loader: Loader,
    large_random_segmentations: list[Segmentation],
):
    loader.add_data(large_random_segmentations)
    evaluator = loader.finalize()

    metrics = evaluator.compute_precision_recall_iou()[MetricType.Accuracy]

    assert len(metrics) == 1
    assert isinstance(metrics[0], Metric)
    assert (
        round(metrics[0].value, 1)  # type: ignore - testing
        == 0.5  # random choice
    )
