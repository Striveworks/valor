from valor_lite.segmentation import (
    Accuracy,
    DataLoader,
    MetricType,
    Segmentation,
)


def test_accuracy_basic_segmenations(basic_segmentations: list[Segmentation]):
    loader = DataLoader()
    loader.add_data(basic_segmentations)
    evaluator = loader.finalize()

    metrics = evaluator.evaluate(as_dict=True)

    actual_metrics = [m for m in metrics[MetricType.Accuracy]]
    expected_metrics = [
        {
            "type": "Accuracy",
            "value": 2 / 3,
            "parameters": {},
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_accuracy_segmentations_from_boxes(
    segmentations_from_boxes: list[Segmentation],
):
    loader = DataLoader()
    loader.add_data(segmentations_from_boxes)
    evaluator = loader.finalize()

    metrics = evaluator.evaluate(as_dict=True)

    actual_metrics = [m for m in metrics[MetricType.Accuracy]]
    expected_metrics = [
        {
            "type": "Accuracy",
            "value": 0.3334,
            "parameters": {},
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_accuracy_large_random_segmentations(
    large_random_segmenations: list[Segmentation],
):
    loader = DataLoader()
    loader.add_data(large_random_segmenations)
    evaluator = loader.finalize()

    metrics = evaluator.evaluate()[MetricType.Accuracy]

    assert len(metrics) == 1
    assert isinstance(metrics[0], Accuracy)
    assert round(metrics[0].value, 1) == 0.5
