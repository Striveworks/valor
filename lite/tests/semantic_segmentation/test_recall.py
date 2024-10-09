from valor_lite.semantic_segmentation import (
    DataLoader,
    MetricType,
    Recall,
    Segmentation,
)


def test_recall_basic_segmentations(basic_segmentations: list[Segmentation]):
    loader = DataLoader()
    loader.add_data(basic_segmentations)
    evaluator = loader.finalize()

    metrics = evaluator.evaluate(as_dict=True)

    actual_metrics = [m for m in metrics[MetricType.Recall]]
    expected_metrics = [
        {
            "type": "Recall",
            "value": 0.5,
            "parameters": {"label": "v1"},
        },
        {
            "type": "Recall",
            "value": 1.0,
            "parameters": {"label": "v2"},
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_recall_segmentations_from_boxes(
    segmentations_from_boxes: list[Segmentation],
):
    loader = DataLoader()
    loader.add_data(segmentations_from_boxes)
    evaluator = loader.finalize()

    metrics = evaluator.evaluate(as_dict=True)

    actual_metrics = [m for m in metrics[MetricType.Recall]]
    expected_metrics = [
        {
            "type": "Recall",
            "value": 0.5,  # 50% overlap
            "parameters": {"label": "v1"},
        },
        {
            "type": "Recall",
            "value": 1 / 15000,  # overlaps 1 pixel out of 15,000 groundtruths
            "parameters": {"label": "v2"},
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_recall_large_random_segmentations(
    large_random_segmentations: list[Segmentation],
):
    loader = DataLoader()
    loader.add_data(large_random_segmentations)
    evaluator = loader.finalize()

    metrics = evaluator.evaluate()

    for m in metrics[MetricType.Recall]:
        assert isinstance(m, Recall)
        match m.label:
            case "v1":
                assert round(m.value, 1) == 0.9
            case "v2":
                assert round(m.value, 2) == 0.09
            case "v3":
                assert round(m.value, 2) == 0.01
            case "v4":
                assert round(m.value, 1) == 0.4
            case "v5":
                assert round(m.value, 1) == 0.4
            case "v6":
                assert round(m.value, 1) == 0.1
            case "v7":
                assert round(m.value, 1) == 0.3
            case "v8":
                assert round(m.value, 1) == 0.3
            case "v9":
                assert round(m.value, 1) == 0.3
            case _:
                assert False
