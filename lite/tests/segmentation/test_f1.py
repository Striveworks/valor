from valor_lite.segmentation import DataLoader, MetricType, Segmentation


def test_f1_basic_segmenations(basic_segmentations: list[Segmentation]):
    loader = DataLoader()
    loader.add_data(basic_segmentations)
    evaluator = loader.finalize()

    metrics = evaluator.evaluate(as_dict=True)

    actual_metrics = [m for m in metrics[MetricType.F1]]
    expected_metrics = [
        {
            "type": "F1",
            "value": 2 / 3,
            "parameters": {"label": "v1"},
        },
        {
            "type": "F1",
            "value": 2 / 3,
            "parameters": {"label": "v2"},
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_f1_segmentations_from_boxes(
    segmentations_from_boxes: list[Segmentation],
):
    loader = DataLoader()
    loader.add_data(segmentations_from_boxes)
    evaluator = loader.finalize()

    metrics = evaluator.evaluate(as_dict=True)

    actual_metrics = [m for m in metrics[MetricType.F1]]
    expected_metrics = [
        {
            "type": "F1",
            "value": 0.5,  # 50% overlap
            "parameters": {"label": "v1"},
        },
        {
            "type": "F1",
            "value": 1 / 10000,  # overlaps 1 pixel out of 20,000
            "parameters": {"label": "v2"},
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics
