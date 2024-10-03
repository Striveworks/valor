from valor_lite.segmentation import DataLoader, MetricType, Segmentation


def test_recall_basic_segmenations(basic_segmentations: list[Segmentation]):
    loader = DataLoader()
    loader.add_data(basic_segmentations)
    evaluator = loader.finalize()

    metrics = evaluator.evaluate(
        score_thresholds=[0.4, 0.6],
        as_dict=True,
    )

    actual_metrics = [m for m in metrics[MetricType.Recall]]
    expected_metrics = [
        {
            "type": "Recall",
            "value": [1.0, 0.5],
            "parameters": {"score_thresholds": [0.4, 0.6], "label": "v1"},
        },
        {
            "type": "Recall",
            "value": [1.0, 1.0],
            "parameters": {"score_thresholds": [0.4, 0.6], "label": "v2"},
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

    metrics = evaluator.evaluate(
        score_thresholds=[0.0],
        as_dict=True,
    )

    actual_metrics = [m for m in metrics[MetricType.Recall]]
    expected_metrics = [
        {
            "type": "Recall",
            "value": [0.5],  # 50% overlap
            "parameters": {"score_thresholds": [0.0], "label": "v1"},
        },
        {
            "type": "Recall",
            "value": [1 / 10000],  # overlaps 1 pixel out of 20,000
            "parameters": {"score_thresholds": [0.0], "label": "v2"},
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_recall_large_random_segmentations(
    large_random_segmenations: list[Segmentation],
):
    loader = DataLoader()
    loader.add_data(large_random_segmenations)
    evaluator = loader.finalize()

    metrics = evaluator.evaluate(
        score_thresholds=[0.0],
        as_dict=True,
    )

    actual_metrics = [m for m in metrics[MetricType.Recall]]
    expected_metrics = [
        {
            "type": "Recall",
            "value": [0.5],  # 50% overlap
            "parameters": {"score_thresholds": [0.0], "label": "v1"},
        },
        {
            "type": "Recall",
            "value": [1 / 10000],  # overlaps 1 pixel out of 20,000
            "parameters": {"score_thresholds": [0.0], "label": "v2"},
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics
