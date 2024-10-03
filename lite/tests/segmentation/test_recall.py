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
