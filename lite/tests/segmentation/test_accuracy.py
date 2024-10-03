from valor_lite.segmentation import DataLoader, MetricType, Segmentation


def test_accuracy_basic_segmenations(basic_segmentations: list[Segmentation]):
    loader = DataLoader()
    loader.add_data(basic_segmentations)
    evaluator = loader.finalize()

    metrics = evaluator.evaluate(
        score_thresholds=[0.4, 0.6],
        as_dict=True,
    )

    actual_metrics = [m for m in metrics[MetricType.Accuracy]]
    expected_metrics = [
        {
            "type": "Accuracy",
            "value": [
                0.75,
                1.0,
            ],
            "parameters": {
                "score_thresholds": [0.4, 0.6],
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics
