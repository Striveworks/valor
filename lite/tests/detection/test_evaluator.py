from valor_lite.detection import DataLoader, Detection, MetricType


def test_metadata_using_torch_metrics_example(
    torchmetrics_detections: list[Detection],
):
    """
    cf with torch metrics/pycocotools results listed here:
    https://github.com/Lightning-AI/metrics/blob/107dbfd5fb158b7ae6d76281df44bd94c836bfce/tests/unittests/detection/test_map.py#L231
    """
    manager = DataLoader()
    manager.add_data(torchmetrics_detections)
    evaluator = manager.finalize()

    assert evaluator.ignored_prediction_labels == [("class", "3")]
    assert evaluator.missing_prediction_labels == []
    assert evaluator.n_datums == 4
    assert evaluator.n_labels == 6
    assert evaluator.n_groundtruths == 20
    assert evaluator.n_predictions == 19

    assert evaluator.metadata == {
        "ignored_prediction_labels": [
            ("class", "3"),
        ],
        "missing_prediction_labels": [],
        "n_datums": 4,
        "n_labels": 6,
        "n_groundtruths": 20,
        "n_predictions": 19,
    }


def test_no_groundtruths(detections_no_groundtruths):

    manager = DataLoader()
    manager.add_data(detections_no_groundtruths)
    evaluator = manager.finalize()

    assert evaluator.ignored_prediction_labels == [("k1", "v1"), ("k2", "v2")]
    assert evaluator.missing_prediction_labels == []
    assert evaluator.n_datums == 2
    assert evaluator.n_labels == 2
    assert evaluator.n_groundtruths == 0
    assert evaluator.n_predictions == 3

    metrics = evaluator.evaluate(
        iou_thresholds=[0.5],
        score_thresholds=[0.5],
    )

    assert len(metrics[MetricType.AP]) == 0


def test_no_predictions(detections_no_predictions):

    manager = DataLoader()
    manager.add_data(detections_no_predictions)
    evaluator = manager.finalize()

    assert evaluator.ignored_prediction_labels == []
    assert evaluator.missing_prediction_labels == [("k1", "v1"), ("k2", "v2")]
    assert evaluator.n_datums == 2
    assert evaluator.n_labels == 2
    assert evaluator.n_groundtruths == 3
    assert evaluator.n_predictions == 0

    metrics = evaluator.evaluate(
        iou_thresholds=[0.5],
        score_thresholds=[0.5],
    )

    assert len(metrics[MetricType.AP]) == 2

    # test AP
    actual_metrics = [m.to_dict() for m in metrics[MetricType.AP]]
    expected_metrics = [
        {
            "type": "AP",
            "value": 0.0,
            "parameters": {
                "iou_threshold": 0.5,
                "label": {"key": "k1", "value": "v1"},
            },
        },
        {
            "type": "AP",
            "value": 0.0,
            "parameters": {
                "iou_threshold": 0.5,
                "label": {"key": "k2", "value": "v2"},
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics
