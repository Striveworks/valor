from valor_lite.detection import DataLoader, Detection, MetricType


def test_metadata_using_torch_metrics_example(
    torchmetrics_detections: list[Detection],
):
    """
    cf with torch metrics/pycocotools results listed here:
    https://github.com/Lightning-AI/metrics/blob/107dbfd5fb158b7ae6d76281df44bd94c836bfce/tests/unittests/detection/test_map.py#L231
    """
    loader = DataLoader()
    loader.add_bounding_boxes(torchmetrics_detections)
    evaluator = loader.finalize()

    assert evaluator.ignored_prediction_labels == ["3"]
    assert evaluator.missing_prediction_labels == []
    assert evaluator.n_datums == 4
    assert evaluator.n_labels == 6
    assert evaluator.n_groundtruths == 20
    assert evaluator.n_predictions == 19

    assert evaluator.metadata == {
        "ignored_prediction_labels": ["3"],
        "missing_prediction_labels": [],
        "n_datums": 4,
        "n_labels": 6,
        "n_groundtruths": 20,
        "n_predictions": 19,
    }


def test_no_groundtruths(detections_no_groundtruths):

    loader = DataLoader()
    loader.add_bounding_boxes(detections_no_groundtruths)
    evaluator = loader.finalize()

    assert evaluator.ignored_prediction_labels == ["v1"]
    assert evaluator.missing_prediction_labels == []
    assert evaluator.n_datums == 2
    assert evaluator.n_labels == 1
    assert evaluator.n_groundtruths == 0
    assert evaluator.n_predictions == 2

    metrics = evaluator.evaluate(
        iou_thresholds=[0.5],
        score_thresholds=[0.5],
    )

    assert len(metrics[MetricType.AP]) == 0


def test_no_predictions(detections_no_predictions):

    loader = DataLoader()
    loader.add_bounding_boxes(detections_no_predictions)
    evaluator = loader.finalize()

    assert evaluator.ignored_prediction_labels == []
    assert evaluator.missing_prediction_labels == ["v1"]
    assert evaluator.n_datums == 2
    assert evaluator.n_labels == 1
    assert evaluator.n_groundtruths == 2
    assert evaluator.n_predictions == 0

    metrics = evaluator.evaluate(
        iou_thresholds=[0.5],
        score_thresholds=[0.5],
    )

    assert len(metrics[MetricType.AP]) == 1

    # test AP
    actual_metrics = [m.to_dict() for m in metrics[MetricType.AP]]
    expected_metrics = [
        {
            "type": "AP",
            "value": 0.0,
            "parameters": {
                "iou_threshold": 0.5,
                "label": "v1",
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_metrics_to_return(basic_detections_first_class: list[Detection]):

    loader = DataLoader()
    loader.add_bounding_boxes(basic_detections_first_class)
    evaluator = loader.finalize()

    metrics_to_return = [
        MetricType.AP,
        MetricType.AR,
    ]
    metrics = evaluator.evaluate(metrics_to_return)
    assert metrics.keys() == set(metrics_to_return)

    metrics_to_return = [
        MetricType.AP,
        MetricType.AR,
        MetricType.ConfusionMatrix,
    ]
    metrics = evaluator.evaluate(metrics_to_return)
    assert metrics.keys() == set(metrics_to_return)
