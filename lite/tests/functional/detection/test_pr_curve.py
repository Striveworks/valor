import numpy as np
from valor_lite.detection import (
    DataLoader,
    Detection,
    MetricType,
    compute_metrics,
)


def test__compute_average_precision():

    sorted_pairs = np.array(
        [
            # dt,  gt,  pd,  iou,  gl,  pl, score,
            [0.0, 0.0, 2.0, 0.25, 0.0, 0.0, 0.95],
            [0.0, 0.0, 3.0, 0.33333, 0.0, 0.0, 0.9],
            [0.0, 0.0, 4.0, 0.66667, 0.0, 0.0, 0.65],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.1],
            [0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.01],
        ]
    )

    label_counts = np.array([[1, 5, 0]])
    iou_thresholds = np.array([0.1, 0.6])
    score_thresholds = np.array([0.0])

    (_, _, _, _, _, pr_curve) = compute_metrics(
        sorted_pairs,
        label_counts=label_counts,
        iou_thresholds=iou_thresholds,
        score_thresholds=score_thresholds,
    )

    assert pr_curve.shape == (2, 1, 101)
    assert np.isclose(pr_curve[0][0], 1.0).all()
    assert np.isclose(pr_curve[1][0], 1 / 3).all()


def test_ap_metrics(basic_detections: list[Detection]):
    """
    Basic object detection test.

    groundtruths
        datum 1
            box 1 - label (k1, v1) - tp
            box 3 - label (k2, v2) - fn missing prediction
        datum 2
            box 2 - label (k1, v1) - fn misclassification

    predictions
        datum 1
            box 1 - label (k1, v1) - score 0.3 - tp
        datum 2
            box 2 - label (k2, v2) - score 0.98 - fp
    """

    manager = DataLoader()
    manager.add_data(basic_detections)
    evaluator = manager.finalize()

    metrics = evaluator.evaluate(
        iou_thresholds=[0.1, 0.6],
    )

    assert evaluator.ignored_prediction_labels == []
    assert evaluator.missing_prediction_labels == []
    assert evaluator.n_datums == 2
    assert evaluator.n_labels == 2
    assert evaluator.n_groundtruths == 3
    assert evaluator.n_predictions == 2

    # test AP
    actual_metrics = [m.to_dict() for m in metrics[MetricType.AP]]
    expected_metrics = [
        {
            "type": "AP",
            "value": 0.0,
            "parameters": {
                "iou": 0.1,
                "label": {"key": "k2", "value": "v2"},
            },
        },
        {
            "type": "AP",
            "value": 0.0,
            "parameters": {
                "iou": 0.6,
                "label": {"key": "k2", "value": "v2"},
            },
        },
        {
            "type": "AP",
            "value": 0.504950495049505,
            "parameters": {
                "iou": 0.1,
                "label": {"key": "k1", "value": "v1"},
            },
        },
        {
            "type": "AP",
            "value": 0.504950495049505,
            "parameters": {
                "iou": 0.6,
                "label": {"key": "k1", "value": "v1"},
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics

    # test mAP
    actual_metrics = [m.to_dict() for m in metrics[MetricType.mAP]]
    expected_metrics = [
        {
            "type": "mAP",
            "value": 0.504950495049505,
            "parameters": {
                "iou": 0.1,
                "label_key": "k1",
            },
        },
        {
            "type": "mAP",
            "value": 0.504950495049505,
            "parameters": {
                "iou": 0.6,
                "label_key": "k1",
            },
        },
        {
            "type": "mAP",
            "value": 0.0,
            "parameters": {
                "iou": 0.1,
                "label_key": "k2",
            },
        },
        {
            "type": "mAP",
            "value": 0.0,
            "parameters": {
                "iou": 0.6,
                "label_key": "k2",
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_ap_using_torch_metrics_example(
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

    metrics = evaluator.evaluate(
        iou_thresholds=[0.5, 0.75],
    )

    # test AP
    actual_metrics = [m.to_dict() for m in metrics[MetricType.AP]]
    expected_metrics = [
        {
            "type": "AP",
            "value": 1.0,
            "parameters": {
                "iou": 0.5,
                "label": {"key": "class", "value": "0"},
            },
        },
        {
            "type": "AP",
            "value": 0.7227722772277229,
            "parameters": {
                "iou": 0.75,
                "label": {"key": "class", "value": "0"},
            },
        },
        {
            "type": "AP",
            "value": 1.0,
            "parameters": {
                "iou": 0.5,
                "label": {"key": "class", "value": "1"},
            },
        },
        {
            "type": "AP",
            "value": 1.0,
            "parameters": {
                "iou": 0.75,
                "label": {"key": "class", "value": "1"},
            },
        },
        {
            "type": "AP",
            "value": 0.504950495049505,
            "parameters": {
                "iou": 0.5,
                "label": {"key": "class", "value": "2"},
            },
        },
        {
            "type": "AP",
            "value": 0.504950495049505,
            "parameters": {
                "iou": 0.75,
                "label": {"key": "class", "value": "2"},
            },
        },
        {
            "type": "AP",
            "value": 1.0,
            "parameters": {
                "iou": 0.5,
                "label": {"key": "class", "value": "4"},
            },
        },
        {
            "type": "AP",
            "value": 1.0,
            "parameters": {
                "iou": 0.75,
                "label": {"key": "class", "value": "4"},
            },
        },
        {
            "type": "AP",
            "value": 0.7909790979097909,
            "parameters": {
                "iou": 0.5,
                "label": {"key": "class", "value": "49"},
            },
        },
        {
            "type": "AP",
            "value": 0.5756718528995757,
            "parameters": {
                "iou": 0.75,
                "label": {"key": "class", "value": "49"},
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics

    # test mAP
    actual_metrics = [m.to_dict() for m in metrics[MetricType.mAP]]
    expected_metrics = [
        {
            "type": "mAP",
            "value": 0.8591859185918592,
            "parameters": {
                "iou": 0.5,
                "label_key": "class",
            },
        },
        {
            "type": "mAP",
            "value": 0.7606789250353607,
            "parameters": {
                "iou": 0.75,
                "label_key": "class",
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics
