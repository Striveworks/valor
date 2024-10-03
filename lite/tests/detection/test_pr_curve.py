import numpy as np
from valor_lite.detection import (
    DataLoader,
    Detection,
    MetricType,
    compute_metrics,
)


def test_pr_curve_simple():

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

    label_metadata = np.array([[1, 5, 0]])
    iou_thresholds = np.array([0.1, 0.6])
    score_thresholds = np.array([0.0])

    (_, _, _, pr_curve) = compute_metrics(
        sorted_pairs,
        label_metadata=label_metadata,
        iou_thresholds=iou_thresholds,
        score_thresholds=score_thresholds,
    )

    assert pr_curve.shape == (2, 1, 101)
    assert np.isclose(pr_curve[0][0], 1.0).all()
    assert np.isclose(pr_curve[1][0], 1 / 3).all()


def test_pr_curve_using_torch_metrics_example(
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

    metrics = evaluator.evaluate(
        iou_thresholds=[0.5, 0.75],
        as_dict=True,
    )

    # AP = 1.0
    a = [1.0 for _ in range(101)]

    # AP = 0.505
    b = [1.0 for _ in range(51)] + [0.0 for _ in range(50)]

    # AP = 0.791
    c = (
        [1.0 for _ in range(71)]
        + [8 / 9 for _ in range(10)]
        + [0.0 for _ in range(20)]
    )

    # AP = 0.722
    d = (
        [1.0 for _ in range(41)]
        + [0.8 for _ in range(40)]
        + [0.0 for _ in range(20)]
    )

    # AP = 0.576
    e = (
        [1.0 for _ in range(41)]
        + [0.8571428571428571 for _ in range(20)]
        + [0.0 for _ in range(40)]
    )

    # test PrecisionRecallCurve
    actual_metrics = [m for m in metrics[MetricType.PrecisionRecallCurve]]
    expected_metrics = [
        {
            "type": "PrecisionRecallCurve",
            "value": a,
            "parameters": {
                "iou_threshold": 0.5,
                "label": "0",
            },
        },
        {
            "type": "PrecisionRecallCurve",
            "value": d,
            "parameters": {
                "iou_threshold": 0.75,
                "label": "0",
            },
        },
        {
            "type": "PrecisionRecallCurve",
            "value": a,
            "parameters": {
                "iou_threshold": 0.5,
                "label": "1",
            },
        },
        {
            "type": "PrecisionRecallCurve",
            "value": a,
            "parameters": {
                "iou_threshold": 0.75,
                "label": "1",
            },
        },
        {
            "type": "PrecisionRecallCurve",
            "value": b,
            "parameters": {
                "iou_threshold": 0.5,
                "label": "2",
            },
        },
        {
            "type": "PrecisionRecallCurve",
            "value": b,
            "parameters": {
                "iou_threshold": 0.75,
                "label": "2",
            },
        },
        {
            "type": "PrecisionRecallCurve",
            "value": a,
            "parameters": {
                "iou_threshold": 0.5,
                "label": "4",
            },
        },
        {
            "type": "PrecisionRecallCurve",
            "value": a,
            "parameters": {
                "iou_threshold": 0.75,
                "label": "4",
            },
        },
        {
            "type": "PrecisionRecallCurve",
            "value": c,
            "parameters": {
                "iou_threshold": 0.5,
                "label": "49",
            },
        },
        {
            "type": "PrecisionRecallCurve",
            "value": e,
            "parameters": {
                "iou_threshold": 0.75,
                "label": "49",
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics
