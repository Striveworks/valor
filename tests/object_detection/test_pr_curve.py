from valor_lite.object_detection import Evaluator, MetricType


def test_pr_curve_using_torch_metrics_example(
    torchmetrics_detections: Evaluator
):
    """
    cf with torch metrics/pycocotools results listed here:
    https://github.com/Lightning-AI/metrics/blob/107dbfd5fb158b7ae6d76281df44bd94c836bfce/tests/unittests/detection/test_map.py#L231
    """
    evaluator = torchmetrics_detections
    assert evaluator.info.number_of_datums == 4
    assert evaluator.info.number_of_labels == 6
    assert evaluator.info.number_of_groundtruth_annotations == 20
    assert evaluator.info.number_of_prediction_annotations == 19

    metrics = evaluator.compute_precision_recall(
        iou_thresholds=[0.5, 0.75],
        score_thresholds=[0.5],
    )

    # test PrecisionRecallCurve
    actual_metrics = [
        m.to_dict() for m in metrics[MetricType.PrecisionRecallCurve]
    ]
    expected_metrics = [
        {
            "type": "PrecisionRecallCurve",
            "value": {
                "precisions": [1.0 for _ in range(101)],
                "scores": (
                    [0.953 for _ in range(21)]
                    + [0.805 for _ in range(20)]
                    + [0.611 for _ in range(20)]
                    + [0.407 for _ in range(20)]
                    + [0.335 for _ in range(20)]
                ),
            },
            "parameters": {
                "iou_threshold": 0.5,
                "label": "0",
            },
        },
        {
            "type": "PrecisionRecallCurve",
            "value": {
                "precisions": (
                    [1.0 for _ in range(41)]
                    + [0.8 for _ in range(40)]
                    + [0.0 for _ in range(20)]
                ),
                "scores": (
                    [0.953 for _ in range(21)]
                    + [0.805 for _ in range(20)]
                    + [0.407 for _ in range(20)]
                    + [0.335 for _ in range(20)]
                    + [0.0 for _ in range(20)]
                ),
            },
            "parameters": {
                "iou_threshold": 0.75,
                "label": "0",
            },
        },
        {
            "type": "PrecisionRecallCurve",
            "value": {
                "precisions": [1.0 for _ in range(101)],
                "scores": [0.3 for _ in range(101)],
            },
            "parameters": {
                "iou_threshold": 0.5,
                "label": "1",
            },
        },
        {
            "type": "PrecisionRecallCurve",
            "value": {
                "precisions": [1.0 for _ in range(101)],
                "scores": [0.3 for _ in range(101)],
            },
            "parameters": {
                "iou_threshold": 0.75,
                "label": "1",
            },
        },
        {
            "type": "PrecisionRecallCurve",
            "value": {
                "precisions": [1.0 for _ in range(51)]
                + [0.0 for _ in range(50)],
                "scores": [0.726 for _ in range(51)]
                + [0.0 for _ in range(50)],
            },
            "parameters": {
                "iou_threshold": 0.5,
                "label": "2",
            },
        },
        {
            "type": "PrecisionRecallCurve",
            "value": {
                "precisions": [1.0 for _ in range(51)]
                + [0.0 for _ in range(50)],
                "scores": [0.726 for _ in range(51)]
                + [0.0 for _ in range(50)],
            },
            "parameters": {
                "iou_threshold": 0.75,
                "label": "2",
            },
        },
        {
            "type": "PrecisionRecallCurve",
            "value": {
                "precisions": [0.0 for _ in range(101)],
                "scores": [0.318] + [0.0 for _ in range(100)],
            },
            "parameters": {
                "iou_threshold": 0.5,
                "label": "3",
            },
        },
        {
            "type": "PrecisionRecallCurve",
            "value": {
                "precisions": [0.0 for _ in range(101)],
                "scores": [0.318] + [0.0 for _ in range(100)],
            },
            "parameters": {
                "iou_threshold": 0.75,
                "label": "3",
            },
        },
        {
            "type": "PrecisionRecallCurve",
            "value": {
                "precisions": [1.0 for _ in range(101)],
                "scores": [0.546 for _ in range(51)]
                + [0.236 for _ in range(50)],
            },
            "parameters": {
                "iou_threshold": 0.5,
                "label": "4",
            },
        },
        {
            "type": "PrecisionRecallCurve",
            "value": {
                "precisions": [1.0 for _ in range(101)],
                "scores": [0.546 for _ in range(51)]
                + [0.236 for _ in range(50)],
            },
            "parameters": {
                "iou_threshold": 0.75,
                "label": "4",
            },
        },
        {
            "type": "PrecisionRecallCurve",
            "value": {
                "precisions": (
                    [1.0 for _ in range(71)]
                    + [8 / 9 for _ in range(10)]
                    + [0.0 for _ in range(20)]
                ),
                "scores": (
                    [0.883 for _ in range(11)]
                    + [0.782 for _ in range(10)]
                    + [0.561 for _ in range(10)]
                    + [0.532 for _ in range(10)]
                    + [0.349 for _ in range(10)]
                    + [0.271 for _ in range(10)]
                    + [0.204 for _ in range(10)]
                    + [0.202 for _ in range(10)]
                    + [0.0 for _ in range(20)]
                ),
            },
            "parameters": {
                "iou_threshold": 0.5,
                "label": "49",
            },
        },
        {
            "type": "PrecisionRecallCurve",
            "value": {
                "precisions": (
                    [1.0 for _ in range(41)]
                    + [0.8571428571428571 for _ in range(20)]
                    + [0.0 for _ in range(40)]
                ),
                "scores": (
                    [0.883 for _ in range(11)]
                    + [0.782 for _ in range(10)]
                    + [0.561 for _ in range(10)]
                    + [0.532 for _ in range(10)]
                    + [0.271 for _ in range(10)]
                    + [0.204 for _ in range(10)]
                    + [0.0 for _ in range(40)]
                ),
            },
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
