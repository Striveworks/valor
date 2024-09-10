import numpy as np
from valor_lite.detection import DataLoader, MetricType, _compute_metrics


def test__compute_average_recall():

    sorted_pairs = np.array(
        [
            # dt,  gt,  pd,  iou,  gl,  pl, score,
            [0.0, 0.0, 2.0, 0.25, 0.0, 0.0, 0.95],
            [0.0, 1.0, 3.0, 0.33333, 0.0, 0.0, 0.9],
            [0.0, 0.0, 4.0, 0.66667, 0.0, 0.0, 0.65],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.1],
            [0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.01],
            [0.0, 2.0, 5.0, 0.5, 1.0, 1.0, 0.95],
        ]
    )

    label_counts = np.array([[2, 5, 0], [1, 1, 0]])
    iou_thresholds = np.array([0.1, 0.6])
    score_thresholds = np.array([0.5, 0.93, 0.98])

    (_, results, _, _, _, _,) = _compute_metrics(
        sorted_pairs,
        label_counts=label_counts,
        iou_thresholds=iou_thresholds,
        score_thresholds=score_thresholds,
    )

    expected = np.array(
        [
            [0.75, 0.5],
            [0.25, 0.5],
            [0.0, 0.0],
        ]
    )

    assert expected.shape == results.shape
    assert np.isclose(results, expected).all()


def test_ar_using_torch_metrics_example(
    evaluate_detection_functional_test_groundtruths,
    evaluate_detection_functional_test_predictions,
):
    """
    cf with torch metrics/pycocotools results listed here:
    https://github.com/Lightning-AI/metrics/blob/107dbfd5fb158b7ae6d76281df44bd94c836bfce/tests/unittests/detection/test_map.py#L231
    """
    manager = DataLoader()
    manager.add_data_from_valor_core(
        groundtruths=evaluate_detection_functional_test_groundtruths,
        predictions=evaluate_detection_functional_test_predictions,
    )
    evaluator = manager.finalize()

    assert evaluator.ignored_prediction_labels == [("class", "3")]
    assert evaluator.missing_prediction_labels == []
    assert evaluator.n_datums == 4
    assert evaluator.n_labels == 6
    assert evaluator.n_groundtruths == 20
    assert evaluator.n_predictions == 19

    score_thresholds = [0.0]
    iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    ar_metrics = evaluator.evaluate(
        iou_thresholds=iou_thresholds,
        score_thresholds=score_thresholds,
    )[MetricType.AR]

    expected_metrics = [
        {
            "type": "AR",
            "value": 0.45,
            "parameters": {
                "ious": iou_thresholds,
                "score_threshold": 0.0,
                "label": {"key": "class", "value": "2"},
            },
        },
        {
            "type": "AR",
            "value": 0.5800000000000001,
            "parameters": {
                "ious": iou_thresholds,
                "score_threshold": 0.0,
                "label": {"key": "class", "value": "49"},
            },
        },
        {
            "type": "AR",
            "value": 0.78,
            "parameters": {
                "ious": iou_thresholds,
                "score_threshold": 0.0,
                "label": {"key": "class", "value": "0"},
            },
        },
        {
            "type": "AR",
            "value": 0.8,
            "parameters": {
                "ious": iou_thresholds,
                "score_threshold": 0.0,
                "label": {"key": "class", "value": "1"},
            },
        },
        {
            "type": "AR",
            "value": 0.65,
            "parameters": {
                "ious": iou_thresholds,
                "score_threshold": 0.0,
                "label": {"key": "class", "value": "4"},
            },
        },
    ]
    actual_metrics = [m.to_dict() for m in ar_metrics]

    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics
