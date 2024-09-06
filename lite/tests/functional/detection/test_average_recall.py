import numpy as np
from valor_lite import schemas
from valor_lite.detection import Manager, MetricType, _compute_average_recall


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

    label_counts = np.array([[2, 5], [1, 1]])
    iou_thresholds = np.array([0.1, 0.6])
    score_thresholds = np.array([0.5, 0.93, 0.98])

    results = _compute_average_recall(
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
    manager = Manager()
    manager.add_data(
        groundtruths=evaluate_detection_functional_test_groundtruths,
        predictions=evaluate_detection_functional_test_predictions,
    )
    manager.finalize()

    assert manager.ignored_prediction_labels == [
        schemas.Label(key="class", value="3")
    ]
    assert manager.missing_prediction_labels == []
    assert manager.n_datums == 4
    assert manager.n_labels == 6
    assert manager.n_groundtruths == 20
    assert manager.n_predictions == 19

    score_thresholds = [0.0]
    iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    ar_metrics = manager.evaluate(
        iou_thresholds=iou_thresholds,
        score_thresholds=score_thresholds,
    )[MetricType.AR]

    expected_metrics = [
        {
            "type": "AR",
            "ious": iou_thresholds,
            "values": {"0.0": 0.45},
            "label": {"key": "class", "value": "2"},
        },
        {
            "type": "AR",
            "ious": iou_thresholds,
            "values": {"0.0": 0.5800000000000001},
            "label": {"key": "class", "value": "49"},
        },
        {
            "type": "AR",
            "ious": iou_thresholds,
            "values": {"0.0": 0.78},
            "label": {"key": "class", "value": "0"},
        },
        {
            "type": "AR",
            "ious": iou_thresholds,
            "values": {"0.0": 0.8},
            "label": {"key": "class", "value": "1"},
        },
        {
            "type": "AR",
            "ious": iou_thresholds,
            "values": {"0.0": 0.65},
            "label": {"key": "class", "value": "4"},
        },
    ]
    actual_metrics = [m.to_dict() for m in ar_metrics]

    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics
