import numpy as np
from valor_lite.detection import (
    DataLoader,
    Detection,
    MetricType,
    compute_metrics,
)


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

    label_metadata = np.array([[2, 5, 0], [1, 1, 0]])
    iou_thresholds = np.array([0.1, 0.6])
    score_thresholds = np.array([0.5, 0.93, 0.98])

    (
        _,
        results,
        _,
        _,
    ) = compute_metrics(
        sorted_pairs,
        label_metadata=label_metadata,
        iou_thresholds=iou_thresholds,
        score_thresholds=score_thresholds,
    )
    (
        average_recall,
        mean_average_recall,
        average_recall_averaged_over_scores,
        mean_average_recall_averaged_over_scores,
    ) = results

    expected = np.array(
        [
            [0.75, 0.5],
            [0.25, 0.5],
            [0.0, 0.0],
        ]
    )
    assert expected.shape == average_recall.shape
    assert np.isclose(average_recall, expected).all()

    expected = np.array(
        [
            [(0.75 + 0.5) / 2.0],
            [(0.25 + 0.5) / 2.0],
            [0.0],
        ]
    )
    assert expected.shape == mean_average_recall.shape
    assert np.isclose(mean_average_recall, expected).all()

    expected = np.array(
        [1 / 3, 1 / 3],
    )
    assert expected.shape == average_recall_averaged_over_scores.shape
    assert np.isclose(average_recall_averaged_over_scores, expected).all()

    expected = np.array(
        [1 / 3],
    )
    assert expected.shape == mean_average_recall_averaged_over_scores.shape
    assert np.isclose(mean_average_recall_averaged_over_scores, expected).all()


def test_ar_metrics(
    basic_detections: list[Detection],
    basic_rotated_detections: list[Detection],
):
    """
    Basic object detection test, testing both axis-aligned and rotated bounding boxes.

    groundtruths
        datum uid1
            box 1 - label (k1, v1) - tp
            box 3 - label (k2, v2) - fn missing prediction
        datum uid2
            box 2 - label (k1, v1) - fn misclassification

    predictions
        datum uid1
            box 1 - label (k1, v1) - score 0.3 - tp
        datum uid2
            box 2 - label (k2, v2) - score 0.98 - fp
    """
    for input_ in [basic_detections, basic_rotated_detections]:
        loader = DataLoader()
        loader.add_data(input_)
        evaluator = loader.finalize()

        metrics = evaluator.evaluate(
            iou_thresholds=[0.1, 0.6],
            score_thresholds=[0.0],
        )

        assert evaluator.ignored_prediction_labels == []
        assert evaluator.missing_prediction_labels == []
        assert evaluator.n_datums == 2
        assert evaluator.n_labels == 2
        assert evaluator.n_groundtruths == 3
        assert evaluator.n_predictions == 2

        # test AR
        actual_metrics = [m.to_dict() for m in metrics[MetricType.AR]]
        expected_metrics = [
            {
                "type": "AR",
                "value": 0.5,
                "parameters": {
                    "score_threshold": 0.0,
                    "iou_thresholds": [0.1, 0.6],
                    "label": {"key": "k1", "value": "v1"},
                },
            },
            {
                "type": "AR",
                "value": 0.0,
                "parameters": {
                    "score_threshold": 0.0,
                    "iou_thresholds": [0.1, 0.6],
                    "label": {"key": "k2", "value": "v2"},
                },
            },
        ]
        for m in actual_metrics:
            assert m in expected_metrics
        for m in expected_metrics:
            assert m in actual_metrics

        # test mAR
        actual_metrics = [m.to_dict() for m in metrics[MetricType.mAR]]
        expected_metrics = [
            {
                "type": "mAR",
                "value": 0.5,
                "parameters": {
                    "score_threshold": 0.0,
                    "iou_thresholds": [0.1, 0.6],
                    "label_key": "k1",
                },
            },
            {
                "type": "mAR",
                "value": 0.0,
                "parameters": {
                    "score_threshold": 0.0,
                    "iou_thresholds": [0.1, 0.6],
                    "label_key": "k2",
                },
            },
        ]
        for m in actual_metrics:
            assert m in expected_metrics
        for m in expected_metrics:
            assert m in actual_metrics

        # test AR Averaged Over IoUs
        actual_metrics = [
            m.to_dict() for m in metrics[MetricType.ARAveragedOverScores]
        ]
        expected_metrics = [
            {
                "type": "ARAveragedOverScores",
                "value": 0.5,
                "parameters": {
                    "score_thresholds": [0.0],
                    "iou_thresholds": [0.1, 0.6],
                    "label": {"key": "k1", "value": "v1"},
                },
            },
            {
                "type": "ARAveragedOverScores",
                "value": 0.0,
                "parameters": {
                    "score_thresholds": [0.0],
                    "iou_thresholds": [0.1, 0.6],
                    "label": {"key": "k2", "value": "v2"},
                },
            },
        ]
        for m in actual_metrics:
            assert m in expected_metrics
        for m in expected_metrics:
            assert m in actual_metrics

        # test mAR Averaged Over IoUs
        actual_metrics = [
            m.to_dict() for m in metrics[MetricType.mARAveragedOverScores]
        ]
        expected_metrics = [
            {
                "type": "mARAveragedOverScores",
                "value": 0.5,
                "parameters": {
                    "score_thresholds": [0.0],
                    "iou_thresholds": [0.1, 0.6],
                    "label_key": "k1",
                },
            },
            {
                "type": "mARAveragedOverScores",
                "value": 0.0,
                "parameters": {
                    "score_thresholds": [0.0],
                    "iou_thresholds": [0.1, 0.6],
                    "label_key": "k2",
                },
            },
        ]
        for m in actual_metrics:
            assert m in expected_metrics
        for m in expected_metrics:
            assert m in actual_metrics


def test_ar_using_torch_metrics_example(
    torchmetrics_detections: list[Detection],
):
    """
    cf with torch metrics/pycocotools results listed here:
    https://github.com/Lightning-AI/metrics/blob/107dbfd5fb158b7ae6d76281df44bd94c836bfce/tests/unittests/detection/test_map.py#L231
    """
    loader = DataLoader()
    loader.add_data(torchmetrics_detections)
    evaluator = loader.finalize()

    assert evaluator.ignored_prediction_labels == [("class", "3")]
    assert evaluator.missing_prediction_labels == []
    assert evaluator.n_datums == 4
    assert evaluator.n_labels == 6
    assert evaluator.n_groundtruths == 20
    assert evaluator.n_predictions == 19

    score_thresholds = [0.0]
    iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    metrics = evaluator.evaluate(
        iou_thresholds=iou_thresholds,
        score_thresholds=score_thresholds,
    )

    # test AR
    actual_metrics = [m.to_dict() for m in metrics[MetricType.AR]]
    expected_metrics = [
        {
            "type": "AR",
            "value": 0.45,
            "parameters": {
                "iou_thresholds": iou_thresholds,
                "score_threshold": 0.0,
                "label": {"key": "class", "value": "2"},
            },
        },
        {
            "type": "AR",
            "value": 0.5800000000000001,
            "parameters": {
                "iou_thresholds": iou_thresholds,
                "score_threshold": 0.0,
                "label": {"key": "class", "value": "49"},
            },
        },
        {
            "type": "AR",
            "value": 0.78,
            "parameters": {
                "iou_thresholds": iou_thresholds,
                "score_threshold": 0.0,
                "label": {"key": "class", "value": "0"},
            },
        },
        {
            "type": "AR",
            "value": 0.8,
            "parameters": {
                "iou_thresholds": iou_thresholds,
                "score_threshold": 0.0,
                "label": {"key": "class", "value": "1"},
            },
        },
        {
            "type": "AR",
            "value": 0.65,
            "parameters": {
                "iou_thresholds": iou_thresholds,
                "score_threshold": 0.0,
                "label": {"key": "class", "value": "4"},
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics

    # test mAR
    actual_metrics = [m.to_dict() for m in metrics[MetricType.mAR]]
    expected_metrics = [
        {
            "type": "mAR",
            "value": 0.652,
            "parameters": {
                "iou_thresholds": iou_thresholds,
                "score_threshold": 0.0,
                "label_key": "class",
            },
        }
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics

    # test ARAveragedOverScores
    actual_metrics = [
        m.to_dict() for m in metrics[MetricType.ARAveragedOverScores]
    ]
    expected_metrics = [
        {
            "type": "ARAveragedOverScores",
            "value": 0.45,
            "parameters": {
                "iou_thresholds": iou_thresholds,
                "score_thresholds": [0.0],
                "label": {"key": "class", "value": "2"},
            },
        },
        {
            "type": "ARAveragedOverScores",
            "value": 0.5800000000000001,
            "parameters": {
                "iou_thresholds": iou_thresholds,
                "score_thresholds": [0.0],
                "label": {"key": "class", "value": "49"},
            },
        },
        {
            "type": "ARAveragedOverScores",
            "value": 0.78,
            "parameters": {
                "iou_thresholds": iou_thresholds,
                "score_thresholds": [0.0],
                "label": {"key": "class", "value": "0"},
            },
        },
        {
            "type": "ARAveragedOverScores",
            "value": 0.8,
            "parameters": {
                "iou_thresholds": iou_thresholds,
                "score_thresholds": [0.0],
                "label": {"key": "class", "value": "1"},
            },
        },
        {
            "type": "ARAveragedOverScores",
            "value": 0.65,
            "parameters": {
                "iou_thresholds": iou_thresholds,
                "score_thresholds": [0.0],
                "label": {"key": "class", "value": "4"},
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics

    # test mARAveragedOverScores
    actual_metrics = [
        m.to_dict() for m in metrics[MetricType.mARAveragedOverScores]
    ]
    expected_metrics = [
        {
            "type": "mARAveragedOverScores",
            "value": 0.652,
            "parameters": {
                "iou_thresholds": iou_thresholds,
                "score_thresholds": [0.0],
                "label_key": "class",
            },
        }
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_ar_true_positive_deassignment(
    detections_tp_deassignment_edge_case: list[Detection],
):

    loader = DataLoader()
    loader.add_data(detections_tp_deassignment_edge_case)
    evaluator = loader.finalize()

    assert evaluator.ignored_prediction_labels == []
    assert evaluator.missing_prediction_labels == []
    assert evaluator.n_datums == 1
    assert evaluator.n_labels == 1
    assert evaluator.n_groundtruths == 2
    assert evaluator.n_predictions == 4

    metrics = evaluator.evaluate(
        iou_thresholds=[0.5],
        score_thresholds=[0.5],
    )

    assert len(metrics) == 14

    # test AR
    actual_metrics = [m.to_dict() for m in metrics[MetricType.AR]]
    expected_metrics = [
        {
            "type": "AR",
            "value": 0.5,
            "parameters": {
                "score_threshold": 0.5,
                "iou_thresholds": [0.5],
                "label": {"key": "k1", "value": "v1"},
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


# TODO update this to just use a for loop in one test for both sets of pairings
def test_ar_ranked_pair_ordering(detection_ranked_pair_ordering: Detection):

    loader = DataLoader()
    loader.add_data(detections=[detection_ranked_pair_ordering])
    evaluator = loader.finalize()

    metrics = evaluator.evaluate(iou_thresholds=[0.5, 0.75])

    actual_metrics = [m.to_dict() for m in metrics[MetricType.AR]]
    expected_metrics = expected_metrics = [
        {
            "label": {"key": "class", "value": "label1"},
            "parameters": {
                "ious": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            },
            "value": 1.0,
            "type": "AR",
        },
        {
            "label": {"key": "class", "value": "label4"},
            "parameters": {
                "ious": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            },
            "value": -1.0,
            "type": "AR",
        },
        {
            "label": {"key": "class", "value": "label2"},
            "parameters": {
                "ious": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            },
            "value": 1.0,
            "type": "AR",
        },
        {
            "label": {"key": "class", "value": "label3"},
            "parameters": {
                "ious": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            },
            "value": 0.0,
            "type": "AR",
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics

    actual_metrics = [m.to_dict() for m in metrics[MetricType.mAR]]
    expected_metrics = expected_metrics = [
        {
            "parameters": {
                "label_key": "class",
                "ious": [
                    0.5,
                    0.55,
                    0.6,
                    0.65,
                    0.7,
                    0.75,
                    0.8,
                    0.85,
                    0.9,
                    0.95,
                ],
            },
            "value": 0.667,
            "type": "mAR",
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_ar_ranked_pair_ordering_with_bitmasks(
    detection_ranked_pair_ordering_with_bitmasks: Detection,
):

    loader = DataLoader()
    loader.add_data(detections=[detection_ranked_pair_ordering_with_bitmasks])
    evaluator = loader.finalize()

    metrics = evaluator.evaluate(iou_thresholds=[0.5, 0.75])

    actual_metrics = [m.to_dict() for m in metrics[MetricType.AR]]
    expected_metrics = [
        {
            "label": {"key": "class", "value": "label1"},
            "parameters": {
                "ious": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            },
            "value": 1.0,
            "type": "AR",
        },
        {
            "label": {"key": "class", "value": "label4"},
            "parameters": {
                "ious": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            },
            "value": -1.0,
            "type": "AR",
        },
        {
            "label": {"key": "class", "value": "label2"},
            "parameters": {
                "ious": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            },
            "value": 1.0,
            "type": "AR",
        },
        {
            "label": {"key": "class", "value": "label3"},
            "parameters": {
                "ious": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            },
            "value": 0.0,
            "type": "AR",
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics

    actual_metrics = [m.to_dict() for m in metrics[MetricType.mAR]]
    expected_metrics = [
        {
            "parameters": {
                "label_key": "class",
                "ious": [
                    0.5,
                    0.55,
                    0.6,
                    0.65,
                    0.7,
                    0.75,
                    0.8,
                    0.85,
                    0.9,
                    0.95,
                ],
            },
            "value": 0.667,
            "type": "mAR",
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics
