import numpy as np
from valor_lite.object_detection import (
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

    (_, results, _, _,) = compute_metrics(
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

    # since only one class, ar == mar
    assert expected.flatten().shape == mean_average_recall.shape
    assert np.isclose(mean_average_recall, expected.flatten()).all()

    expected = np.array(
        [1 / 3, 1 / 3],
    )
    assert expected.shape == average_recall_averaged_over_scores.shape
    assert np.isclose(average_recall_averaged_over_scores, expected).all()

    expected = np.array(
        [1 / 3],
    )
    assert isinstance(mean_average_recall_averaged_over_scores, float)
    assert np.isclose(
        mean_average_recall_averaged_over_scores, expected.flatten()
    ).all()


def test_ar_metrics_first_class(
    basic_detections_first_class: list[Detection],
    basic_rotated_detections_first_class: list[Detection],
):
    """
    Basic object detection test.

    groundtruths
        datum uid1
            box 1 - label v1 - tp
        datum uid2
            box 2 - label v1 - fn missing prediction

    predictions
        datum uid1
            box 1 - label v1 - score 0.3 - tp
        datum uid2
           none
    """
    for input_, method in [
        (basic_detections_first_class, DataLoader.add_bounding_boxes),
        (basic_rotated_detections_first_class, DataLoader.add_polygons),
    ]:
        loader = DataLoader()
        method(loader, input_)
        evaluator = loader.finalize()

        metrics = evaluator.evaluate(
            iou_thresholds=[0.1, 0.6],
            score_thresholds=[0.0],
            as_dict=True,
        )

        assert evaluator.ignored_prediction_labels == []
        assert evaluator.missing_prediction_labels == []
        assert evaluator.n_datums == 2
        assert evaluator.n_labels == 1
        assert evaluator.n_groundtruths == 2
        assert evaluator.n_predictions == 1

        # test AR
        actual_metrics = [m for m in metrics[MetricType.AR]]
        expected_metrics = [
            {
                "type": "AR",
                "value": 0.5,
                "parameters": {
                    "score_threshold": 0.0,
                    "iou_thresholds": [0.1, 0.6],
                    "label": "v1",
                },
            },
        ]
        for m in actual_metrics:
            assert m in expected_metrics
        for m in expected_metrics:
            assert m in actual_metrics

        # test mAR
        actual_metrics = [m for m in metrics[MetricType.mAR]]
        expected_metrics = [
            {
                "type": "mAR",
                "value": 0.5,
                "parameters": {
                    "score_threshold": 0.0,
                    "iou_thresholds": [0.1, 0.6],
                },
            },
        ]
        for m in actual_metrics:
            assert m in expected_metrics
        for m in expected_metrics:
            assert m in actual_metrics

        # test AR Averaged Over IoUs
        actual_metrics = [m for m in metrics[MetricType.ARAveragedOverScores]]
        expected_metrics = [
            {
                "type": "ARAveragedOverScores",
                "value": 0.5,
                "parameters": {
                    "score_thresholds": [0.0],
                    "iou_thresholds": [0.1, 0.6],
                    "label": "v1",
                },
            },
        ]
        for m in actual_metrics:
            assert m in expected_metrics
        for m in expected_metrics:
            assert m in actual_metrics

        # test mAR Averaged Over IoUs
        actual_metrics = [m for m in metrics[MetricType.mARAveragedOverScores]]
        expected_metrics = [
            {
                "type": "mARAveragedOverScores",
                "value": 0.5,
                "parameters": {
                    "score_thresholds": [0.0],
                    "iou_thresholds": [0.1, 0.6],
                },
            },
        ]
        for m in actual_metrics:
            assert m in expected_metrics
        for m in expected_metrics:
            assert m in actual_metrics


def test_ar_metrics_second_class(
    basic_detections_second_class: list[Detection],
    basic_rotated_detections_second_class: list[Detection],
):
    """
    Basic object detection test.

    groundtruths
        datum uid1
            box 3 - label v2 - fn missing prediction
        datum uid2
           none
    predictions
        datum uid1
            none
        datum uid2
            box 2 - label v2 - score 0.98 - fp
    """
    for input_, method in [
        (basic_detections_second_class, DataLoader.add_bounding_boxes),
        (basic_rotated_detections_second_class, DataLoader.add_polygons),
    ]:
        loader = DataLoader()
        method(loader, input_)
        evaluator = loader.finalize()

        metrics = evaluator.evaluate(
            iou_thresholds=[0.1, 0.6],
            score_thresholds=[0.0],
        )

        assert evaluator.ignored_prediction_labels == []
        assert evaluator.missing_prediction_labels == []
        assert evaluator.n_datums == 2
        assert evaluator.n_labels == 1
        assert evaluator.n_groundtruths == 1
        assert evaluator.n_predictions == 1

        # test AR
        actual_metrics = [m.to_dict() for m in metrics[MetricType.AR]]
        expected_metrics = [
            {
                "type": "AR",
                "value": 0.0,
                "parameters": {
                    "score_threshold": 0.0,
                    "iou_thresholds": [0.1, 0.6],
                    "label": "v2",
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
                "value": 0.0,
                "parameters": {
                    "score_threshold": 0.0,
                    "iou_thresholds": [0.1, 0.6],
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
                "value": 0.0,
                "parameters": {
                    "score_thresholds": [0.0],
                    "iou_thresholds": [0.1, 0.6],
                    "label": "v2",
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
                "value": 0.0,
                "parameters": {
                    "score_thresholds": [0.0],
                    "iou_thresholds": [0.1, 0.6],
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
    loader.add_bounding_boxes(torchmetrics_detections)
    evaluator = loader.finalize()

    assert evaluator.ignored_prediction_labels == ["3"]
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
        as_dict=True,
    )

    # test AR
    actual_metrics = [m for m in metrics[MetricType.AR]]
    expected_metrics = [
        {
            "type": "AR",
            "value": 0.45,
            "parameters": {
                "iou_thresholds": iou_thresholds,
                "score_threshold": 0.0,
                "label": "2",
            },
        },
        {
            "type": "AR",
            "value": 0.5800000000000001,
            "parameters": {
                "iou_thresholds": iou_thresholds,
                "score_threshold": 0.0,
                "label": "49",
            },
        },
        {
            "type": "AR",
            "value": 0.78,
            "parameters": {
                "iou_thresholds": iou_thresholds,
                "score_threshold": 0.0,
                "label": "0",
            },
        },
        {
            "type": "AR",
            "value": 0.8,
            "parameters": {
                "iou_thresholds": iou_thresholds,
                "score_threshold": 0.0,
                "label": "1",
            },
        },
        {
            "type": "AR",
            "value": 0.65,
            "parameters": {
                "iou_thresholds": iou_thresholds,
                "score_threshold": 0.0,
                "label": "4",
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics

    # test mAR
    actual_metrics = [m for m in metrics[MetricType.mAR]]
    expected_metrics = [
        {
            "type": "mAR",
            "value": 0.652,
            "parameters": {
                "iou_thresholds": iou_thresholds,
                "score_threshold": 0.0,
            },
        }
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics

    # test ARAveragedOverScores
    actual_metrics = [m for m in metrics[MetricType.ARAveragedOverScores]]
    expected_metrics = [
        {
            "type": "ARAveragedOverScores",
            "value": 0.45,
            "parameters": {
                "iou_thresholds": iou_thresholds,
                "score_thresholds": [0.0],
                "label": "2",
            },
        },
        {
            "type": "ARAveragedOverScores",
            "value": 0.5800000000000001,
            "parameters": {
                "iou_thresholds": iou_thresholds,
                "score_thresholds": [0.0],
                "label": "49",
            },
        },
        {
            "type": "ARAveragedOverScores",
            "value": 0.78,
            "parameters": {
                "iou_thresholds": iou_thresholds,
                "score_thresholds": [0.0],
                "label": "0",
            },
        },
        {
            "type": "ARAveragedOverScores",
            "value": 0.8,
            "parameters": {
                "iou_thresholds": iou_thresholds,
                "score_thresholds": [0.0],
                "label": "1",
            },
        },
        {
            "type": "ARAveragedOverScores",
            "value": 0.65,
            "parameters": {
                "iou_thresholds": iou_thresholds,
                "score_thresholds": [0.0],
                "label": "4",
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics

    # test mARAveragedOverScores
    actual_metrics = [m for m in metrics[MetricType.mARAveragedOverScores]]
    expected_metrics = [
        {
            "type": "mARAveragedOverScores",
            "value": 0.652,
            "parameters": {
                "iou_thresholds": iou_thresholds,
                "score_thresholds": [0.0],
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
    loader.add_bounding_boxes(detections_tp_deassignment_edge_case)
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
        as_dict=True,
    )

    assert len(metrics) == 15

    # test AR
    actual_metrics = [m for m in metrics[MetricType.AR]]
    expected_metrics = [
        {
            "type": "AR",
            "value": 0.5,
            "parameters": {
                "score_threshold": 0.5,
                "iou_thresholds": [0.5],
                "label": "v1",
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_ar_ranked_pair_ordering(
    detection_ranked_pair_ordering: Detection,
    detection_ranked_pair_ordering_with_bitmasks: Detection,
    detection_ranked_pair_ordering_with_polygons: Detection,
):

    for input_, method in [
        (detection_ranked_pair_ordering, DataLoader.add_bounding_boxes),
        (
            detection_ranked_pair_ordering_with_bitmasks,
            DataLoader.add_bitmasks,
        ),
        (
            detection_ranked_pair_ordering_with_polygons,
            DataLoader.add_polygons,
        ),
    ]:
        loader = DataLoader()
        method(loader, detections=[input_])
        evaluator = loader.finalize()

        assert evaluator.metadata == {
            "ignored_prediction_labels": ["label4"],
            "missing_prediction_labels": [],
            "n_datums": 1,
            "n_groundtruths": 3,
            "n_labels": 4,
            "n_predictions": 4,
        }

        metrics = evaluator.evaluate(
            iou_thresholds=[0.5, 0.75],
            score_thresholds=[0.0],
            as_dict=True,
        )

        actual_metrics = [m for m in metrics[MetricType.AR]]
        expected_metrics = expected_metrics = [
            {
                "type": "AR",
                "value": 1.0,
                "parameters": {
                    "score_threshold": 0.0,
                    "iou_thresholds": [0.5, 0.75],
                    "label": "label1",
                },
            },
            {
                "type": "AR",
                "value": 1.0,
                "parameters": {
                    "score_threshold": 0.0,
                    "iou_thresholds": [0.5, 0.75],
                    "label": "label2",
                },
            },
            {
                "type": "AR",
                "value": 0.0,
                "parameters": {
                    "score_threshold": 0.0,
                    "iou_thresholds": [0.5, 0.75],
                    "label": "label3",
                },
            },
        ]
        for m in actual_metrics:
            assert m in expected_metrics
        for m in expected_metrics:
            assert m in actual_metrics

        actual_metrics = [m for m in metrics[MetricType.mAR]]
        expected_metrics = expected_metrics = [
            {
                "type": "mAR",
                "value": 0.6666666666666666,
                "parameters": {
                    "score_threshold": 0.0,
                    "iou_thresholds": [0.5, 0.75],
                },
            },
        ]
        for m in actual_metrics:
            assert m in expected_metrics
        for m in expected_metrics:
            assert m in actual_metrics
