from pathlib import Path

from valor_lite.object_detection import DataLoader, Detection, MetricType


def test_ar_metrics_first_class(
    tmp_path: Path,
    basic_detections_first_class: list[Detection],
    basic_rotated_detections_first_class: list[Detection],
):
    """
    Basic object detection test.

    groundtruths
        datum uid1
            box 1 - label v1 - tp
        datum uid2
            box 2 - label v1 - fn unmatched ground truths

    predictions
        datum uid1
            box 1 - label v1 - score 0.3 - tp
        datum uid2
           none
    """
    for desc, input_, method in [
        ("bbox", basic_detections_first_class, DataLoader.add_bounding_boxes),
        (
            "poly",
            basic_rotated_detections_first_class,
            DataLoader.add_polygons,
        ),
    ]:
        loader = DataLoader.create(f"{tmp_path}_{desc}")
        method(loader, input_)
        evaluator = loader.finalize()

        metrics = evaluator.evaluate(
            iou_thresholds=[0.1, 0.6],
            score_thresholds=[0.0],
        )

        assert evaluator.metadata.number_of_datums == 2
        assert evaluator.metadata.number_of_labels == 1
        assert evaluator.metadata.number_of_ground_truths == 2
        assert evaluator.metadata.number_of_predictions == 1

        # test AR
        actual_metrics = [m.to_dict() for m in metrics[MetricType.AR]]
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
        actual_metrics = [m.to_dict() for m in metrics[MetricType.mAR]]
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

        # test AR Averaged Over IOUs
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
                    "label": "v1",
                },
            },
        ]
        for m in actual_metrics:
            assert m in expected_metrics
        for m in expected_metrics:
            assert m in actual_metrics

        # test mAR Averaged Over IOUs
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
                },
            },
        ]
        for m in actual_metrics:
            assert m in expected_metrics
        for m in expected_metrics:
            assert m in actual_metrics


def test_ar_metrics_second_class(
    tmp_path: Path,
    basic_detections_second_class: list[Detection],
    basic_rotated_detections_second_class: list[Detection],
):
    """
    Basic object detection test.

    groundtruths
        datum uid1
            box 3 - label v2 - fn unmatched ground truths
        datum uid2
           none
    predictions
        datum uid1
            none
        datum uid2
            box 2 - label v2 - score 0.98 - fp
    """
    for desc, input_, method in [
        ("bbox", basic_detections_second_class, DataLoader.add_bounding_boxes),
        (
            "poly",
            basic_rotated_detections_second_class,
            DataLoader.add_polygons,
        ),
    ]:
        loader = DataLoader.create(f"{tmp_path}_{desc}")
        method(loader, input_)
        evaluator = loader.finalize()

        metrics = evaluator.evaluate(
            iou_thresholds=[0.1, 0.6],
            score_thresholds=[0.0],
        )

        assert evaluator.metadata.number_of_datums == 2
        assert evaluator.metadata.number_of_labels == 1
        assert evaluator.metadata.number_of_ground_truths == 1
        assert evaluator.metadata.number_of_predictions == 1

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

        # test AR Averaged Over IOUs
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

        # test mAR Averaged Over IOUs
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
    tmp_path: Path,
    torchmetrics_detections: list[Detection],
):
    """
    cf with torch metrics/pycocotools results listed here:
    https://github.com/Lightning-AI/metrics/blob/107dbfd5fb158b7ae6d76281df44bd94c836bfce/tests/unittests/detection/test_map.py#L231
    """

    loader = DataLoader.create(tmp_path)
    loader.add_bounding_boxes(torchmetrics_detections)
    evaluator = loader.finalize()

    assert evaluator.metadata.number_of_datums == 4
    assert evaluator.metadata.number_of_labels == 6
    assert evaluator.metadata.number_of_ground_truths == 20
    assert evaluator.metadata.number_of_predictions == 19

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
            "value": 0.0,
            "parameters": {
                "iou_thresholds": iou_thresholds,
                "score_threshold": 0.0,
                "label": "3",
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
    actual_metrics = [m.to_dict() for m in metrics[MetricType.mAR]]
    expected_metrics = [
        {
            "type": "mAR",
            "value": 0.5433333333333333,
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
            "value": 0.0,
            "parameters": {
                "iou_thresholds": iou_thresholds,
                "score_thresholds": [0.0],
                "label": "3",
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
    actual_metrics = [
        m.to_dict() for m in metrics[MetricType.mARAveragedOverScores]
    ]
    expected_metrics = [
        {
            "type": "mARAveragedOverScores",
            "value": 0.5433333333333333,
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
    tmp_path: Path,
    detections_tp_deassignment_edge_case: list[Detection],
):

    loader = DataLoader.create(tmp_path)
    loader.add_bounding_boxes(detections_tp_deassignment_edge_case)
    evaluator = loader.finalize()

    assert evaluator.metadata.number_of_datums == 1
    assert evaluator.metadata.number_of_labels == 1
    assert evaluator.metadata.number_of_ground_truths == 2
    assert evaluator.metadata.number_of_predictions == 4

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
                "label": "v1",
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_ar_ranked_pair_ordering(
    tmp_path: Path,
    detection_ranked_pair_ordering: Detection,
    detection_ranked_pair_ordering_with_bitmasks: Detection,
    detection_ranked_pair_ordering_with_polygons: Detection,
):

    for desc, input_, method in [
        (
            "bbox",
            detection_ranked_pair_ordering,
            DataLoader.add_bounding_boxes,
        ),
        (
            "poly",
            detection_ranked_pair_ordering_with_bitmasks,
            DataLoader.add_bitmasks,
        ),
        (
            "bitmask",
            detection_ranked_pair_ordering_with_polygons,
            DataLoader.add_polygons,
        ),
    ]:
        loader = DataLoader.create(f"{tmp_path}_{desc}")
        method(loader, detections=[input_])
        evaluator = loader.finalize()

        assert evaluator.metadata.to_dict() == {
            "number_of_datums": 1,
            "number_of_ground_truths": 3,
            "number_of_labels": 4,
            "number_of_predictions": 4,
        }

        metrics = evaluator.evaluate(
            iou_thresholds=[0.5, 0.75],
            score_thresholds=[0.0],
        )

        actual_metrics = [m.to_dict() for m in metrics[MetricType.AR]]
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
            {
                "type": "AR",
                "value": 0.0,
                "parameters": {
                    "score_threshold": 0.0,
                    "iou_thresholds": [0.5, 0.75],
                    "label": "label4",
                },
            },
        ]
        for m in actual_metrics:
            assert m in expected_metrics
        for m in expected_metrics:
            assert m in actual_metrics

        actual_metrics = [m.to_dict() for m in metrics[MetricType.mAR]]
        expected_metrics = expected_metrics = [
            {
                "type": "mAR",
                "value": 0.5,
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
