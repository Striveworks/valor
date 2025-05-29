import numpy as np
import pytest

from valor_lite.object_detection import (
    DataLoader,
    Detection,
    Metric,
    MetricType,
)


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
    assert evaluator.metadata.number_of_datums == 4
    assert evaluator.metadata.number_of_labels == 6
    assert evaluator.metadata.number_of_ground_truths == 20
    assert evaluator.metadata.number_of_predictions == 19

    assert evaluator.ignored_prediction_labels == ["3"]
    assert evaluator.missing_prediction_labels == []
    assert evaluator.metadata.to_dict() == {
        "number_of_datums": 4,
        "number_of_labels": 6,
        "number_of_ground_truths": 20,
        "number_of_predictions": 19,
    }


def test_no_thresholds(detection_ranked_pair_ordering):
    loader = DataLoader()
    loader.add_bounding_boxes([detection_ranked_pair_ordering])
    evaluator = loader.finalize()

    evaluator.evaluate(
        iou_thresholds=[0.5],
        score_thresholds=[0.5],
    )

    # test compute_precision_recall
    with pytest.raises(ValueError):
        evaluator.compute_precision_recall(
            iou_thresholds=[],
            score_thresholds=[0.5],
        )
    with pytest.raises(ValueError):
        evaluator.compute_precision_recall(
            iou_thresholds=[0.5],
            score_thresholds=[],
        )

    # test compute_confusion_matrix
    with pytest.raises(ValueError):
        evaluator.compute_confusion_matrix(
            iou_thresholds=[0.5],
            score_thresholds=[],
        )
    with pytest.raises(ValueError):
        evaluator.compute_confusion_matrix(
            iou_thresholds=[],
            score_thresholds=[0.5],
        )


def test_no_groundtruths(detections_no_groundtruths):

    loader = DataLoader()
    loader.add_bounding_boxes(detections_no_groundtruths)
    evaluator = loader.finalize()

    assert evaluator.ignored_prediction_labels == ["v1"]
    assert evaluator.missing_prediction_labels == []
    assert evaluator.metadata.number_of_datums == 2
    assert evaluator.metadata.number_of_labels == 1
    assert evaluator.metadata.number_of_ground_truths == 0
    assert evaluator.metadata.number_of_predictions == 2

    with pytest.warns(UserWarning):
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
    assert evaluator.metadata.number_of_datums == 2
    assert evaluator.metadata.number_of_labels == 1
    assert evaluator.metadata.number_of_ground_truths == 2
    assert evaluator.metadata.number_of_predictions == 0

    with pytest.warns(UserWarning):
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


def _flatten_metrics(m) -> list:
    if isinstance(m, dict):
        keys = list(m.keys())
        values = [
            inner_value
            for value in m.values()
            for inner_value in _flatten_metrics(value)
        ]
        return keys + values
    elif isinstance(m, list):
        return [
            inner_value
            for value in m
            for inner_value in _flatten_metrics(value)
        ]
    elif isinstance(m, Metric):
        return _flatten_metrics(m.to_dict())
    else:
        return [m]


def test_output_types_dont_contain_numpy(basic_detections: list[Detection]):
    manager = DataLoader()
    manager.add_bounding_boxes(basic_detections)
    evaluator = manager.finalize()

    metrics = evaluator.evaluate(
        score_thresholds=[0.25, 0.75],
    )

    values = _flatten_metrics(metrics)
    for value in values:
        if isinstance(value, (np.generic, np.ndarray)):
            raise TypeError(f"Value `{value}` has type `{type(value)}`.")
