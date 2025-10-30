import json
from pathlib import Path

import numpy as np
import pytest

from valor_lite.cache.persistent import FileCacheReader
from valor_lite.object_detection import Evaluator, Metric, MetricType


def test_evaluator_file_not_found(tmp_path: Path):
    path = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        Evaluator.load(path)


def test_evaluator_not_a_directory(tmp_path: Path):
    filepath = tmp_path / "file"
    with open(filepath, "w") as f:
        json.dump({}, f, indent=2)
    with pytest.raises(NotADirectoryError):
        Evaluator.load(filepath)


def test_evaluator_valid_thresholds(tmp_path: Path):
    eval = Evaluator(
        path=tmp_path,
        detailed_reader=None,  # type: ignore - testing
        ranked_reader=None,  # type: ignore - testing
        info=None,  # type: ignore - testing
        index_to_label={},
        number_of_groundtruths_per_label=np.ones(1, dtype=np.uint64),
    )
    for fn in [
        eval.compute_precision_recall,
        eval.compute_examples,
        eval.compute_confusion_matrix,
        eval.compute_confusion_matrix_with_examples,
    ]:
        with pytest.raises(ValueError) as e:
            fn(iou_thresholds=[], score_thresholds=[0.5])
        assert "IOU" in str(e)
        with pytest.raises(ValueError) as e:
            fn(iou_thresholds=[0.5], score_thresholds=[])
        assert "score" in str(e)


def test_info_using_torch_metrics_example(torchmetrics_detections: Evaluator):
    """
    cf with torch metrics/pycocotools results listed here:
    https://github.com/Lightning-AI/metrics/blob/107dbfd5fb158b7ae6d76281df44bd94c836bfce/tests/unittests/detection/test_map.py#L231
    """
    evaluator = torchmetrics_detections

    assert evaluator.info.number_of_datums == 4
    assert evaluator.info.number_of_labels == 6
    assert evaluator.info.number_of_groundtruth_annotations == 20
    assert evaluator.info.number_of_prediction_annotations == 19


def test_no_thresholds(detection_ranked_pair_ordering: Evaluator):
    evaluator = detection_ranked_pair_ordering
    evaluator.compute_precision_recall(
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


def test_no_groundtruths(detections_no_groundtruths: Evaluator):
    evaluator = detections_no_groundtruths
    assert evaluator.info.number_of_datums == 2
    assert evaluator.info.number_of_labels == 1
    assert evaluator.info.number_of_groundtruth_annotations == 0
    assert evaluator.info.number_of_prediction_annotations == 2

    metrics = evaluator.compute_precision_recall(
        iou_thresholds=[0.5],
        score_thresholds=[0.5],
    )
    actual_metrics = [m.to_dict() for m in metrics[MetricType.AP]]
    assert len(actual_metrics) == 1
    expected_metrics = [
        {
            "type": "AP",
            "parameters": {"iou_threshold": 0.5, "label": "v1"},
            "value": 0.0,
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_no_predictions(detections_no_predictions: Evaluator):
    evaluator = detections_no_predictions
    assert evaluator.info.number_of_datums == 2
    assert evaluator.info.number_of_labels == 1
    assert evaluator.info.number_of_groundtruth_annotations == 2
    assert evaluator.info.number_of_prediction_annotations == 0

    metrics = evaluator.compute_precision_recall(
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


def test_output_types_dont_contain_numpy(basic_detections: Evaluator):
    evaluator = basic_detections

    metrics = evaluator.compute_precision_recall(
        iou_thresholds=[0.5],
        score_thresholds=[0.25, 0.75],
    )

    values = _flatten_metrics(metrics)
    for value in values:
        if isinstance(value, (np.generic, np.ndarray)):
            raise TypeError(f"Value `{value}` has type `{type(value)}`.")


def test_evaluator_deletion(
    false_negatives_single_datum_detections: Evaluator,
):
    # create evaluator
    evaluator = false_negatives_single_datum_detections

    if isinstance(evaluator.detailed_reader, FileCacheReader):
        path = evaluator._path
        assert path

        # check both caches exist
        assert path.exists()
        assert evaluator._generate_detailed_cache_path(path).exists()
        assert evaluator._generate_ranked_cache_path(path).exists()
        assert evaluator._generate_metadata_path(path).exists()

        # verify deletion
        evaluator.delete()
        assert not path.exists()
        assert not evaluator._generate_detailed_cache_path(path).exists()
        assert not evaluator._generate_ranked_cache_path(path).exists()
        assert not evaluator._generate_metadata_path(path).exists()
