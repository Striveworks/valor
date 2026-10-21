import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from valor_lite.object_detection import (
    BoundingBox,
    Detection,
    Evaluator,
    Loader,
    Metric,
    MetricType,
)


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


@pytest.mark.parametrize("persistent", [False, True])
def test_cache_introspection(tmp_path: Path, persistent: bool):
    metadata_fields: list[tuple[str, str | pa.DataType]] = [
        ("weather", pa.string()),
        ("site", pa.int32()),
    ]
    if persistent:
        loader = Loader.persistent(
            path=tmp_path,
            batch_size=1,
            rows_per_file=1,
            metadata_fields=metadata_fields,
        )
    else:
        loader = Loader.in_memory(metadata_fields=metadata_fields)
    loader.add_bounding_boxes(
        [
            Detection(
                uid=f"datum-{i}",
                metadata={"weather": weather, "site": i},
                groundtruths=[
                    BoundingBox(
                        uid=f"gt-{i}-{j}",
                        xmin=0,
                        xmax=10,
                        ymin=0,
                        ymax=10,
                        labels=["label"],
                    )
                    for j in range(2)
                ],
                predictions=[],
            )
            for i, weather in enumerate([None, "rain"])
        ]
    )
    evaluator = loader.finalize()
    if persistent:
        evaluator = Evaluator.load(tmp_path)

    assert evaluator.metadata_fields == ["weather", "site"]
    assert evaluator.reader is evaluator._detailed_reader
    assert evaluator.reader.schema.field("site").type == pa.int32()
    assert evaluator.reserved_fields == [
        "datum_uid",
        "datum_id",
        "gt_uid",
        "gt_id",
        "gt_label",
        "gt_label_id",
        "pd_uid",
        "pd_id",
        "pd_label",
        "pd_label_id",
        "pd_score",
        "iou",
    ]

    ungrouped = list(
        evaluator.reader.iterate_tables(columns=evaluator.metadata_fields)
    )
    assert len(ungrouped) == (2 if persistent else 1)
    assert all(
        table.column_names == ["weather", "site"] for table in ungrouped
    )
    assert sorted(
        pa.concat_tables(ungrouped).to_pylist(), key=lambda row: row["site"]
    ) == [
        {"weather": None, "site": 0},
        {"weather": None, "site": 0},
        {"weather": "rain", "site": 1},
        {"weather": "rain", "site": 1},
    ]


def test_evaluator_valid_thresholds():
    eval = Evaluator(
        detailed_reader=None,  # type: ignore - testing
        ranked_reader=None,  # type: ignore - testing
        index_to_label={},
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


def test_info_using_torch_metrics_example(torchmetrics_evaluator: Evaluator):
    """
    cf with torch metrics/pycocotools results listed here:
    https://github.com/Lightning-AI/metrics/blob/107dbfd5fb158b7ae6d76281df44bd94c836bfce/tests/unittests/detection/test_map.py#L231
    """
    assert torchmetrics_evaluator.info.number_of_datums == 4
    assert torchmetrics_evaluator.info.number_of_labels == 6
    assert torchmetrics_evaluator.info.number_of_groundtruth_annotations == 20
    assert torchmetrics_evaluator.info.number_of_prediction_annotations == 19


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


def test_evaluator_loading_using_torch_metrics_example(
    tmp_path: Path, torchmetrics_detections: list[Detection[BoundingBox]]
):
    loader = Loader.persistent(path=tmp_path)
    loader.add_bounding_boxes(torchmetrics_detections)
    original_evaluator = loader.finalize()
    loaded_evaluator = Evaluator.load(path=tmp_path)

    kwargs = dict(
        score_thresholds=[0.25, 0.5, 0.75, 0.9],
        iou_thresholds=[0.1, 0.25, 0.5, 0.75],
    )
    assert original_evaluator.compute_precision_recall(
        **kwargs
    ) == loaded_evaluator.compute_precision_recall(**kwargs)
