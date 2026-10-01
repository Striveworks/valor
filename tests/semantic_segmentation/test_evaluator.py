import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from valor_lite.cache import FileCacheWriter
from valor_lite.semantic_segmentation import (
    Builder,
    Evaluator,
    Loader,
    Metric,
    MetricType,
    Segmentation,
)


def test_evaluator_file_not_found(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        Evaluator.load(tmp_path / "does_not_exist")


def test_evaluator_not_a_directory(tmp_path: Path):
    filepath = tmp_path / "file"
    filepath.write_text("{}")
    with pytest.raises(NotADirectoryError):
        Evaluator.load(filepath)


def test_info_using_large_random_segmentations(
    loader: Loader, large_random_segmentations
):
    loader.add_data(large_random_segmentations)
    info = loader.finalize().info
    assert info.number_of_datums == 3
    assert info.number_of_labels == 6
    assert info.number_of_pixels == 12000000
    assert (
        info.number_of_groundtruth_pixels
        == info.number_of_prediction_pixels
        == info.number_of_pixels
    )


def _flatten_metrics(m) -> list:
    if isinstance(m, dict):
        return list(m.keys()) + [
            v for value in m.values() for v in _flatten_metrics(value)
        ]
    if isinstance(m, list):
        return [v for value in m for v in _flatten_metrics(value)]
    if isinstance(m, Metric):
        return _flatten_metrics(m.to_dict())
    return [m]


def test_output_types_dont_contain_numpy(
    loader: Loader, segmentations_from_boxes
):
    loader.add_data(segmentations_from_boxes)
    metrics = loader.finalize().compute_precision_recall_iou()
    assert not any(
        isinstance(value, (np.generic, np.ndarray))
        for value in _flatten_metrics(metrics)
    )
    json.dumps(
        {
            key.value: [m.to_dict() for m in values]
            for key, values in metrics.items()
        },
        allow_nan=False,
    )


@pytest.mark.parametrize("zero_side", ["groundtruths", "predictions"])
def test_zero_filled_side_is_background(loader: Loader, zero_side):
    kwargs = {
        "groundtruths": np.array([[1, 0], [0, 2]]),
        "predictions": np.array([[1, 0], [0, 2]]),
    }
    kwargs[zero_side] = np.zeros((2, 2), dtype=np.uint16)
    loader.add_data(
        [
            Segmentation(
                "image",
                groundtruths=kwargs["groundtruths"],
                predictions=kwargs["predictions"],
                labels=["sky", "road", "car"],
            )
        ]
    )
    evaluator = loader.finalize()
    expected = np.array(
        [[2, 1, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    )
    if zero_side == "predictions":
        expected = expected.T
    np.testing.assert_array_equal(
        evaluator._compute_confusion_matrix_intermediate(), expected
    )
    assert evaluator.info.number_of_groundtruth_pixels == (
        0 if zero_side == "groundtruths" else 2
    )
    assert evaluator.info.number_of_prediction_pixels == (
        0 if zero_side == "predictions" else 2
    )
    assert (
        evaluator.compute_precision_recall_iou()[MetricType.Accuracy][0].value
        == 0.5
    )


def test_evaluator_loading(tmp_path: Path, basic_segmentations):
    loader = Loader.persistent(tmp_path)
    loader.add_data(basic_segmentations)
    original = loader.finalize()
    evaluator = Evaluator.load(tmp_path)
    assert evaluator.info == original.info
    assert (
        evaluator.compute_precision_recall_iou()
        == original.compute_precision_recall_iou()
    )


@pytest.mark.parametrize("batch_size,rows_per_file", [(1, 1), (10000, 100000)])
def test_filtered_cache_preserves_vocabulary_on_reload(
    tmp_path: Path, batch_size, rows_per_file
):
    loader = Loader.persistent(
        tmp_path / "source", batch_size=batch_size, rows_per_file=rows_per_file
    )
    loader.add_data(
        [
            Segmentation(
                "image",
                np.array([[1, 2]]),
                np.array([[1, 2]]),
                ["sky", "road", "absent"],
            )
        ]
    )
    original = loader.finalize()
    path = tmp_path / "filtered"
    filtered = original.filter(
        groundtruths=pc.field("gt_label") == "road",
        predictions=pc.field("pd_label") == "road",
        path=path,
    )
    reloaded = Evaluator.load(path)
    assert reloaded._index_to_label == {1: "road"}
    assert not (path / "labels.json").exists()
    assert reloaded.info.number_of_pixels == 1
    assert (
        reloaded.compute_precision_recall_iou()
        == filtered.compute_precision_recall_iou()
    )
    assert reloaded.info == filtered.info
    assert (
        reloaded.compute_precision_recall_iou()[MetricType.mIOU][0].value
        == 1.0
    )


@pytest.mark.parametrize(
    "metadata_type,metadata_value",
    [(None, None), ("bool", False), ("string", "user data")],
)
def test_legacy_cache_with_noncontiguous_ids(
    tmp_path: Path, metadata_type, metadata_value
):
    metadata_fields = (
        [("gt_valid", metadata_type), ("pd_valid", metadata_type)]
        if metadata_type
        else []
    )
    metadata = {name: metadata_value for name, _ in metadata_fields}
    schema = pa.schema(
        [
            ("datum_uid", pa.string()),
            ("datum_id", pa.int64()),
            ("gt_label", pa.string()),
            ("gt_label_id", pa.int64()),
            ("pd_label", pa.string()),
            ("pd_label_id", pa.int64()),
            ("count", pa.uint64()),
            *metadata_fields,
        ]
    )
    writer = FileCacheWriter.create(
        path=tmp_path / "counts",
        schema=schema,
        batch_size=100,
        rows_per_file=100,
        compression="snappy",
    )
    (tmp_path / "metadata.json").write_text(json.dumps(dict(metadata_fields)))
    builder = Builder(writer, metadata_fields=metadata_fields)
    builder._writer.write_rows(
        [
            {
                **metadata,
                "datum_uid": "image",
                "datum_id": 0,
                "gt_label": "road",
                "gt_label_id": 7,
                "pd_label": "road",
                "pd_label_id": 7,
                "count": 2,
            },
            {
                **metadata,
                "datum_uid": "image",
                "datum_id": 0,
                "gt_label": None,
                "gt_label_id": -1,
                "pd_label": "road",
                "pd_label_id": 7,
                "count": 1,
            },
        ]
    )
    original = builder.finalize()
    assert not (tmp_path / "labels.json").exists()
    reloaded = Evaluator.load(tmp_path)
    assert (
        reloaded.compute_precision_recall_iou()
        == original.compute_precision_recall_iou()
    )
    np.testing.assert_array_equal(
        reloaded._compute_confusion_matrix_intermediate(), [[0, 1], [0, 2]]
    )
    assert (
        reloaded.compute_precision_recall_iou()[MetricType.Precision][0].value
        == 2 / 3
    )
    assert reloaded.info.number_of_pixels == 3
    assert reloaded.info.number_of_groundtruth_pixels == 2
    assert reloaded.info.number_of_prediction_pixels == 3
    filtered = reloaded.filter(
        groundtruths=pc.field("gt_label") == "road",
        predictions=pc.field("pd_label") == "road",
        path=tmp_path / "filtered",
    )
    assert (
        filtered.compute_precision_recall_iou()
        == reloaded.compute_precision_recall_iou()
    )
    assert (
        Evaluator.load(tmp_path / "filtered").compute_precision_recall_iou()
        == filtered.compute_precision_recall_iou()
    )
    for table in filtered._reader.iterate_tables():
        for name, _ in metadata_fields:
            assert table[name].to_pylist() == [metadata_value] * table.num_rows


def test_memory_and_persistent_parity(
    tmp_path: Path, basic_segmentations_three_labels
):
    evaluators = []
    for loader in [
        Loader.in_memory(batch_size=1),
        Loader.persistent(tmp_path, batch_size=1, rows_per_file=1),
    ]:
        loader.add_data(basic_segmentations_three_labels)
        evaluators.append(loader.finalize())
    assert evaluators[0].info == evaluators[1].info
    assert (
        evaluators[0].compute_precision_recall_iou()
        == evaluators[1].compute_precision_recall_iou()
    )


def test_absent_classes_are_stored_in_rows(tmp_path: Path):
    loader = Loader.persistent(tmp_path)
    loader.add_data(
        [
            Segmentation(
                "image", np.array([[1]]), np.array([[1]]), ["sky", "road"]
            )
        ]
    )
    original = loader.finalize()
    reloaded = Evaluator.load(tmp_path)
    rows = [
        row
        for tbl in reloaded._reader.iterate_tables()
        for row in tbl.to_pylist()
    ]
    assert [
        (row["gt_label"], row["pd_label"], row["count"]) for row in rows
    ] == [("sky", "sky", 1), ("road", "road", 0)]
    assert (
        reloaded.compute_precision_recall_iou()
        == original.compute_precision_recall_iou()
    )
    assert (
        reloaded.compute_precision_recall_iou()[MetricType.mIOU][0].value
        == 0.5
    )
    assert not (tmp_path / "labels.json").exists()


def test_inline_datum_filter_matches_materialized_filter(tmp_path: Path):
    loader = Loader.in_memory()
    loader.add_data(
        [
            Segmentation("first", np.array([[1]]), np.array([[1]]), ["sky"]),
            Segmentation("second", np.array([[1]]), np.array([[1]]), ["road"]),
        ]
    )
    evaluator = loader.finalize()
    expression = pc.field("datum_uid") == "second"
    filtered = evaluator.filter(datums=expression)
    assert evaluator.get_info(datums=expression) == filtered.info
    assert (
        evaluator.compute_precision_recall_iou(datums=expression)
        == filtered.compute_precision_recall_iou()
    )
    assert filtered._index_to_label == {1: "road"}
    missing = evaluator.compute_precision_recall_iou(
        datums=pc.field("datum_uid") == "missing"
    )
    assert missing[MetricType.Accuracy][0].value == 0
    assert missing[MetricType.mIOU][0].value == 0


def test_one_sided_filter_drops_excluded_vocabulary_on_reload(tmp_path: Path):
    loader = Loader.persistent(tmp_path / "source")
    loader.add_data(
        [
            Segmentation(
                "image", np.array([[1]]), np.array([[2]]), ["sky", "road"]
            )
        ]
    )
    evaluator = loader.finalize()
    path = tmp_path / "filtered"
    filtered = evaluator.filter(
        groundtruths=pc.field("gt_label") == "road", path=path
    )
    reloaded = Evaluator.load(path)
    rows = [
        row
        for tbl in reloaded._reader.iterate_tables()
        for row in tbl.to_pylist()
    ]
    assert len(rows) == 1
    assert rows[0]["gt_label"] is None and rows[0]["gt_label_id"] == -1
    assert rows[0]["pd_label"] == "road" and rows[0]["pd_label_id"] == 1
    assert (
        reloaded.info.number_of_pixels
        == reloaded.info.number_of_prediction_pixels
        == 1
    )
    assert reloaded.info.number_of_groundtruth_pixels == 0
    assert reloaded._index_to_label == {1: "road"}
    assert [
        metric.parameters["label"]
        for metric in reloaded.compute_precision_recall_iou()[MetricType.IOU]
    ] == ["road"]
    assert (
        reloaded.compute_precision_recall_iou()
        == filtered.compute_precision_recall_iou()
    )
    # A later filter cannot reactivate the already-masked ground truth.
    refiltered = reloaded.filter(
        groundtruths=pc.field("gt_label") == "sky",
        path=tmp_path / "refiltered",
    )
    assert (
        refiltered.compute_precision_recall_iou()
        == reloaded.compute_precision_recall_iou()
    )
    background = reloaded.filter(
        predictions=pc.field("pd_label") == "sky", path=tmp_path / "background"
    )
    assert background._index_to_label == {}
    assert background.info.number_of_pixels == 1
    assert (
        background.compute_precision_recall_iou()[MetricType.Accuracy][0].value
        == 1.0
    )
    assert not (path / "labels.json").exists()


@pytest.mark.parametrize("source", ["memory", "persistent", "loaded"])
def test_filters_preserve_label_name_overrides(tmp_path: Path, source):
    loader = (
        Loader.in_memory(batch_size=1)
        if source == "memory"
        else Loader.persistent(
            tmp_path / "source", batch_size=1, rows_per_file=1
        )
    )
    loader.add_data(
        [
            Segmentation("first", np.array([[1]]), np.array([[1]]), ["sky"]),
            Segmentation("second", np.array([[1]]), np.array([[1]]), ["road"]),
        ]
    )
    overrides = {0: "excluded", 1: "renamed"}
    evaluator = loader.finalize(index_to_label_override=overrides)
    if source == "loaded":
        evaluator = Evaluator.load(
            tmp_path / "source", index_to_label_override=overrides
        )
    expression = pc.field("datum_uid") == "second"
    inline_metrics = evaluator.compute_precision_recall_iou(datums=expression)
    assert inline_metrics[MetricType.IOU] == [Metric.iou(1.0, "renamed")]
    filtered = evaluator.filter(datums=expression, path=tmp_path / "datums")
    assert filtered._index_to_label == {1: "renamed"}
    assert filtered.compute_precision_recall_iou() == inline_metrics
    filtered = evaluator.filter(
        groundtruths=pc.field("gt_label") == "road",
        predictions=pc.field("pd_label") == "road",
        path=tmp_path / "annotations",
    )
    assert filtered._index_to_label == {1: "renamed"}
    assert filtered.compute_precision_recall_iou() == inline_metrics


@pytest.mark.parametrize("persistent", [False, True])
def test_validity_names_are_user_metadata(tmp_path: Path, persistent):
    metadata_fields = [("gt_valid", "bool"), ("pd_valid", "bool")]
    loader = (
        Loader.persistent(tmp_path / "source", metadata_fields=metadata_fields)
        if persistent
        else Loader.in_memory(metadata_fields=metadata_fields)
    )
    loader.add_data(
        [
            Segmentation(
                "image",
                np.array([[1]]),
                np.array([[1]]),
                ["cat"],
                metadata={"gt_valid": False, "pd_valid": False},
            )
        ]
    )
    evaluator = loader.finalize()
    assert (
        evaluator.compute_precision_recall_iou()[MetricType.Accuracy][0].value
        == 1.0
    )
    filtered = evaluator.filter(
        groundtruths=~pc.field("gt_valid"),
        predictions=~pc.field("pd_valid"),
        path=tmp_path / "filtered",
    )
    assert filtered.info == evaluator.info
    assert (
        filtered.compute_precision_recall_iou()
        == evaluator.compute_precision_recall_iou()
    )
    for table in filtered._reader.iterate_tables():
        assert table["gt_valid"].to_pylist() == [False]
        assert table["pd_valid"].to_pylist() == [False]


def test_legacy_background_counts_survive_loading(tmp_path: Path):
    builder = Builder.persistent(tmp_path / "source")
    builder._writer.write_rows(
        [
            {
                "datum_uid": "image",
                "datum_id": 0,
                "gt_label": None,
                "gt_label_id": -1,
                "pd_label": None,
                "pd_label_id": -1,
                "count": 2,
            }
        ]
    )
    builder.finalize()
    evaluator = Evaluator.load(tmp_path / "source")
    assert evaluator.info.number_of_pixels == 2
    assert evaluator.info.number_of_groundtruth_pixels == 0
    assert evaluator.info.number_of_prediction_pixels == 0
    assert (
        evaluator.compute_precision_recall_iou()[MetricType.Accuracy][0].value
        == 1.0
    )
    filtered = evaluator.filter(
        datums=pc.field("datum_uid") == "image", path=tmp_path / "filtered"
    )
    assert (
        filtered.compute_precision_recall_iou()
        == evaluator.compute_precision_recall_iou()
    )
