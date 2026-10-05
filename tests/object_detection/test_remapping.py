from copy import deepcopy
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pytest

from valor_lite.cache import FileCacheReader
from valor_lite.object_detection import (
    BoundingBox,
    Detection,
    Evaluator,
    Loader,
)
from valor_lite.object_detection.evaluator import Builder


def _box(uid, xmin, labels, scores=None):
    return BoundingBox(uid, xmin, xmin + 10, 0, 10, labels, scores or [])


@pytest.fixture
def remapping_detections():
    return [
        Detection(
            "changed_match",
            [_box("g0", 0, ["cat"]), _box("g1", 5, ["dog"])],
            [_box("p0", 5, ["cat"], [0.9])],
        ),
        Detection(
            "colliding_labels",
            [_box("g2", 0, ["dog"])],
            [
                _box("p1", 0, ["cat", "dog"], [0.8, 0.3]),
                _box("p2", 2, ["dog"], [0.7]),
            ],
        ),
        Detection("gt_only", [_box("g3", 0, ["cat"])], []),
        Detection("pd_only", [], [_box("p3", 0, ["cat", "dog"], [0.6, 0.2])]),
        Detection(
            "no_overlap",
            [_box("g4", 0, ["bird"])],
            [_box("p4", 20, ["dog"], [0.4])],
        ),
    ]


def _table(reader):
    return pa.concat_tables(list(reader.iterate_tables()))


def _normalized_rows(evaluator, reader_name):
    rows = _table(getattr(evaluator, reader_name)).to_pylist()
    for row in rows:
        for side in ("gt", "pd"):
            column = f"{side}_label_id"
            row[column] = evaluator._index_to_label.get(row[column])
    return rows


def _assert_metrics_equal(actual, expected):
    thresholds = dict(
        iou_thresholds=[0.25, 0.5, 0.75, 1.0],
        score_thresholds=[0.1, 0.5, 0.85],
    )
    actual_pr = actual.compute_precision_recall(**thresholds)
    expected_pr = expected.compute_precision_recall(**thresholds)
    for metric_type, metrics in actual_pr.items():
        # Label indices may differ from a fresh loader's discovery order.
        assert sorted(
            [m.to_dict() for m in metrics], key=lambda m: str(m["parameters"])
        ) == sorted(
            [m.to_dict() for m in expected_pr[metric_type]],
            key=lambda m: str(m["parameters"]),
        )
    for method in (
        "compute_confusion_matrix",
        "compute_examples",
        "compute_confusion_matrix_with_examples",
    ):
        assert [
            m.to_dict() for m in getattr(actual, method)(**thresholds)
        ] == [m.to_dict() for m in getattr(expected, method)(**thresholds)]


@pytest.mark.parametrize(
    "mapping",
    [
        {},
        {"absent": "unused"},
        {"cat": "cat"},
        {"cat": "feline"},
        {"cat": "animal", "dog": "animal"},
        {"cat": "dog"},
        {"cat": "dog", "dog": "cat"},
        {"cat": "dog", "dog": "animal"},
    ],
)
def test_remap_matches_fresh_loader(
    loader: Loader, tmp_path: Path, remapping_detections, mapping
):
    loader.add_bounding_boxes(remapping_detections)
    source = loader.finalize(batch_size=1)
    detailed_before = _table(source._detailed_reader)
    ranked_before = _table(source._ranked_reader)
    vocabulary_before = source._index_to_label.copy()
    mapping_before = mapping.copy()

    destination = tmp_path / "remapped"
    remapped = source.remap_labels(mapping, batch_size=1, path=destination)

    if isinstance(source._detailed_reader, FileCacheReader):
        fresh_loader = Loader.persistent(
            path=tmp_path / "fresh",
            batch_size=source._detailed_reader.batch_size,
            rows_per_file=source._detailed_reader.rows_per_file,
            compression=source._detailed_reader.compression,
            metadata_fields=source._metadata_fields,
        )
        assert isinstance(remapped._detailed_reader, FileCacheReader)
        assert remapped._detailed_reader.path == destination / "detailed"
        assert (
            remapped._detailed_reader.count_tables()
            == source._detailed_reader.count_tables()
        )
        for setting in ("batch_size", "rows_per_file", "compression"):
            assert getattr(remapped._detailed_reader, setting) == getattr(
                source._detailed_reader, setting
            )
    else:
        fresh_loader = Loader.in_memory(
            batch_size=source._detailed_reader.batch_size,
            metadata_fields=source._metadata_fields,
        )
        assert not destination.exists()

    detections = deepcopy(remapping_detections)
    for detection in detections:
        for annotation in detection.groundtruths + detection.predictions:
            annotation.labels = [
                mapping.get(label, label) for label in annotation.labels
            ]
    fresh_loader.add_bounding_boxes(detections)
    fresh = fresh_loader.finalize(batch_size=1)

    for reader_name in ("_detailed_reader", "_ranked_reader"):
        assert _normalized_rows(remapped, reader_name) == _normalized_rows(
            fresh, reader_name
        )
    _assert_metrics_equal(remapped, fresh)
    assert remapped.info == fresh.info
    assert remapped._metadata_fields == source._metadata_fields
    assert remapped is not source
    assert set(remapped._index_to_label.values()) == {
        mapping.get(label, label) for label in vocabulary_before.values()
    }
    assert source._index_to_label == vocabulary_before
    assert mapping == mapping_before
    assert _table(source._detailed_reader).equals(detailed_before)
    assert _table(source._ranked_reader).equals(ranked_before)

    if isinstance(source._detailed_reader, FileCacheReader):
        reloaded = Evaluator.load(destination)
        _assert_metrics_equal(reloaded, remapped)
        assert reloaded.info == remapped.info


def test_remap_recovers_discarded_match(
    loader: Loader, tmp_path: Path, remapping_detections
):
    loader.add_bounding_boxes(remapping_detections)
    source = loader.finalize()
    remapped = source.remap_labels(
        {"cat": "animal", "dog": "animal"}, path=tmp_path / "remapped"
    )
    condition = pc.field("datum_uid") == "changed_match"
    before = _table(source._ranked_reader).filter(condition).to_pylist()
    after = _table(remapped._ranked_reader).filter(condition).to_pylist()
    assert len(before) == len(after) == 1
    assert before[0]["gt_id"] == 0
    assert before[0]["iou"] == pytest.approx(1 / 3)
    assert after[0]["gt_id"] == 1
    assert after[0]["iou"] == 1.0

    # Collapsed prediction labels retain the best score only in ranked data.
    rows = _table(remapped._ranked_reader).filter(
        pc.field("datum_uid") == "colliding_labels"
    )
    assert rows["pd_score"].to_pylist() == [0.8, 0.7]
    assert rows["iou_prev"].to_pylist() == [0.0, 2.0]


def test_remap_and_filter(
    loader: Loader, tmp_path: Path, remapping_detections
):
    loader.add_bounding_boxes(remapping_detections)
    source = loader.finalize()
    filtered = source.filter(
        predictions=pc.field("pd_label") == "cat",
        path=tmp_path / "filtered",
    )
    remapped = filtered.remap_labels(
        {"cat": "feline"}, path=tmp_path / "filtered_remapped"
    )
    before = _table(filtered._detailed_reader)
    after = _table(remapped._detailed_reader)
    # Filtering leaves stale strings behind; remapping must not revive IDs.
    for side in ("gt", "pd"):
        assert pc.equal(  # type: ignore[reportAttributeAccessIssue]
            before[f"{side}_label_id"], -1
        ).equals(
            pc.equal(  # type: ignore[reportAttributeAccessIssue]
                after[f"{side}_label_id"], -1
            )
        )
    assert set(remapped._index_to_label.values()) == {"feline", "dog", "bird"}

    remapped_first = source.remap_labels(
        {"cat": "feline"}, path=tmp_path / "remapped"
    )
    filtered_second = remapped_first.filter(
        predictions=pc.field("pd_label") == "feline",
        path=tmp_path / "remapped_filtered",
    )
    _assert_metrics_equal(remapped, filtered_second)

    # A vocabulary entry removed from the rows by filtering is still remapped.
    cat_only = source.filter(
        datums=pc.field("datum_uid") == "gt_only",
        path=tmp_path / "cat_only",
    )
    renamed = cat_only.remap_labels(
        {"dog": "canine"}, path=tmp_path / "cat_only_remapped"
    )
    assert set(renamed._index_to_label.values()) == {"cat", "canine", "bird"}
    assert renamed.info.number_of_groundtruth_annotations == 1


@pytest.mark.parametrize("mapping", [None, [], {"cat": 1}, {1: "cat"}])
def test_invalid_mapping(mapping):
    loader = Loader.in_memory()
    loader.add_bounding_boxes([Detection("d", [_box("g", 0, ["cat"])], [])])
    evaluator = loader.finalize()
    with pytest.raises(TypeError, match="mapping must be a dict"):
        evaluator.remap_labels(mapping)


def test_persistent_remap_requires_new_path(tmp_path: Path):
    source_path = tmp_path / "source"
    loader = Loader.persistent(source_path)
    loader.add_bounding_boxes([Detection("d", [_box("g", 0, ["cat"])], [])])
    source = loader.finalize()
    before = _table(source._detailed_reader)
    with pytest.raises(ValueError, match="path"):
        source.remap_labels({"cat": "dog"})
    with pytest.raises(FileExistsError):
        source.remap_labels({"cat": "dog"}, path=source_path)
    assert _table(Evaluator.load(source_path)._detailed_reader).equals(before)


@pytest.mark.parametrize(
    "mapping", [{}, {"cat": "feline"}, {"cat": "dog", "dog": "cat"}]
)
def test_renaming_reuses_ranked_pairs(
    loader, tmp_path, remapping_detections, monkeypatch, mapping
):
    loader.add_bounding_boxes(remapping_detections)
    source = loader.finalize()

    def fail_ranking(*args, **kwargs):
        pytest.fail("A one-to-one rename should not rebuild ranked pairs")

    monkeypatch.setattr(Builder, "_rank", fail_ranking)
    remapped = source.remap_labels(mapping, path=tmp_path / "renamed")
    columns = [
        name
        for name in source._ranked_reader.schema.names
        if name not in ("gt_label_id", "pd_label_id")
    ]
    assert (
        _table(remapped._ranked_reader)
        .select(columns)
        .equals(_table(source._ranked_reader).select(columns))
    )


def test_dense_remapping_matches_sorted_reference(loader, tmp_path):
    from valor_lite.object_detection.computation import rank_table

    loader.add_bounding_boxes(
        [
            Detection(
                f"dense{datum}",
                [
                    _box(f"g{datum}_{i}", i / 10, ["cat" if i % 2 else "dog"])
                    for i in range(20)
                ],
                [
                    _box(
                        f"p{datum}_{i}",
                        i % 10 / 10,
                        ["cat", "dog"],
                        [0.9 - (i % 10) / 100, 0.5],
                    )
                    for i in range(384)
                ],
            )
            for datum in range(2)
        ]
    )
    source = loader.finalize()
    remapped = source.remap_labels(
        {"cat": "animal", "dog": "animal"},
        path=tmp_path / "remapped",
        batch_size=7,
    )
    columns = [
        name
        for name in remapped._ranked_reader.schema.names
        if name != "iou_prev"
    ]
    reference = rank_table(_table(remapped._detailed_reader).select(columns))
    actual = _table(remapped._ranked_reader)
    # Tied rows from different fragments can interleave differently, but the
    # selected matches and their AP boundaries must be identical.
    sorting = [("datum_id", "ascending"), ("pd_id", "ascending")]
    assert actual.sort_by(sorting).equals(reference.sort_by(sorting))
