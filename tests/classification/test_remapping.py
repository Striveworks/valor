import json
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pytest

from valor_lite.cache import FileCacheReader
from valor_lite.classification import Classification, Evaluator, Loader


@pytest.fixture
def remapping_classifications():
    return [
        Classification(
            uid=f"datum{i}",
            groundtruth=groundtruth,
            predictions=["cat", "dog", "bird"],
            scores=scores,
            metadata={"test": float(i)},
        )
        for i, (groundtruth, scores) in enumerate(
            [
                ("dog", [0.7, 0.7, 0.1]),
                ("cat", [0.2, 0.8, 0.8]),
                ("bird", [0.4, 0.35, 0.6]),
                ("cat", [0.9, 0.1, 0.0]),
                ("dog", [0.0, 0.0, 0.0]),
                ("bird", [0.2, 0.7, 0.1]),
            ]
        )
    ]


def _table(reader):
    return pa.concat_tables(list(reader.iterate_tables()))


def _metric_dicts(metrics):
    return sorted(
        [metric.to_dict() for metric in metrics],
        key=lambda metric: json.dumps(metric["parameters"], sort_keys=True),
    )


def _assert_metrics_equal(actual, expected):
    actual_roc = actual.compute_rocauc()
    expected_roc = expected.compute_rocauc()
    for metric_type, metrics in actual_roc.items():
        assert _metric_dicts(metrics) == _metric_dicts(
            expected_roc[metric_type]
        )
    for hardmax in (True, False):
        thresholds = dict(score_thresholds=[0.0, 0.5, 0.75], hardmax=hardmax)
        actual_pr = actual.compute_precision_recall(**thresholds)
        expected_pr = expected.compute_precision_recall(**thresholds)
        for metric_type, metrics in actual_pr.items():
            assert _metric_dicts(metrics) == _metric_dicts(
                expected_pr[metric_type]
            )
        for method in (
            "compute_confusion_matrix",
            "compute_examples",
            "compute_confusion_matrix_with_examples",
        ):
            assert _metric_dicts(getattr(actual, method)(**thresholds)) == (
                _metric_dicts(getattr(expected, method)(**thresholds))
            )


def _normalized_rows(evaluator):
    rows = _table(evaluator._reader).to_pylist()
    for row in rows:
        for side in ("gt", "pd"):
            column = f"{side}_label_id"
            row[column] = evaluator._index_to_label[row[column]]
    return sorted(rows, key=lambda row: (row["datum_id"], row["pd_label"]))


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
        {"cat": "animal", "dog": "animal", "bird": "animal"},
    ],
)
def test_remap_matches_fresh_loader(
    loader: Loader, tmp_path: Path, remapping_classifications, mapping
):
    loader.add_data(remapping_classifications)
    source = loader.finalize(batch_size=1)
    data_before = _table(source._reader)
    roc_before = _table(source._roc_curve_reader)
    vocabulary_before = source._index_to_label.copy()
    mapping_before = mapping.copy()
    destination = tmp_path / "remapped"
    remapped = source.remap_labels(mapping, batch_size=1, path=destination)

    if isinstance(source._reader, FileCacheReader):
        fresh_loader = Loader.persistent(
            path=tmp_path / "fresh",
            batch_size=source._reader.batch_size,
            rows_per_file=source._reader.rows_per_file,
            compression=source._reader.compression,
            metadata_fields=source._metadata_fields,
        )
        assert isinstance(remapped._reader, FileCacheReader)
        assert remapped._reader.path == destination / "cache"
        assert remapped._reader.count_tables() == source._reader.count_tables()
        for setting in ("batch_size", "rows_per_file", "compression"):
            assert getattr(remapped._reader, setting) == getattr(
                source._reader, setting
            )
    else:
        fresh_loader = Loader.in_memory(
            batch_size=source._reader.batch_size,
            metadata_fields=source._metadata_fields,
        )
        assert not destination.exists()

    classifications = []
    for classification in remapping_classifications:
        # Keep maximum scores with original input order breaking ties.
        predictions = {}
        for index in sorted(
            range(len(classification.scores)),
            key=lambda index: -classification.scores[index],
        ):
            label = classification.predictions[index]
            predictions.setdefault(
                mapping.get(label, label), classification.scores[index]
            )
        classifications.append(
            Classification(
                uid=classification.uid,
                groundtruth=mapping.get(
                    classification.groundtruth, classification.groundtruth
                ),
                predictions=list(predictions),
                scores=list(predictions.values()),
                metadata=classification.metadata,
            )
        )
    fresh_loader.add_data(classifications)
    fresh = fresh_loader.finalize(batch_size=1)
    assert _normalized_rows(remapped) == _normalized_rows(fresh)
    _assert_metrics_equal(remapped, fresh)
    assert remapped.info == fresh.info
    assert remapped is not source
    assert set(remapped._index_to_label.values()) == {
        mapping.get(label, label) for label in vocabulary_before.values()
    }
    assert mapping == mapping_before
    assert source._index_to_label == vocabulary_before
    assert _table(source._reader).equals(data_before)
    assert _table(source._roc_curve_reader).equals(roc_before)

    if isinstance(source._reader, FileCacheReader):
        reloaded = Evaluator.load(destination)
        _assert_metrics_equal(reloaded, remapped)
        assert reloaded.info == remapped.info


def test_remap_collapses_predictions_and_preserves_winners(
    loader: Loader, tmp_path: Path, remapping_classifications
):
    loader.add_data(remapping_classifications)
    source = loader.finalize()
    remapped = source.remap_labels(
        {"cat": "animal", "dog": "animal"}, path=tmp_path / "remapped"
    )
    table = _table(remapped._reader)
    assert table.num_rows == 12
    for datum in range(6):
        rows = table.filter(pc.field("datum_id") == datum).to_pylist()
        assert len(rows) == 2
        assert sum(row["pd_winner"] for row in rows) == 1
    tied = table.filter(pc.field("datum_id") == 0).to_pylist()
    animal = next(row for row in tied if row["pd_label"] == "animal")
    assert animal["pd_winner"]
    assert animal["match"]
    assert animal["pd_score"] == 0.7

    # Scores are maximized, not summed: bird still wins over merged animal.
    rows = table.filter(pc.field("datum_id") == 2).to_pylist()
    assert (
        next(row for row in rows if row["pd_label"] == "animal")["pd_score"]
        == 0.4
    )
    assert next(row for row in rows if row["pd_winner"])["pd_label"] == "bird"


def test_remap_filtered_cache(
    loader: Loader, tmp_path: Path, remapping_classifications
):
    loader.add_data(remapping_classifications)
    source = loader.finalize()
    filtered = source.filter(
        predictions=pc.field("pd_label") != "cat", path=tmp_path / "filtered"
    )
    remapped = filtered.remap_labels(
        {"dog": "animal", "bird": "animal"}, path=tmp_path / "remapped"
    )
    # Removing the original winner must not promote a remaining prediction.
    row = _table(remapped._reader).filter(pc.field("datum_id") == 0)
    assert row.num_rows == 1
    assert row["pd_winner"].to_pylist() == [False]

    renamed = source.remap_labels({"cat": "feline"}, path=tmp_path / "renamed")
    filtered_after = renamed.filter(
        predictions=pc.field("pd_label") == "feline",
        path=tmp_path / "filtered_after",
    )
    filtered_before = source.filter(
        predictions=pc.field("pd_label") == "cat",
        path=tmp_path / "filtered_before",
    ).remap_labels({"cat": "feline"}, path=tmp_path / "renamed_after")
    _assert_metrics_equal(filtered_after, filtered_before)

    # Empty fragments and vocabulary entries absent from the rows survive.
    subset = source.filter(
        datums=pc.field("datum_uid") == "datum3",
        predictions=pc.field("pd_label") == "cat",
        path=tmp_path / "subset",
    ).remap_labels({"dog": "canine"}, path=tmp_path / "subset_remapped")
    assert set(subset._index_to_label.values()) == {"cat", "canine", "bird"}
    assert subset.info.number_of_datums == 1
    assert subset.info.number_of_rows == 1


@pytest.mark.parametrize("mapping", [None, [], {"cat": 1}, {1: "cat"}])
def test_invalid_mapping(mapping):
    loader = Loader.in_memory()
    loader.add_data([Classification("d", "cat", ["cat"], [1.0])])
    source = loader.finalize()
    with pytest.raises(TypeError, match="mapping must be a dict"):
        source.remap_labels(mapping)


def test_persistent_remap_requires_new_path(tmp_path: Path):
    source_path = tmp_path / "source"
    loader = Loader.persistent(source_path)
    loader.add_data([Classification("d", "cat", ["cat"], [1.0])])
    source = loader.finalize()
    before = _table(source._reader)
    with pytest.raises(ValueError, match="path"):
        source.remap_labels({"cat": "dog"})
    with pytest.raises(FileExistsError):
        source.remap_labels({"cat": "dog"}, path=source_path)
    assert _table(Evaluator.load(source_path)._reader).equals(before)
