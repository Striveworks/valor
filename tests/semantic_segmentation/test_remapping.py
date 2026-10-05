from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from valor_lite.exceptions import EmptyCacheError
from valor_lite.semantic_segmentation import Evaluator, Loader, Segmentation


def _segmentations():
    labels = ["cat", "dog", "bird"]
    return [
        Segmentation(
            uid="datum0",
            groundtruths=np.array([[1, 1, 2, 2], [3, 0, 0, 1]]),
            predictions=np.array([[2, 1, 1, 2], [3, 1, 0, 0]]),
            labels=labels.copy(),
            metadata={"gt_xmin": 0.0, "pd_xmin": 0.0},
        ),
        Segmentation(
            uid="datum1",
            groundtruths=np.zeros((2, 4), dtype=np.uint16),
            predictions=np.array([[0, 1, 1, 0], [0, 1, 0, 1]]),
            labels=labels.copy(),
            metadata={"gt_xmin": 1.0, "pd_xmin": 1.0},
        ),
    ]


def _remap_segmentation(segmentation, mapping):
    labels = list(
        dict.fromkeys(
            mapping.get(label, label) for label in segmentation.labels
        )
    )
    indices = {label: index + 1 for index, label in enumerate(labels)}
    remap = np.zeros(len(segmentation.labels) + 1, dtype=np.uint16)
    for index, label in enumerate(segmentation.labels, start=1):
        remap[index] = indices[mapping.get(label, label)]
    return Segmentation(
        uid=segmentation.uid,
        groundtruths=remap[segmentation.groundtruths],
        predictions=remap[segmentation.predictions],
        labels=labels,
        metadata=segmentation.metadata,
    )


def _metrics(evaluator, datums=None):
    return evaluator.compute_precision_recall_iou(datums=datums)


def _assert_values_equal(actual, expected):
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_values_equal(actual[key], expected[key])
    elif isinstance(expected, (int, float)):
        assert actual == pytest.approx(expected)
    else:
        assert actual == expected


def _assert_metrics_equal(actual, expected, datums=None):
    actual_metrics = _metrics(actual, datums)
    expected_metrics = _metrics(expected, datums)
    assert actual_metrics.keys() == expected_metrics.keys()
    for metric_type, actual_values in actual_metrics.items():
        expected_values = expected_metrics[metric_type]
        actual_values = sorted(
            actual_values, key=lambda metric: str(metric.parameters)
        )
        expected_values = sorted(
            expected_values, key=lambda metric: str(metric.parameters)
        )
        assert len(actual_values) == len(expected_values)
        for actual_metric, expected_metric in zip(
            actual_values, expected_values
        ):
            assert actual_metric.parameters == expected_metric.parameters
            _assert_values_equal(actual_metric.value, expected_metric.value)


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
def test_remap_matches_loading_remapped_label_maps(tmp_path: Path, mapping):
    segmentations = _segmentations()
    metadata = [("gt_xmin", pa.float64()), ("pd_xmin", pa.float64())]
    loader = Loader.in_memory(metadata_fields=metadata)
    loader.add_data(segmentations)
    source = loader.finalize()
    before = pa.concat_tables(list(source._reader.iterate_tables()))

    remapped = source.remap_labels(mapping)
    fresh_loader = Loader.in_memory(metadata_fields=source._metadata_fields)
    fresh_loader.add_data(
        [_remap_segmentation(item, mapping) for item in segmentations]
    )
    fresh = fresh_loader.finalize()

    _assert_metrics_equal(remapped, fresh)
    for segmentation in segmentations:
        _assert_metrics_equal(
            remapped,
            fresh,
            datums=pc.field("datum_uid") == segmentation.uid,
        )
    assert remapped.info.number_of_pixels == fresh.info.number_of_pixels
    assert remapped.info.number_of_labels == fresh.info.number_of_labels
    after = pa.concat_tables(list(remapped._reader.iterate_tables()))
    for column in ("datum_uid", "datum_id", "count", "gt_xmin", "pd_xmin"):
        assert after[column].equals(before[column])
    assert remapped is not source
    unchanged = pa.concat_tables(list(source._reader.iterate_tables()))
    assert unchanged.equals(before)


def test_remap_persistent_cache_round_trip(tmp_path: Path):
    source_path = tmp_path / "source"
    destination = tmp_path / "remapped"
    loader = Loader.persistent(
        source_path,
        metadata_fields=[("gt_xmin", pa.float64()), ("pd_xmin", pa.float64())],
    )
    loader.add_data(_segmentations())
    source = loader.finalize()
    remapped = source.remap_labels(
        {"cat": "animal", "dog": "animal"}, path=destination
    )
    reloaded = Evaluator.load(destination)
    _assert_metrics_equal(remapped, reloaded)
    assert reloaded.info == remapped.info
    assert remapped._index_to_label == {0: "animal", 1: "bird"}


def test_remap_filters_and_preserves_background(tmp_path: Path):
    metadata = [("gt_xmin", pa.float64()), ("pd_xmin", pa.float64())]
    loader = Loader.in_memory(metadata_fields=metadata)
    loader.add_data(_segmentations())
    source = loader.finalize()
    filtered = source.filter(
        groundtruths=pc.field("gt_label") != "cat",
        predictions=pc.field("pd_label") != "dog",
    )
    remapped = filtered.remap_labels({"cat": "animal", "dog": "animal"})
    before = pa.concat_tables(list(filtered._reader.iterate_tables()))
    after = pa.concat_tables(list(remapped._reader.iterate_tables()))
    for side in ("gt", "pd"):
        invalid = pc.equal(  # type: ignore[reportAttributeAccessIssue]
            before[f"{side}_label_id"], -1
        )
        assert invalid.equals(
            pc.equal(  # type: ignore[reportAttributeAccessIssue]
                after[f"{side}_label_id"], -1
            )
        )
        assert (
            before[f"{side}_label"]
            .filter(invalid)
            .equals(after[f"{side}_label"].filter(invalid))
        )
    assert remapped.info.number_of_pixels == filtered.info.number_of_pixels
    assert (
        remapped.info.number_of_groundtruth_pixels
        == filtered.info.number_of_groundtruth_pixels
    )
    assert (
        remapped.info.number_of_prediction_pixels
        == filtered.info.number_of_prediction_pixels
    )


def test_filtered_remapping_drops_unretained_labels():
    loader = Loader.in_memory()
    loader.add_data(_segmentations())
    source = loader.finalize()
    filtered = source.filter(
        groundtruths=pc.field("gt_label") == "cat",
        predictions=pc.field("pd_label") == "bird",
    )

    assert filtered._index_to_label == {0: "cat", 2: "bird"}
    remapped = filtered.remap_labels({"cat": "animal", "bird": "animal"})
    assert remapped._index_to_label == {0: "animal"}


@pytest.mark.parametrize("mapping", [None, [], {"cat": 1}, {1: "cat"}])
def test_invalid_mapping(mapping):
    loader = Loader.in_memory()
    loader.add_data(_segmentations())
    with pytest.raises(TypeError, match="mapping must be a dict"):
        loader.finalize().remap_labels(mapping)


def test_persistent_remap_requires_a_new_path(tmp_path: Path):
    source_path = tmp_path / "source"
    loader = Loader.persistent(source_path)
    loader.add_data(_segmentations())
    source = loader.finalize()
    before = pa.concat_tables(list(source._reader.iterate_tables()))
    with pytest.raises(ValueError, match="path"):
        source.remap_labels({"cat": "animal"})
    with pytest.raises(FileExistsError):
        source.remap_labels({"cat": "animal"}, path=source_path)
    reloaded = Evaluator.load(source_path)
    assert pa.concat_tables(list(reloaded._reader.iterate_tables())).equals(
        before
    )


def test_remap_background_only():
    loader = Loader.in_memory()
    loader.add_data(
        [
            Segmentation(
                "blank",
                np.zeros((2, 2), dtype=np.uint16),
                np.zeros((2, 2), dtype=np.uint16),
                [],
            )
        ]
    )
    remapped = loader.finalize().remap_labels({"cat": "feline"})
    assert remapped._index_to_label == {}
    assert remapped.info.number_of_pixels == 4
    np.testing.assert_array_equal(
        remapped._compute_confusion_matrix_intermediate(), [[4]]
    )


def test_remap_empty_cache_raises():
    with pytest.raises(EmptyCacheError):
        Loader.in_memory().finalize()
