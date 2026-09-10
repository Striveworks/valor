from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from valor_lite.cache import FileCacheReader
from valor_lite.semantic_segmentation import (
    Bitmask,
    Evaluator,
    Loader,
    Segmentation,
)


@pytest.fixture
def remapping_segmentations():
    labels = ["cat", "dog", "bird"]
    segmentations = []
    for datum, (groundtruths, predictions) in enumerate(
        [
            ([0, 0, 1, 1, 2, -1, -1, 0], [1, 0, 0, 1, 2, 0, -1, -1]),
            ([0, 0, 0, 0, 0, 0, 0, 0], [-1] * 8),
            ([-1] * 8, [1, 1, 1, 1, 1, 1, 1, 1]),
            ([-1] * 8, [-1] * 8),
        ]
    ):
        annotations = []
        for pixels in (groundtruths, predictions):
            pixels = np.array(pixels).reshape(2, 4)
            annotations.append(
                [
                    Bitmask(mask=pixels == index, label=label)
                    for index, label in enumerate(labels)
                    if (pixels == index).any()
                ]
            )
        segmentations.append(
            Segmentation(
                uid=f"datum{datum}",
                groundtruths=annotations[0],
                predictions=annotations[1],
                shape=(2, 4),
                metadata={"gt_xmin": float(datum), "pd_xmin": float(datum)},
            )
        )
    return segmentations


def _table(reader):
    return pa.concat_tables(list(reader.iterate_tables()))


def _assert_metrics_equal(actual, expected, datums=None):
    actual_metrics = actual.compute_precision_recall_iou(datums=datums)
    expected_metrics = expected.compute_precision_recall_iou(datums=datums)
    assert actual_metrics.keys() == expected_metrics.keys()
    for metric_type, metrics in actual_metrics.items():
        actual_values = sorted(metrics, key=lambda m: str(m.parameters))
        expected_values = sorted(
            expected_metrics[metric_type], key=lambda m: str(m.parameters)
        )
        assert len(actual_values) == len(expected_values)
        for actual_value, expected_value in zip(
            actual_values, expected_values
        ):
            assert actual_value.parameters == expected_value.parameters
            if isinstance(expected_value.value, float):
                assert actual_value.value == pytest.approx(
                    expected_value.value
                )
            else:
                assert actual_value.value == expected_value.value


def _remap_masks(bitmasks, mapping):
    masks = {}
    for bitmask in bitmasks:
        label = mapping.get(bitmask.label, bitmask.label)
        if label in masks:
            masks[label] |= bitmask.mask
        else:
            masks[label] = bitmask.mask.copy()
    return [Bitmask(mask=mask, label=label) for label, mask in masks.items()]


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
    loader: Loader, tmp_path: Path, remapping_segmentations, mapping
):
    loader.add_data(remapping_segmentations)
    source = loader.finalize()
    before = _table(source._reader)
    vocabulary_before = source._index_to_label.copy()
    mapping_before = mapping.copy()
    destination = tmp_path / "remapped"
    remapped = source.remap_labels(mapping, path=destination)

    if isinstance(source._reader, FileCacheReader):
        fresh_loader = Loader.persistent(
            path=tmp_path / "fresh",
            batch_size=source._reader.batch_size,
            rows_per_file=source._reader.rows_per_file,
            compression=source._reader.compression,
            metadata_fields=source._metadata_fields,
        )
        assert isinstance(remapped._reader, FileCacheReader)
        assert remapped._reader.path == destination / "counts"
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

    fresh_loader.add_data(
        [
            Segmentation(
                uid=segmentation.uid,
                groundtruths=_remap_masks(segmentation.groundtruths, mapping),
                predictions=_remap_masks(segmentation.predictions, mapping),
                shape=segmentation.shape,
                metadata=segmentation.metadata,
            )
            for segmentation in remapping_segmentations
        ]
    )
    fresh = fresh_loader.finalize()
    _assert_metrics_equal(remapped, fresh)
    for datum in remapping_segmentations:
        _assert_metrics_equal(
            remapped, fresh, datums=pc.field("datum_uid") == datum.uid
        )

    # Cache rows stay separate to retain metadata; fresh masks are unioned.
    assert remapped.info.number_of_rows == source.info.number_of_rows
    for field in (
        "number_of_datums",
        "number_of_pixels",
        "number_of_groundtruth_pixels",
        "number_of_prediction_pixels",
        "number_of_labels",
        "metadata_fields",
    ):
        assert getattr(remapped.info, field) == getattr(fresh.info, field)
    after = _table(remapped._reader)
    for column in ("datum_uid", "datum_id", "count", "gt_xmin", "pd_xmin"):
        assert after[column].equals(before[column])
    for side in ("gt", "pd"):
        assert pc.equal(after[f"{side}_label_id"], -1).equals(
            pc.equal(before[f"{side}_label_id"], -1)
        )
    assert after.schema == before.schema
    assert remapped is not source
    assert set(remapped._index_to_label.values()) == {
        mapping.get(label, label) for label in vocabulary_before.values()
    }
    assert mapping == mapping_before
    assert source._index_to_label == vocabulary_before
    assert _table(source._reader).equals(before)

    if isinstance(source._reader, FileCacheReader):
        reloaded = Evaluator.load(destination)
        _assert_metrics_equal(reloaded, remapped)
        assert reloaded.info == remapped.info


def test_remap_sums_pixel_counts(
    loader: Loader, tmp_path: Path, remapping_segmentations
):
    loader.add_data(remapping_segmentations)
    source = loader.finalize()
    remapped = source.remap_labels(
        {"cat": "animal", "dog": "animal"}, path=tmp_path / "remapped"
    )
    assert remapped._index_to_label == {0: "animal", 1: "bird"}
    actual = remapped._compute_confusion_matrix_intermediate(
        datums=pc.field("datum_uid") == "datum0"
    )
    # Background, animal, bird. Cross-label errors become animal matches.
    np.testing.assert_array_equal(actual, [[1, 1, 0], [1, 4, 0], [0, 0, 1]])
    assert remapped.info.number_of_pixels == 32
    assert remapped.info.number_of_groundtruth_pixels == 14
    assert remapped.info.number_of_prediction_pixels == 14


def test_remap_filtered_cache(
    loader: Loader, tmp_path: Path, remapping_segmentations
):
    loader.add_data(remapping_segmentations)
    source = loader.finalize()
    filtered = source.filter(
        groundtruths=pc.field("gt_label") != "cat",
        predictions=pc.field("pd_label") != "dog",
        path=tmp_path / "filtered",
    )
    remapped = filtered.remap_labels(
        {"cat": "animal", "dog": "animal"}, path=tmp_path / "remapped"
    )
    before = _table(filtered._reader)
    after = _table(remapped._reader)
    for side in ("gt", "pd"):
        invalid = pc.equal(before[f"{side}_label_id"], -1)
        assert invalid.equals(pc.equal(after[f"{side}_label_id"], -1))
        assert (
            before[f"{side}_label"]
            .filter(invalid)
            .equals(after[f"{side}_label"].filter(invalid))
        )
    for field in (
        "number_of_pixels",
        "number_of_groundtruth_pixels",
        "number_of_prediction_pixels",
    ):
        assert getattr(remapped.info, field) == getattr(filtered.info, field)

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

    # Filtering out datums leaves empty fragments in persistent caches.
    subset = source.filter(
        datums=pc.field("datum_uid") == "datum1", path=tmp_path / "subset"
    ).remap_labels({"bird": "avian"}, path=tmp_path / "subset_remapped")
    assert set(subset._index_to_label.values()) == {"cat", "dog", "avian"}
    assert subset.info.number_of_datums == 1
    assert subset.info.number_of_pixels == 8


def test_remap_preserves_annotation_metadata(loader: Loader, tmp_path: Path):
    loader.add_data(
        [
            Segmentation(
                "datum",
                [
                    Bitmask(
                        np.array([[True, False]]), "cat", {"gt_xmin": 1.0}
                    ),
                    Bitmask(
                        np.array([[False, True]]), "dog", {"gt_xmin": 2.0}
                    ),
                ],
                [Bitmask(np.array([[True, True]]), "bird", {"pd_xmin": 3.0})],
                shape=(1, 2),
            )
        ]
    )
    remapped = loader.finalize().remap_labels(
        {"cat": "animal", "dog": "animal"}, path=tmp_path / "remapped"
    )
    for value in (1.0, 2.0):
        info = remapped.get_info(
            groundtruths=(pc.field("gt_label") == "animal")
            & (pc.field("gt_xmin") == value)
        )
        assert info.number_of_groundtruth_pixels == 1


def test_remap_zero_count_label(loader: Loader, tmp_path: Path):
    loader.add_data(
        [
            Segmentation(
                "d", [Bitmask(np.zeros((2, 2), dtype=bool), "cat")], [], (2, 2)
            )
        ]
    )
    remapped = loader.finalize().remap_labels(
        {"cat": "feline"}, path=tmp_path / "remapped"
    )
    assert remapped._index_to_label == {0: "feline"}
    assert remapped.info.number_of_pixels == 4
    assert remapped.info.number_of_groundtruth_pixels == 0
    assert remapped.info.number_of_prediction_pixels == 0


def test_remap_background_only(loader: Loader, tmp_path: Path):
    loader.add_data([Segmentation("d", [], [], (2, 2))])
    remapped = loader.finalize().remap_labels(
        {"cat": "feline"}, path=tmp_path / "remapped"
    )
    assert remapped._index_to_label == {}
    assert remapped.info.number_of_pixels == 4
    np.testing.assert_array_equal(
        remapped._compute_confusion_matrix_intermediate(), [[4]]
    )


@pytest.mark.parametrize("mapping", [None, [], {"cat": 1}, {1: "cat"}])
def test_invalid_mapping(mapping):
    loader = Loader.in_memory()
    loader.add_data([Segmentation("d", [], [], (2, 2))])
    with pytest.raises(TypeError, match="mapping must be a dict"):
        loader.finalize().remap_labels(mapping)


def test_persistent_remap_requires_new_path(tmp_path: Path):
    source_path = tmp_path / "source"
    loader = Loader.persistent(source_path)
    loader.add_data([Segmentation("d", [], [], (2, 2))])
    source = loader.finalize()
    before = _table(source._reader)
    with pytest.raises(ValueError, match="path"):
        source.remap_labels({"cat": "dog"})
    with pytest.raises(FileExistsError):
        source.remap_labels({"cat": "dog"}, path=source_path)
    assert _table(Evaluator.load(source_path)._reader).equals(before)
