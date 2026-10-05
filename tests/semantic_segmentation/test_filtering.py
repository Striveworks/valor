from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pytest

from valor_lite.exceptions import EmptyCacheError
from valor_lite.semantic_segmentation import Loader, MetricType, Segmentation


def prune_fields_containing_zeros(data: dict | list):
    if isinstance(data, list):
        for element in data:
            prune_fields_containing_zeros(element)
    elif isinstance(data, dict):
        for key in list(data.keys()):
            if isinstance(data[key], dict):
                prune_fields_containing_zeros(data[key])
                if len(data[key]) == 0:
                    data.pop(key)
            elif data[key] == 0:
                data.pop(key)
    return data


def test_filtering_by_datum(
    loader: Loader,
    tmp_path: Path,
    segmentations_from_boxes: list[Segmentation],
):
    loader.add_data(segmentations_from_boxes)
    evaluator = loader.finalize()

    assert evaluator.info.number_of_datums == 2
    assert evaluator.info.number_of_labels == 2
    assert evaluator.info.number_of_groundtruth_pixels == 25000
    assert evaluator.info.number_of_prediction_pixels == 15000
    assert evaluator.info.number_of_pixels == 540000

    # test datum filtering
    confusion_matrix = evaluator._compute_confusion_matrix_intermediate(
        datums=pc.field("datum_uid") == "uid1",
    )
    assert np.all(
        confusion_matrix
        == np.array(
            [
                [255000, 5000, 0],
                [5000, 5000, 0],
                [0, 0, 0],
            ],
        )
    )

    # test filter cache and evaluate
    filtered_evaluator = evaluator.filter(
        datums=pc.field("datum_uid") == "uid1",
        path=tmp_path / "filtered1",
    )
    confusion_matrix = (
        filtered_evaluator._compute_confusion_matrix_intermediate()
    )
    assert np.all(
        confusion_matrix
        == np.array(
            [
                [255000, 5000, 0],
                [5000, 5000, 0],
                [0, 0, 0],
            ],
        )
    )

    assert filtered_evaluator.info.number_of_datums == 1
    assert filtered_evaluator.info.number_of_pixels == 270000

    filtered_evaluator = evaluator.filter(
        datums=pc.field("datum_uid") == "uid2",
        path=tmp_path / "filtered2",
    )
    confusion_matrix = (
        filtered_evaluator._compute_confusion_matrix_intermediate()
    )
    assert np.all(
        confusion_matrix
        == np.array(
            [
                [250001, 0, 4999],
                [0, 0, 0],
                [14999, 0, 1],
            ]
        )
    )

    assert filtered_evaluator.info.number_of_datums == 1
    assert filtered_evaluator.info.number_of_pixels == 270000
    np.testing.assert_array_equal(
        evaluator._compute_confusion_matrix_intermediate(
            datums=pc.field("datum_uid") == "uid2"
        ),
        confusion_matrix,
    )

    # test filter all
    with pytest.raises(EmptyCacheError):
        filtered_evaluator = evaluator.filter(
            datums=pc.field("datum_uid") == "non_existent_uid",
            path=tmp_path / "filtered3",
        )


def test_filtering_by_metadata(loader: Loader, tmp_path: Path):
    loader.add_data(
        [
            Segmentation(
                "image",
                np.array([[1, 2], [2, 3]]),
                np.array([[2, 2], [3, 3]]),
                ["sky", "road", "car"],
                metadata={"split": "validation"},
            )
        ]
    )
    evaluator = loader.finalize()
    filtered = evaluator.filter(
        datums=pc.field("split") == "validation", path=tmp_path / "filtered"
    )
    assert filtered.info.number_of_datums == 1
    assert filtered.info.number_of_pixels == 4
    assert (
        evaluator.get_info(datums=pc.field("split") == "validation")
        == filtered.info
    )


def test_filtering_all_annotations(
    loader: Loader, tmp_path: Path, basic_segmentations
):
    loader.add_data(basic_segmentations)
    evaluator = loader.finalize()
    gt = pc.field("gt_label") == "missing"
    pd = pc.field("pd_label") == "missing"
    with pytest.raises(EmptyCacheError):
        evaluator.filter(
            groundtruths=gt, predictions=pd, path=tmp_path / "filtered"
        )
    info = evaluator.get_info(groundtruths=gt, predictions=pd)
    assert (
        info.number_of_rows
        == info.number_of_labels
        == info.number_of_datums
        == 0
    )
    assert (
        info.number_of_groundtruth_pixels
        == info.number_of_prediction_pixels
        == info.number_of_pixels
        == 0
    )


def test_filtering_labels(
    loader: Loader,
    basic_segmentations_three_labels: list[Segmentation],
    tmp_path: Path,
):
    loader.add_data(basic_segmentations_three_labels)
    evaluator = loader.finalize()

    assert evaluator._index_to_label == {
        0: "v1",
        1: "v2",
        2: "v3",
    }
    assert evaluator.compute_precision_recall_iou()

    metrics = evaluator.compute_precision_recall_iou()
    cm = metrics.pop(MetricType.ConfusionMatrix)
    assert len(cm) == 1
    assert prune_fields_containing_zeros(cm[0].to_dict()) == {
        "type": "ConfusionMatrix",
        "value": {
            "confusion_matrix": {
                "v1": {
                    "v1": {
                        "iou": 0.5,
                    },
                    "v3": {
                        "iou": 0.5,
                    },
                },
                "v2": {
                    "v2": {
                        "iou": 0.5,
                    },
                },
                "v3": {
                    "v2": {
                        "iou": 0.5,
                    },
                },
            },
        },
    }

    filtered = evaluator.filter(
        groundtruths=pc.field("gt_label").isin(["v2", "v3"]),
        predictions=pc.field("pd_label").isin(["v2", "v3"]),
        path=tmp_path / "filter",
    )

    assert filtered._index_to_label == {1: "v2", 2: "v3"}
    assert (
        filtered.info.number_of_pixels == 9
    )  # Three pairs per image; both-failed pairs excluded.
    assert (
        filtered.compute_precision_recall_iou()[MetricType.Accuracy][0].value
        == 1 / 3
    )
    assert filtered.compute_precision_recall_iou()

    metrics = filtered.compute_precision_recall_iou()
    assert metrics[MetricType.mIOU][0].value == 0.25
    assert [
        metric.parameters["label"] for metric in metrics[MetricType.IOU]
    ] == [
        "v2",
        "v3",
    ]
    cm = metrics.pop(MetricType.ConfusionMatrix)
    assert len(cm) == 1
    assert prune_fields_containing_zeros(cm[0].to_dict()) == {
        "type": "ConfusionMatrix",
        "value": {
            "confusion_matrix": {
                "v2": {
                    "v2": {
                        "iou": 0.5,
                    },
                },
                "v3": {
                    "v2": {
                        "iou": 0.5,
                    },
                },
            },
            "unmatched_predictions": {"v3": {"ratio": 1.0}},
        },
    }


@pytest.mark.parametrize("side", ["groundtruths", "predictions"])
def test_single_filter_retains_remaining_side(
    loader: Loader, tmp_path: Path, basic_segmentations, side
):
    loader.add_data(basic_segmentations)
    evaluator = loader.finalize()
    column = "gt_label" if side == "groundtruths" else "pd_label"
    kwargs = {side: pc.field(column) == "missing"}
    filtered = evaluator.filter(path=tmp_path / "single", **kwargs)
    assert filtered.info.number_of_pixels == 4
    assert evaluator.get_info(**kwargs) == filtered.info
    assert filtered.info.number_of_groundtruth_pixels == (
        0 if side == "groundtruths" else 3
    )
    assert filtered.info.number_of_prediction_pixels == (
        0 if side == "predictions" else 3
    )
    assert (
        filtered.compute_precision_recall_iou()[MetricType.Accuracy][0].value
        == 0.25
    )


def test_filtering_to_absent_class_does_not_report_perfect_accuracy(
    loader: Loader, tmp_path: Path
):
    loader.add_data(
        [
            Segmentation(
                "image", np.array([[1]]), np.array([[1]]), ["sky", "absent"]
            )
        ]
    )
    filtered = loader.finalize().filter(
        groundtruths=pc.field("gt_label") == "absent",
        predictions=pc.field("pd_label") == "absent",
        path=tmp_path / "absent",
    )
    assert filtered._index_to_label == {1: "absent"}
    assert filtered.info.number_of_pixels == 0
    assert (
        filtered.compute_precision_recall_iou()[MetricType.Accuracy][0].value
        == 0
    )
    assert (
        filtered.compute_precision_recall_iou()[MetricType.mIOU][0].value == 0
    )


@pytest.mark.parametrize("side", ["groundtruths", "predictions", "both"])
def test_ignored_class_uses_background(loader: Loader, tmp_path: Path, side):
    loader.add_data(
        [
            Segmentation(
                "image",
                np.array([[1, 2, 2, 2, 1]]),
                np.array([[2, 1, 2, 2, 1]]),
                ["cat", "dog"],
            )
        ]
    )
    evaluator = loader.finalize()
    filters = {}
    if side in ("groundtruths", "both"):
        filters["groundtruths"] = pc.field("gt_label") != "cat"
    if side in ("predictions", "both"):
        filters["predictions"] = pc.field("pd_label") != "cat"
    filtered = evaluator.filter(path=tmp_path / "filtered", **filters)
    assert evaluator.get_info(**filters) == filtered.info
    metrics = filtered.compute_precision_recall_iou()
    assert filtered._index_to_label == (
        {1: "dog"} if side == "both" else {0: "cat", 1: "dog"}
    )
    assert [
        metric.parameters["label"] for metric in metrics[MetricType.IOU]
    ] == (["dog"] if side == "both" else ["cat", "dog"])
    assert metrics[MetricType.mIOU][0].value == (
        0.5 if side == "both" else 0.25
    )
    assert metrics[MetricType.Accuracy][0].value == (
        0.5 if side == "both" else 0.4
    )
    if side == "both":
        np.testing.assert_array_equal(
            filtered._compute_confusion_matrix_intermediate(), [[0, 1], [1, 2]]
        )
        assert metrics[MetricType.Precision][0].value == 2 / 3
        assert metrics[MetricType.Recall][0].value == 2 / 3
    # Filtering a copy does not change the source's classes or metrics.
    assert evaluator._index_to_label == {0: "cat", 1: "dog"}
    assert (
        evaluator.compute_precision_recall_iou()[MetricType.Accuracy][0].value
        == 0.6
    )


@pytest.mark.parametrize("side", ["groundtruths", "predictions"])
def test_exclusions_match_background(loader: Loader, tmp_path: Path, side):
    foreground = np.array([[1]])
    background = np.array([[0]])
    loader.add_data(
        [
            Segmentation(
                "image",
                foreground if side == "groundtruths" else background,
                foreground if side == "predictions" else background,
                ["cat"],
            )
        ]
    )
    evaluator = loader.finalize()
    assert (
        evaluator.compute_precision_recall_iou()[MetricType.Accuracy][0].value
        == 0.0
    )
    column = "gt_label" if side == "groundtruths" else "pd_label"
    filters = {side: pc.field(column) != "cat"}
    filtered = evaluator.filter(path=tmp_path / "background", **filters)
    np.testing.assert_array_equal(
        filtered._compute_confusion_matrix_intermediate(), [[1]]
    )
    assert filtered._index_to_label == {}
    assert filtered.info.number_of_pixels == 1
    assert evaluator.get_info(**filters) == filtered.info
    assert (
        filtered.compute_precision_recall_iou()[MetricType.Accuracy][0].value
        == 1.0
    )
    assert (
        filtered.compute_precision_recall_iou()[MetricType.mIOU][0].value
        == 0.0
    )
