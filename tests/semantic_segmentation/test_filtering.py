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

    # test filter all
    with pytest.raises(EmptyCacheError):
        filtered_evaluator = evaluator.filter(
            datums=pc.field("datum_uid") == "non_existent_uid",
            path=tmp_path / "filtered3",
        )


def test_filtering_by_annotation_info(
    loader: Loader,
    tmp_path: Path,
    segmentations_from_boxes: list[Segmentation],
):
    loader.add_data(segmentations_from_boxes)
    evaluator = loader.finalize()

    total_pixels = 540_000
    assert evaluator.info.number_of_datums == 2
    assert evaluator.info.number_of_labels == 2
    assert evaluator.info.number_of_groundtruth_pixels == 25000
    assert evaluator.info.number_of_prediction_pixels == 15000
    assert evaluator.info.number_of_pixels == total_pixels

    # test groundtruth filtering
    filtered_evaluator = evaluator.filter(
        groundtruths=pc.field("gt_xmin") < 100,
        path=tmp_path / "gt_filter_1",
    )
    confusion_matrix = (
        filtered_evaluator._compute_confusion_matrix_intermediate()
    )
    assert np.all(
        confusion_matrix
        == np.array(
            [
                [520000, 5000, 5000],
                [5000, 5000, 0],
                [0, 0, 0],
            ]
        )
    )
    assert confusion_matrix.sum() == total_pixels

    filtered_evaluator = evaluator.filter(
        groundtruths=pc.field("gt_xmin") > 100,
        path=tmp_path / "gt_filter_2",
    )
    confusion_matrix = (
        filtered_evaluator._compute_confusion_matrix_intermediate()
    )
    assert np.all(
        confusion_matrix
        == np.array(
            [
                [510001, 10000, 4999],
                [0, 0, 0],
                [14999, 0, 1],
            ]
        )
    )
    assert confusion_matrix.sum() == total_pixels

    # test prediction filtering
    filtered_evaluator = evaluator.filter(
        predictions=pc.field("pd_xmin") < 100,
        path=tmp_path / "pd_filter_1",
    )
    confusion_matrix = (
        filtered_evaluator._compute_confusion_matrix_intermediate()
    )
    assert np.all(
        confusion_matrix
        == np.array(
            [
                [510000, 5000, 0],
                [5000, 5000, 0],
                [15000, 0, 0],
            ]
        )
    )
    assert confusion_matrix.sum() == total_pixels

    filtered_evaluator = evaluator.filter(
        predictions=pc.field("pd_xmin") > 100,
        path=tmp_path / "pd_filter_2",
    )
    confusion_matrix = (
        filtered_evaluator._compute_confusion_matrix_intermediate()
    )
    assert np.all(
        confusion_matrix
        == np.array(
            [
                [510001, 0, 4999],
                [10000, 0, 0],
                [14999, 0, 1],
            ]
        )
    )
    assert confusion_matrix.sum() == total_pixels

    # filter out all gts and pds
    filtered_evaluator = evaluator.filter(
        groundtruths=pc.field("gt_xmin") > 1000,
        predictions=pc.field("pd_xmin") > 1000,
        path=tmp_path / "joint_filter",
    )
    confusion_matrix = (
        filtered_evaluator._compute_confusion_matrix_intermediate()
    )
    assert np.all(
        confusion_matrix
        == np.array(
            [
                [total_pixels, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]
        )
    )
    assert confusion_matrix.sum() == total_pixels


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

    assert filtered._index_to_label == {
        0: "v1",
        1: "v2",
        2: "v3",
    }
    assert filtered.compute_precision_recall_iou()

    metrics = filtered.compute_precision_recall_iou()
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
            "unmatched_predictions": {
                "v3": {
                    "ratio": 1.0,
                },
            },
        },
    }
