from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pytest

from valor_lite.exceptions import EmptyCacheError
from valor_lite.semantic_segmentation import Loader, Segmentation


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
            ]
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
