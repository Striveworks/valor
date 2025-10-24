from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pytest

from valor_lite.cache import DataType
from valor_lite.exceptions import EmptyFilterError
from valor_lite.semantic_segmentation import DataLoader, Segmentation
from valor_lite.semantic_segmentation.evaluator import Filter


def test_filtering_raises(
    tmp_path: Path,
    segmentations_from_boxes: list[Segmentation],
):

    loader = DataLoader.create(tmp_path)
    loader.add_data(segmentations_from_boxes)
    evaluator = loader.finalize()
    assert evaluator._confusion_matrix.shape == (3, 3)

    with pytest.raises(EmptyFilterError):
        evaluator.create_filter(datums=[])
    assert evaluator._confusion_matrix.shape == (3, 3)


def test_filtering_by_datum(
    tmp_path: Path,
    segmentations_from_boxes: list[Segmentation],
):

    loader = DataLoader.create(tmp_path)
    loader.add_data(segmentations_from_boxes)
    evaluator = loader.finalize()

    assert evaluator.metadata.number_of_datums == 2
    assert evaluator.metadata.number_of_labels == 2
    assert evaluator.metadata.number_of_ground_truths == 25000
    assert evaluator.metadata.number_of_predictions == 15000
    assert evaluator.metadata.number_of_pixels == 540000

    # test datum filtering
    filter_ = evaluator.create_filter(datums=["uid1"])
    filtered_evaluator = evaluator.filter(tmp_path / "uid1", filter_)
    confusion_matrix = filtered_evaluator._confusion_matrix
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

    filter_ = evaluator.create_filter(datums=["uid2"])
    filtered_evaluator = evaluator.filter(tmp_path / "uid2", filter_)
    confusion_matrix = filtered_evaluator._confusion_matrix
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
    with pytest.raises(EmptyFilterError):
        evaluator.create_filter(datums=[])


def test_filtering_by_annotation_metadata(
    tmp_path: Path,
    segmentations_from_boxes: list[Segmentation],
):

    loader = DataLoader.create(
        tmp_path,
        groundtruth_metadata_types={
            "gt_xmin": DataType.FLOAT,
        },
        prediction_metadata_types={
            "pd_xmin": DataType.FLOAT,
        },
    )
    loader.add_data(segmentations_from_boxes)
    evaluator = loader.finalize()

    total_pixels = 540_000
    assert evaluator.metadata.number_of_datums == 2
    assert evaluator.metadata.number_of_labels == 2
    assert evaluator.metadata.number_of_ground_truths == 25000
    assert evaluator.metadata.number_of_predictions == 15000
    assert evaluator.metadata.number_of_pixels == total_pixels

    # test groundtruth filtering
    filter_ = Filter(groundtruths=pc.field("gt_xmin") < 100)
    filtered_evaluator = evaluator.filter(tmp_path / "gt_filter_1", filter_)
    confusion_matrix = filtered_evaluator._confusion_matrix
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

    filter_ = Filter(groundtruths=pc.field("gt_xmin") > 100)
    filtered_evaluator = evaluator.filter(tmp_path / "gt_filter_2", filter_)
    confusion_matrix = filtered_evaluator._confusion_matrix
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
    filter_ = Filter(predictions=pc.field("pd_xmin") < 100)
    filtered_evaluator = evaluator.filter(tmp_path / "pd_filter_1", filter_)
    confusion_matrix = filtered_evaluator._confusion_matrix
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

    filter_ = Filter(predictions=pc.field("pd_xmin") > 100)
    filtered_evaluator = evaluator.filter(tmp_path / "pd_filter_2", filter_)
    confusion_matrix = filtered_evaluator._confusion_matrix
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
    filter_ = Filter(
        groundtruths=pc.field("gt_xmin") > 1000,
        predictions=pc.field("pd_xmin") > 1000,
    )
    filtered_evaluator = evaluator.filter(tmp_path / "joint_filter", filter_)
    confusion_matrix = filtered_evaluator._confusion_matrix
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
