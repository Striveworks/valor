import numpy as np
import pytest

from valor_lite.exceptions import EmptyFilterError
from valor_lite.semantic_segmentation import (
    DataLoader,
    Filter,
    Metadata,
    Segmentation,
)


def test_filtering(segmentations_from_boxes: list[Segmentation]):

    loader = DataLoader()
    loader.add_data(segmentations_from_boxes)
    evaluator = loader.finalize()

    assert evaluator.metadata.to_dict() == {
        "number_of_datums": 2,
        "number_of_labels": 2,
        "number_of_ground_truths": 25000,
        "number_of_predictions": 15000,
        "number_of_pixels": 540000,
    }

    assert evaluator.ignored_prediction_labels == []
    assert evaluator.missing_prediction_labels == []
    assert evaluator.metadata.number_of_datums == 2
    assert (
        evaluator._label_metadata == np.array([[10000, 10000], [15000, 5000]])
    ).all()

    # test datum filtering
    filter_ = evaluator.create_filter(datums=["uid1"])
    confusion_matrices, label_metadata = evaluator.filter(filter_)
    assert np.all(
        confusion_matrices
        == np.array(
            [
                [
                    [255000, 5000, 0],
                    [5000, 5000, 0],
                    [0, 0, 0],
                ]
            ]
        )
    )
    assert np.all(label_metadata == np.array([[10000, 10000], [0, 0]]))

    filter_ = evaluator.create_filter(datums=["uid2"])
    confusion_matrices, label_metadata = evaluator.filter(filter_)
    assert np.all(
        confusion_matrices
        == np.array(
            [
                [
                    [250001, 0, 4999],
                    [0, 0, 0],
                    [14999, 0, 1],
                ]
            ]
        )
    )
    assert (label_metadata == np.array([[0, 0], [15000, 5000]])).all()

    # test label filtering
    filter_ = evaluator.create_filter(labels=["v1"])
    confusion_matrices, label_metadata = evaluator.filter(filter_)
    assert np.all(
        confusion_matrices
        == np.array(
            [
                [
                    [255000, 5000, 0],
                    [5000, 5000, 0],
                    [0, 0, 0],
                ],
                [
                    [270000, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ],
            ]
        )
    )
    assert (label_metadata == np.array([[10000, 10000], [0, 0]])).all()

    filter_ = evaluator.create_filter(labels=["v2"])
    confusion_matrices, label_metadata = evaluator.filter(filter_)
    assert np.all(
        confusion_matrices
        == np.array(
            [
                [
                    [270000, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ],
                [
                    [250001, 0, 4999],
                    [0, 0, 0],
                    [14999, 0, 1],
                ],
            ]
        )
    )
    assert (label_metadata == np.array([[0, 0], [15000, 5000]])).all()

    # test joint filtering
    filter_ = evaluator.create_filter(datums=["uid1"], labels=["v1"])
    confusion_matrices, label_metadata = evaluator.filter(filter_)
    assert np.all(
        confusion_matrices
        == np.array(
            [
                [
                    [255000, 5000, 0],
                    [5000, 5000, 0],
                    [0, 0, 0],
                ]
            ]
        )
    )
    assert (label_metadata == np.array([[10000, 10000], [0, 0]])).all()

    filter_ = evaluator.create_filter(datums=["uid1"], labels=["v2"])
    confusion_matrices, label_metadata = evaluator.filter(filter_)
    assert np.all(
        confusion_matrices
        == np.array(
            [
                [
                    [270000, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ]
            ]
        )
    )
    assert (label_metadata == np.array([[0, 0], [0, 0]])).all()

    # test filter all
    with pytest.raises(EmptyFilterError):
        evaluator.create_filter(datums=[])


def test_filtering_raises(segmentations_from_boxes: list[Segmentation]):

    loader = DataLoader()
    loader.add_data(segmentations_from_boxes)
    evaluator = loader.finalize()

    assert evaluator._confusion_matrices.shape == (2, 3, 3)

    with pytest.raises(EmptyFilterError):
        evaluator.create_filter(labels=[])

    assert evaluator._confusion_matrices.shape == (2, 3, 3)

    with pytest.raises(EmptyFilterError):
        evaluator.create_filter(datums=[])

    assert evaluator._confusion_matrices.shape == (2, 3, 3)


def test_filtering_invalid_indices(
    segmentations_from_boxes: list[Segmentation],
):

    loader = DataLoader()
    loader.add_data(segmentations_from_boxes)
    evaluator = loader.finalize()

    # test negative indices
    with pytest.raises(ValueError) as e:
        evaluator.create_filter(datums=np.array([-1]))
    assert "cannot be negative" in str(e)
    with pytest.raises(ValueError) as e:
        evaluator.create_filter(labels=np.array([-1]))
    assert "cannot be negative" in str(e)

    # test indices larger than arrays
    with pytest.raises(ValueError) as e:
        evaluator.create_filter(datums=np.array([1000]))
    assert "cannot exceed total number of datums" in str(e)
    with pytest.raises(ValueError) as e:
        evaluator.create_filter(labels=np.array([1000]))
    assert "cannot exceed total number of labels" in str(e)


def test_filter_object():

    mask = np.array([True, False, False])
    true_mask = np.array([True, True, True])
    false_mask = ~true_mask

    # check that no datums are defined
    with pytest.raises(EmptyFilterError) as e:
        Filter(datum_mask=false_mask, label_mask=mask, metadata=Metadata())
    assert "filter removes all datums" in str(e)

    # check that no labels are defined
    with pytest.raises(EmptyFilterError) as e:
        Filter(
            datum_mask=mask,
            label_mask=true_mask,
            metadata=Metadata(),
        )
