import numpy as np
import pytest

from valor_lite.semantic_segmentation import DataLoader, Segmentation


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
        "is_filtered": False,
    }

    assert evaluator.ignored_prediction_labels == []
    assert evaluator.missing_prediction_labels == []
    assert evaluator.metadata.number_of_datums == 2
    assert (
        evaluator._label_metadata == np.array([[10000, 10000], [15000, 5000]])
    ).all()

    # test datum filtering
    filter_ = evaluator.create_filter(datum_ids=["uid1"])
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

    filter_ = evaluator.create_filter(datum_ids=["uid2"])
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
    filter_ = evaluator.create_filter(datum_ids=["uid1"], labels=["v1"])
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

    filter_ = evaluator.create_filter(datum_ids=["uid1"], labels=["v2"])
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
    with pytest.warns(UserWarning):
        filter_ = evaluator.create_filter(datum_ids=[])
    with pytest.warns(UserWarning):
        confusion_matrices, label_metadata = evaluator.filter(filter_)
    assert np.all(confusion_matrices == np.array([]))
    assert (label_metadata == np.array([[0, 0], [0, 0]])).all()


def test_filtering_warning(segmentations_from_boxes: list[Segmentation]):

    loader = DataLoader()
    loader.add_data(segmentations_from_boxes)
    evaluator = loader.finalize()

    assert evaluator._confusion_matrices.shape == (2, 3, 3)

    with pytest.warns():
        filter_ = evaluator.create_filter(labels=[])
    with pytest.warns(UserWarning):
        confusion_matrices, _ = evaluator.filter(filter_)
    assert confusion_matrices.shape == (0,)

    assert evaluator._confusion_matrices.shape == (2, 3, 3)

    with pytest.warns():
        filter_ = evaluator.create_filter(datum_ids=[])
    with pytest.warns(UserWarning):
        confusion_matrices, _ = evaluator.filter(filter_)
    assert confusion_matrices.shape == (0,)

    assert evaluator._confusion_matrices.shape == (2, 3, 3)
