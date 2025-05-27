import numpy as np
import pytest

from valor_lite.semantic_segmentation import DataLoader, Segmentation


def test_filtering(segmentations_from_boxes: list[Segmentation]):

    loader = DataLoader()
    loader.add_data(segmentations_from_boxes)
    evaluator = loader.finalize()

    assert evaluator.metadata == {
        "ignored_prediction_labels": [],
        "missing_prediction_labels": [],
        "number_of_datums": 2,
        "number_of_labels": 2,
        "number_of_groundtruths": 25000,
        "number_of_predictions": 15000,
        "number_of_pixels": 540000,
        "is_filtered": False,
    }

    assert evaluator.n_datums == 2
    assert (
        evaluator._label_metadata == np.array([[10000, 10000], [15000, 5000]])
    ).all()

    # test datum filtering
    evaluator.apply_filter(datum_ids=["uid1"])
    assert np.all(
        evaluator.confusion_matrices
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
    assert np.all(
        evaluator.label_metadata == np.array([[10000, 10000], [0, 0]])
    )

    evaluator.apply_filter(datum_ids=["uid2"])
    assert np.all(
        evaluator.confusion_matrices
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
    assert (
        evaluator.label_metadata == np.array([[0, 0], [15000, 5000]])
    ).all()

    # test label filtering
    evaluator.apply_filter(labels=["v1"])
    assert np.all(
        evaluator.confusion_matrices
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
    assert (
        evaluator.label_metadata == np.array([[10000, 10000], [0, 0]])
    ).all()

    evaluator.apply_filter(labels=["v2"])
    assert np.all(
        evaluator.confusion_matrices
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
    assert (
        evaluator.label_metadata == np.array([[0, 0], [15000, 5000]])
    ).all()

    # test joint filtering
    evaluator.apply_filter(datum_ids=["uid1"], labels=["v1"])
    assert np.all(
        evaluator.confusion_matrices
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
    assert (
        evaluator.label_metadata == np.array([[10000, 10000], [0, 0]])
    ).all()

    evaluator.apply_filter(datum_ids=["uid1"], labels=["v2"])
    assert np.all(
        evaluator.confusion_matrices
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
    assert (evaluator.label_metadata == np.array([[0, 0], [0, 0]])).all()

    # test filter all
    with pytest.warns(UserWarning):
        evaluator.apply_filter(datum_ids=[])
    assert np.all(evaluator.confusion_matrices == np.array([]))
    assert (evaluator.label_metadata == np.array([[0, 0], [0, 0]])).all()


def test_filtering_warning(segmentations_from_boxes: list[Segmentation]):

    loader = DataLoader()
    loader.add_data(segmentations_from_boxes)
    evaluator = loader.finalize()

    assert evaluator.confusion_matrices.shape == (2, 3, 3)

    with pytest.warns():
        evaluator.apply_filter(labels=[])
    assert evaluator.confusion_matrices.shape == (0,)

    evaluator.clear_filter()
    assert evaluator.confusion_matrices.shape == (2, 3, 3)

    with pytest.warns():
        evaluator.apply_filter(datum_ids=[])
    assert evaluator.confusion_matrices.shape == (0,)

    evaluator.clear_filter()
    assert evaluator.confusion_matrices.shape == (2, 3, 3)
