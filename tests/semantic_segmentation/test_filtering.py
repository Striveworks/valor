import numpy as np
import pytest

from valor_lite.exceptions import EmptyFilterError
from valor_lite.semantic_segmentation import DataLoader, Segmentation


def test_filtering(segmentations_from_boxes: list[Segmentation]):

    loader = DataLoader()
    loader.add_data(segmentations_from_boxes)
    evaluator = loader.finalize()

    assert evaluator.metadata.number_of_datums == 2
    assert evaluator.metadata.number_of_labels == 2
    assert evaluator.metadata.number_of_ground_truths == 25000
    assert evaluator.metadata.number_of_predictions == 15000
    assert evaluator.metadata.number_of_pixels == 540000

    # test datum filtering
    filter_ = evaluator.create_filter(datums=["uid1"])
    filtered_evaluator = evaluator.filter(filter_)
    confusion_matrix = filtered_evaluator._confusion_matrix
    print(confusion_matrix)
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
    filtered_evaluator = evaluator.filter(filter_)
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


def test_filtering_raises(segmentations_from_boxes: list[Segmentation]):

    loader = DataLoader()
    loader.add_data(segmentations_from_boxes)
    evaluator = loader.finalize()
    assert evaluator._confusion_matrix.shape == (3, 3)

    with pytest.raises(EmptyFilterError):
        evaluator.create_filter(datums=[])
    assert evaluator._confusion_matrix.shape == (3, 3)
