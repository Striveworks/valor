import numpy as np
import pytest

from valor_lite.exceptions import EmptyCacheError
from valor_lite.semantic_segmentation import DataLoader, Segmentation


def test_no_data():
    loader = DataLoader()
    with pytest.raises(EmptyCacheError):
        loader.finalize()


def test_empty_input():
    loader = DataLoader()
    loader.add_data(
        segmentations=[
            Segmentation(
                uid="0", groundtruths=[], predictions=[], shape=(10, 10)
            ),
        ]
    )
    evaluator = loader.finalize()
    assert evaluator._confusion_matrix.shape == (1, 1)
    assert (evaluator._confusion_matrix == np.array([100])).all()
