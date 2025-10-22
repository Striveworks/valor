import pytest

from valor_lite.exceptions import EmptyCacheError
from valor_lite.semantic_segmentation import DataLoader, Segmentation


def test_no_data():
    loader = DataLoader()
    with pytest.raises(EmptyCacheError):
        loader.finalize()


def test_datum_already_exists():

    loader = DataLoader()
    with pytest.raises(ValueError) as e:
        loader.add_data(
            segmentations=[
                Segmentation(
                    uid="0", groundtruths=[], predictions=[], shape=(10, 10)
                ),
                Segmentation(
                    uid="0", groundtruths=[], predictions=[], shape=(10, 10)
                ),
            ]
        )
    assert "already exists" in str(e)
