import pytest
from valor_lite.semantic_segmentation import DataLoader


def test_no_data():
    loader = DataLoader()
    with pytest.raises(ValueError):
        loader.finalize()
