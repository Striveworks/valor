import pytest
from valor_lite.segmentation import DataLoader


def test_no_data():
    loader = DataLoader()
    with pytest.raises(ValueError):
        loader.finalize()
