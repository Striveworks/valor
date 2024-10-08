import pytest
from valor_lite.cv.segmentation import DataLoader


def test_no_data():
    loader = DataLoader()
    with pytest.raises(ValueError):
        loader.finalize()
