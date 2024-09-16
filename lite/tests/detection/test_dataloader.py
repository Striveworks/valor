import pytest
from valor_lite.detection import DataLoader


def test_no_data():
    loader = DataLoader()
    with pytest.raises(ValueError):
        loader.finalize()


def test_valor_integration():
    pass
