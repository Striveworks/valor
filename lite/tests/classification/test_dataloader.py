import pytest
from valor_lite.classification import Classification, DataLoader


def test_no_data():
    loader = DataLoader()
    with pytest.raises(ValueError):
        loader.finalize()


def test_missing_predictions(
    classifications_no_predictions: list[Classification],
):
    loader = DataLoader()

    with pytest.raises(ValueError) as e:
        loader.add_data(classifications_no_predictions)
    assert "Classifications must contain at least one prediction" in str(e)
