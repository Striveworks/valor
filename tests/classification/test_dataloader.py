import pytest

from valor_lite.classification import Classification, DataLoader
from valor_lite.exceptions import EmptyEvaluatorError


def test_finalization_no_data():
    loader = DataLoader()
    with pytest.raises(EmptyEvaluatorError):
        loader.finalize()


def test_unmatched_ground_truths(
    classifications_no_predictions: list[Classification],
):
    loader = DataLoader()

    with pytest.raises(ValueError) as e:
        loader.add_data(classifications_no_predictions)
    assert "Classifications must contain at least one prediction" in str(e)
