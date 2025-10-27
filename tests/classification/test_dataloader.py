from pathlib import Path

import pytest

from valor_lite.classification import Classification, DataLoader
from valor_lite.exceptions import EmptyCacheError


def test_finalization_no_data(tmp_path: Path):
    loader = DataLoader.create(tmp_path)
    with pytest.raises(EmptyCacheError):
        loader.finalize()


def test_unmatched_ground_truths(
    tmp_path: Path,
    classifications_no_predictions: list[Classification],
):
    loader = DataLoader.create(tmp_path)

    with pytest.raises(ValueError) as e:
        loader.add_data(classifications_no_predictions)
    assert "Classifications must contain at least one prediction" in str(e)
