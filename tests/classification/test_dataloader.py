from pathlib import Path

import pytest

from valor_lite.classification import Classification, Loader
from valor_lite.exceptions import EmptyCacheError


def test_loader_no_data(tmp_path: Path):
    loader = Loader.persistent(tmp_path)
    with pytest.raises(EmptyCacheError):
        loader.finalize()


def test_finalization_no_data(tmp_path: Path):
    loader = Loader.persistent(tmp_path)
    with pytest.raises(EmptyCacheError):
        loader.finalize()


def test_unmatched_ground_truths(
    tmp_path: Path,
    classifications_no_predictions: list[Classification],
):
    loader = Loader.persistent(tmp_path)

    with pytest.raises(ValueError) as e:
        loader.add_data(classifications_no_predictions)
    assert "Classifications must contain at least one prediction" in str(e)


def test_evaluator_deletion(
    tmp_path: Path, basic_classifications: list[Classification]
):
    loader = Loader.persistent(tmp_path)
    loader.add_data(basic_classifications)
    assert tmp_path == loader._path

    # check both caches exist
    assert tmp_path.exists()
    assert loader._generate_cache_path(tmp_path).exists()
    assert loader._generate_metadata_path(tmp_path).exists()

    # verify deletion
    loader.delete(loader._path)
    assert not tmp_path.exists()
    assert not loader._generate_cache_path(tmp_path).exists()
    assert not loader._generate_metadata_path(tmp_path).exists()
