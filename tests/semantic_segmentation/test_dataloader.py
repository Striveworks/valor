import json
from pathlib import Path

import numpy as np
import pytest

from valor_lite.exceptions import EmptyCacheError
from valor_lite.semantic_segmentation import DataLoader, Segmentation


def test_no_data(tmp_path: Path):
    loader = DataLoader.create(tmp_path)
    with pytest.raises(EmptyCacheError):
        loader.finalize()


def test_loader_file_not_found(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        DataLoader.load(tmp_path)


def test_loader_not_a_directory(tmp_path: Path):
    filepath = tmp_path / "file"
    with open(filepath, "w") as f:
        json.dump({}, f, indent=2)
    with pytest.raises(NotADirectoryError):
        DataLoader.load(filepath)


def test_empty_input(tmp_path: Path):
    loader = DataLoader.create(tmp_path)
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


def test_loader_deletion(
    tmp_path: Path, basic_segmentations: list[Segmentation]
):
    loader = DataLoader.create(tmp_path)
    loader.add_data(basic_segmentations)
    assert tmp_path == loader.path

    # check only detailed cache exists
    assert tmp_path.exists()
    assert loader._generate_cache_path(tmp_path).exists()
    assert loader._generate_metadata_path(tmp_path).exists()

    # verify deletion
    DataLoader.delete(tmp_path)
    assert not tmp_path.exists()
    assert not loader._generate_cache_path(tmp_path).exists()
    assert not loader._generate_metadata_path(tmp_path).exists()
