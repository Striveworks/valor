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
