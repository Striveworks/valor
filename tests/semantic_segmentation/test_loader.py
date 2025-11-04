from pathlib import Path

import numpy as np
import pytest

from valor_lite.exceptions import EmptyCacheError
from valor_lite.semantic_segmentation import Loader, Segmentation


def test_no_data(loader: Loader):
    with pytest.raises(EmptyCacheError):
        loader.finalize()


def test_empty_input(loader: Loader):
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
    tmp_path: Path,
    basic_segmentations: list[Segmentation],
):
    loader = Loader.persistent(tmp_path)
    loader.add_data(basic_segmentations)
    _ = loader.finalize()
    assert tmp_path == loader._path
    assert loader._datum_count == 1

    # fail if attempting to create new loader with same path
    with pytest.raises(FileExistsError):
        Loader.persistent(tmp_path)

    # delete if exists toggle
    loader = Loader.persistent(tmp_path, delete_if_exists=True)
    loader.add_data(basic_segmentations)
    _ = loader.finalize()
    assert loader._datum_count == 1

    # check only detailed cache exists
    assert tmp_path.exists()
    assert loader._generate_cache_path(tmp_path).exists()
    assert loader._generate_metadata_path(tmp_path).exists()

    # verify deletion
    loader.delete()
    assert not tmp_path.exists()
    assert not loader._generate_cache_path(tmp_path).exists()
    assert not loader._generate_metadata_path(tmp_path).exists()
