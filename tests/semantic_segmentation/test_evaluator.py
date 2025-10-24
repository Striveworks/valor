import json
from pathlib import Path

import numpy as np
import pytest

from valor_lite.semantic_segmentation import (
    Bitmask,
    DataLoader,
    Evaluator,
    Metric,
    Segmentation,
)


def test_evaluator_file_not_found(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        Evaluator.load(tmp_path)


def test_evaluator_not_a_directory(tmp_path: Path):
    filepath = tmp_path / "file"
    with open(filepath, "w") as f:
        json.dump({}, f, indent=2)
    with pytest.raises(NotADirectoryError):
        Evaluator.load(filepath)


def test_metadata_using_large_random_segmentations(
    tmp_path: Path,
    large_random_segmentations: list[Segmentation],
):
    loader = DataLoader.create(tmp_path)
    loader.add_data(large_random_segmentations)
    evaluator = loader.finalize()

    assert evaluator.metadata.number_of_datums == 3
    assert evaluator.metadata.number_of_labels == 9
    assert (
        evaluator.metadata.number_of_pixels == 3 * 2000 * 2000
    )  # 3x (2000,2000) bitmasks

    # check static values
    assert evaluator._info.number_of_datums == 3
    assert evaluator._info.number_of_labels == 9
    assert evaluator._info.number_of_pixels == 12000000


def _flatten_metrics(m) -> list:
    if isinstance(m, dict):
        keys = list(m.keys())
        values = [
            inner_value
            for value in m.values()
            for inner_value in _flatten_metrics(value)
        ]
        return keys + values
    elif isinstance(m, list):
        return [
            inner_value
            for value in m
            for inner_value in _flatten_metrics(value)
        ]
    elif isinstance(m, Metric):
        return _flatten_metrics(m.to_dict())
    else:
        return [m]


def test_output_types_dont_contain_numpy(
    tmp_path: Path,
    segmentations_from_boxes: list[Segmentation],
):
    loader = DataLoader.create(tmp_path)
    loader.add_data(segmentations_from_boxes)
    evaluator = loader.finalize()

    metrics = evaluator.evaluate()

    values = _flatten_metrics(metrics)
    for value in values:
        if isinstance(value, (np.generic, np.ndarray)):
            raise TypeError(value)


def test_label_mismatch(tmp_path: Path):

    loader = DataLoader.create(tmp_path)
    loader.add_data(
        segmentations=[
            Segmentation(
                uid="uid0",
                groundtruths=[
                    Bitmask(
                        mask=np.array(
                            [
                                [True, True],
                                [False, False],
                            ]
                        ),
                        label="v1",
                    )
                ],
                predictions=[
                    Bitmask(
                        mask=np.array(
                            [
                                [True, False],
                                [False, False],
                            ]
                        ),
                        label="v2",
                    ),
                    Bitmask(
                        mask=np.array(
                            [
                                [False, False],
                                [False, True],
                            ]
                        ),
                        label="v3",
                    ),
                ],
                shape=(2, 2),
            )
        ]
    )
    evaluator = loader.finalize()
    assert np.all(
        evaluator._confusion_matrix
        == np.array(
            [
                [
                    [1, 0, 0, 1],
                    [1, 0, 1, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ]
            ]
        )
    )


def test_empty_groundtruths(tmp_path: Path):

    loader = DataLoader.create(tmp_path)
    loader.add_data(
        segmentations=[
            Segmentation(
                uid="uid0",
                groundtruths=[],
                predictions=[
                    Bitmask(
                        mask=np.array(
                            [
                                [True, False],
                                [False, False],
                            ]
                        ),
                        label="v2",
                    ),
                    Bitmask(
                        mask=np.array(
                            [
                                [False, False],
                                [False, True],
                            ]
                        ),
                        label="v3",
                    ),
                ],
                shape=(2, 2),
            )
        ]
    )
    evaluator = loader.finalize()
    assert np.all(
        evaluator._confusion_matrix
        == np.array(
            [
                [
                    [2, 1, 1],
                    [0, 0, 0],
                    [0, 0, 0],
                ]
            ]
        )
    )


def test_empty_predictions(tmp_path: Path):

    loader = DataLoader.create(tmp_path)
    loader.add_data(
        segmentations=[
            Segmentation(
                uid="uid0",
                groundtruths=[
                    Bitmask(
                        mask=np.array(
                            [
                                [True, False],
                                [False, False],
                            ]
                        ),
                        label="v2",
                    ),
                    Bitmask(
                        mask=np.array(
                            [
                                [False, False],
                                [False, True],
                            ]
                        ),
                        label="v3",
                    ),
                ],
                predictions=[],
                shape=(2, 2),
            )
        ]
    )
    evaluator = loader.finalize()
    assert np.all(
        evaluator._confusion_matrix
        == np.array(
            [
                [2, 0, 0],
                [1, 0, 0],
                [1, 0, 0],
            ]
        )
    )


def test_loader_deletion(
    tmp_path: Path, basic_segmentations: list[Segmentation]
):
    loader = DataLoader.create(tmp_path)
    loader.add_data(basic_segmentations)
    evaluator = loader.finalize()
    assert tmp_path == evaluator.path

    # check only detailed cache exists
    assert tmp_path.exists()
    assert evaluator._generate_cache_path(tmp_path).exists()
    assert evaluator._generate_metadata_path(tmp_path).exists()

    # verify deletion
    evaluator.delete()
    assert not tmp_path.exists()
    assert not evaluator._generate_cache_path(tmp_path).exists()
    assert not evaluator._generate_metadata_path(tmp_path).exists()
