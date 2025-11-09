import json
from pathlib import Path

import numpy as np
import pytest

from valor_lite.classification import Classification, Evaluator, Loader, Metric
from valor_lite.classification.shared import (
    generate_cache_path,
    generate_intermediate_cache_path,
    generate_metadata_path,
    generate_roc_curve_cache_path,
)


def test_evaluator_file_not_found(tmp_path: Path):
    path = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        Evaluator.load(path)


def test_evaluator_not_a_directory(tmp_path: Path):
    filepath = tmp_path / "file"
    with open(filepath, "w") as f:
        json.dump({}, f, indent=2)
    with pytest.raises(NotADirectoryError):
        Evaluator.load(filepath)


def test_evaluator_valid_thresholds(tmp_path: Path):
    eval = Evaluator(
        reader=None,  # type: ignore - testing
        info=None,  # type: ignore - testing
        index_to_label={},
        label_counts=np.ones(1, dtype=np.uint64),
    )
    for fn in [
        eval.compute_rocauc,
        eval.compute_precision_recall,
        eval.compute_examples,
        eval.compute_confusion_matrix,
        eval.compute_confusion_matrix_with_examples,
    ]:
        with pytest.raises(ValueError) as e:
            fn(score_thresholds=[])
        assert "score" in str(e)


def test_info_using_classification_example(
    tmp_path: Path,
    classifications_animal_example: list[Classification],
):
    loader = Loader.persistent(tmp_path)
    loader.add_data(classifications_animal_example)
    evaluator = loader.finalize()

    assert evaluator.info.number_of_datums == 6
    assert evaluator.info.number_of_labels == 3
    assert evaluator.info.number_of_rows == 3 * 6


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
    basic_classifications: list[Classification],
):
    loader = Loader.persistent(tmp_path)
    loader.add_data(basic_classifications)
    evaluator = loader.finalize()

    metrics = evaluator.compute_precision_recall(
        score_thresholds=[0.25, 0.75],
    )
    values = _flatten_metrics(metrics)
    for value in values:
        if isinstance(value, (np.generic, np.ndarray, Metric)):
            raise TypeError(value)


def test_evaluator_exists_on_disk(
    tmp_path: Path, basic_classifications: list[Classification]
):
    loader = Loader.persistent(tmp_path)
    loader.add_data(basic_classifications)
    _ = loader.finalize()

    # check both caches exist
    assert tmp_path.exists()
    assert generate_cache_path(tmp_path).exists()
    assert generate_intermediate_cache_path(tmp_path).exists()
    assert generate_roc_curve_cache_path(tmp_path).exists()
    assert generate_metadata_path(tmp_path).exists()
