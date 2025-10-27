from pathlib import Path

import numpy as np

from valor_lite.classification import Classification, DataLoader, Metric


def test_metadata_using_classification_example(
    tmp_path: Path,
    classifications_animal_example: list[Classification],
):
    loader = DataLoader.create(tmp_path)
    loader.add_data(classifications_animal_example)
    evaluator = loader.finalize()

    assert evaluator.metadata.number_of_datums == 6
    assert evaluator.metadata.number_of_labels == 3
    assert evaluator.metadata.number_of_ground_truths == 6
    assert evaluator.metadata.number_of_predictions == 3 * 6


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
    loader = DataLoader.create(tmp_path)
    loader.add_data(basic_classifications)
    evaluator = loader.finalize()

    metrics = evaluator.evaluate(
        score_thresholds=[0.25, 0.75],
    )
    values = _flatten_metrics(metrics)
    for value in values:
        if isinstance(value, (np.generic, np.ndarray, Metric)):
            raise TypeError(value)
