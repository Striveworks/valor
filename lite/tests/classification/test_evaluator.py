import numpy as np
from valor_lite.classification import Classification, DataLoader


def test_metadata_using_classification_example(
    classifications_animal_example: list[Classification],
):
    manager = DataLoader()
    manager.add_data(classifications_animal_example)
    evaluator = manager.finalize()

    assert evaluator.ignored_prediction_labels == []
    assert evaluator.missing_prediction_labels == []
    assert evaluator.n_datums == 6
    assert evaluator.n_labels == 3
    assert evaluator.n_groundtruths == 6
    assert evaluator.n_predictions == 3 * 6

    assert evaluator.metadata == {
        "n_datums": 6,
        "n_groundtruths": 6,
        "n_predictions": 3 * 6,
        "n_labels": 3,
        "ignored_prediction_labels": [],
        "missing_prediction_labels": [],
    }


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
    else:
        return [m]


def test_output_types_dont_contain_numpy(
    basic_classifications: list[Classification],
):
    manager = DataLoader()
    manager.add_data(basic_classifications)
    evaluator = manager.finalize()

    metrics = evaluator.evaluate(
        score_thresholds=[0.25, 0.75],
        as_dict=True,
    )

    values = _flatten_metrics(metrics)
    for value in values:
        if isinstance(value, (np.generic, np.ndarray)):
            raise TypeError(value)
