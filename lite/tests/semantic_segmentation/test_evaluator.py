import numpy as np
from valor_lite.semantic_segmentation import DataLoader, Segmentation


def test_metadata_using_large_random_segmentations(
    large_random_segmentations: list[Segmentation],
):
    manager = DataLoader()
    manager.add_data(large_random_segmentations)
    evaluator = manager.finalize()

    assert evaluator.ignored_prediction_labels == []
    assert evaluator.missing_prediction_labels == []
    assert evaluator.n_datums == 3
    assert evaluator.n_labels == 9
    assert evaluator.n_groundtruths == 9
    assert evaluator.n_predictions == 9
    assert evaluator.n_groundtruth_pixels == 36000000
    assert evaluator.n_prediction_pixels == 36000000

    assert evaluator.metadata == {
        "ignored_prediction_labels": [],
        "missing_prediction_labels": [],
        "number_of_datums": 3,
        "number_of_labels": 9,
        "number_of_groundtruths": 9,
        "number_of_predictions": 9,
        "number_of_groundtruth_pixels": 36000000,
        "number_of_prediction_pixels": 36000000,
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
    segmentations_from_boxes: list[Segmentation],
):
    manager = DataLoader()
    manager.add_data(segmentations_from_boxes)
    evaluator = manager.finalize()

    metrics = evaluator.evaluate(
        as_dict=True,
    )

    values = _flatten_metrics(metrics)
    for value in values:
        if isinstance(value, (np.generic, np.ndarray)):
            raise TypeError(value)
