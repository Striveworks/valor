import numpy as np

from valor_lite.semantic_segmentation import (
    Bitmask,
    DataLoader,
    Metric,
    Segmentation,
)


def test_metadata_using_large_random_segmentations(
    large_random_segmentations: list[Segmentation],
):
    manager = DataLoader()
    manager.add_data(large_random_segmentations)
    evaluator = manager.finalize()

    assert evaluator.ignored_prediction_labels == []
    assert evaluator.missing_prediction_labels == []
    assert evaluator.metadata.number_of_datums == 3
    assert evaluator.metadata.number_of_labels == 9
    assert (
        evaluator.metadata.number_of_pixels == 3 * 2000 * 2000
    )  # 3x (2000,2000) bitmasks

    metadata = evaluator.metadata.to_dict()
    # pop randomly changing values
    metadata.pop("number_of_ground_truths")
    metadata.pop("number_of_predictions")
    assert metadata == {
        "number_of_datums": 3,
        "number_of_labels": 9,
        "number_of_pixels": 12000000,
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
    elif isinstance(m, Metric):
        return _flatten_metrics(m.to_dict())
    else:
        return [m]


def test_output_types_dont_contain_numpy(
    segmentations_from_boxes: list[Segmentation],
):
    manager = DataLoader()
    manager.add_data(segmentations_from_boxes)
    evaluator = manager.finalize()

    metrics = evaluator.evaluate()

    values = _flatten_metrics(metrics)
    for value in values:
        if isinstance(value, (np.generic, np.ndarray)):
            raise TypeError(value)


def test_label_mismatch():

    loader = DataLoader()
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
        evaluator._confusion_matrices
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
    assert np.all(
        evaluator._label_metadata
        == np.array(
            [
                [2, 0],
                [0, 1],
                [0, 1],
            ]
        )
    )


def test_empty_groundtruths():

    loader = DataLoader()
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
        evaluator._confusion_matrices
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
    assert np.all(
        evaluator._label_metadata
        == np.array(
            [
                [0, 1],
                [0, 1],
            ]
        )
    )


def test_empty_predictions():

    loader = DataLoader()
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
        evaluator._confusion_matrices
        == np.array(
            [
                [
                    [2, 0, 0],
                    [1, 0, 0],
                    [1, 0, 0],
                ]
            ]
        )
    )
    assert np.all(
        evaluator._label_metadata
        == np.array(
            [
                [1, 0],
                [1, 0],
            ]
        )
    )
