import numpy as np
from valor_lite.classification import (
    Classification,
    DataLoader,
    MetricType,
    compute_metrics,
)


def test_counts_computation():

    # groundtruth, prediction, score
    data = np.array(
        [
            # datum 0
            [0, 0, 1.0],  # tp
            [0, 1, 0.0],  # tn
            [0, 2, 0.0],  # tn
            [0, 3, 0.0],  # tn
            # datum 1
            [0, 0, 0.0],  # fn
            [0, 1, 0.0],  # tn
            [0, 2, 1.0],  # fp
            [0, 3, 0.0],  # tn
            # datum 2
            [3, 0, 0.0],  # tn
            [3, 1, 0.0],  # tn
            [3, 2, 0.0],  # tn
            [3, 3, 0.3],  # fn for score threshold > 0.3
        ],
        dtype=np.float64,
    )

    # groundtruth count, prediction count, label key
    label_metadata = np.array(
        [
            [2, 1, 0],
            [0, 2, 0],
            [1, 0, 0],
            [1, 1, 0],
        ],
        dtype=np.int32,
    )

    score_thresholds = np.array([0.25, 0.75], dtype=np.float64)

    (counts, _, _, _, _, _, _,) = compute_metrics(
        data=data,
        label_metadata=label_metadata,
        score_thresholds=score_thresholds,
        n_datums=3,
    )

    # score threshold, label, count metric
    assert counts.shape == (2, 4, 4)

    # label 0
    # score >= 0.25
    assert counts[0][0][0] == 1  # tp
    assert counts[0][0][1] == 0  # fp
    assert counts[0][0][2] == 1  # fn
    assert counts[0][0][3] == 1  # tn
    # score >= 0.75
    assert counts[1][0][0] == 1  # tp
    assert counts[1][0][1] == 0  # fp
    assert counts[1][0][2] == 1  # fn
    assert counts[1][0][3] == 1  # tn

    # label 1
    # score >= 0.25
    assert counts[0][1][0] == 0  # tp
    assert counts[0][1][1] == 0  # fp
    assert counts[0][1][2] == 0  # fn
    assert counts[0][1][3] == 3  # tn
    # score >= 0.75
    assert counts[1][1][0] == 0  # tp
    assert counts[1][1][1] == 0  # fp
    assert counts[1][1][2] == 0  # fn
    assert counts[1][1][3] == 3  # tn

    # label 2
    # score >= 0.25
    assert counts[0][2][0] == 0  # tp
    assert counts[0][2][1] == 1  # fp
    assert counts[0][2][2] == 0  # fn
    assert counts[0][2][3] == 2  # tn
    # score >= 0.75
    assert counts[1][2][0] == 0  # tp
    assert counts[1][2][1] == 1  # fp
    assert counts[1][2][2] == 0  # fn
    assert counts[1][2][3] == 2  # tn

    # label 3
    # score >= 0.25
    assert counts[0][3][0] == 1  # tp
    assert counts[0][3][1] == 0  # fp
    assert counts[0][3][2] == 0  # fn
    assert counts[0][3][3] == 2  # tn
    # score >= 0.75
    assert counts[1][3][0] == 0  # tp
    assert counts[1][3][1] == 0  # fp
    assert counts[1][3][2] == 1  # fn
    assert counts[1][3][3] == 2  # tn


def test_counts_basic(basic_classifications: list[Classification]):
    loader = DataLoader()
    loader.add_data(basic_classifications)
    evaluator = loader.finalize()

    metrics = evaluator.evaluate(score_thresholds=[0.25, 0.75])

    # test Counts
    actual_metrics = [m.to_dict() for m in metrics[MetricType.Counts]]
    expected_metrics = [
        {
            "type": "Counts",
            "value": {
                "tp": [1, 1],
                "fp": [0, 0],
                "fn": [1, 1],
                "tn": [1, 1],
            },
            "parameters": {
                "score_thresholds": [0.25, 0.75],
                "label": {"key": "class", "value": "0"},
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": [0, 0],
                "fp": [0, 0],
                "fn": [0, 0],
                "tn": [3, 3],
            },
            "parameters": {
                "score_thresholds": [0.25, 0.75],
                "label": {"key": "class", "value": "1"},
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": [0, 0],
                "fp": [1, 1],
                "fn": [0, 0],
                "tn": [2, 2],
            },
            "parameters": {
                "score_thresholds": [0.25, 0.75],
                "label": {"key": "class", "value": "2"},
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": [1, 0],
                "fp": [0, 0],
                "fn": [0, 1],
                "tn": [2, 2],
            },
            "parameters": {
                "score_thresholds": [0.25, 0.75],
                "label": {"key": "class", "value": "3"},
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_counts_with_example(classifications: list[Classification]):

    loader = DataLoader()
    loader.add_data(classifications)
    evaluator = loader.finalize()

    metrics = evaluator.evaluate(score_thresholds=[0.5])

    # test Counts
    actual_metrics = [m.to_dict() for m in metrics[MetricType.Counts]]
    expected_metrics = [
        {
            "type": "Counts",
            "value": {
                "tp": [1],
                "fp": [0],
                "fn": [2],
                "tn": [3],
            },
            "parameters": {
                "score_thresholds": [0.5],
                "label": {"key": "animal", "value": "bird"},
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": [0],
                "fp": [1],
                "fn": [2],
                "tn": [3],
            },
            "parameters": {
                "score_thresholds": [0.5],
                "label": {"key": "animal", "value": "dog"},
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": [1],
                "fp": [2],
                "fn": [0],
                "tn": [3],
            },
            "parameters": {
                "score_thresholds": [0.5],
                "label": {"key": "animal", "value": "cat"},
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": [1],
                "fp": [1],
                "fn": [1],
                "tn": [3],
            },
            "parameters": {
                "score_thresholds": [0.5],
                "label": {"key": "color", "value": "white"},
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": [1],
                "fp": [1],
                "fn": [1],
                "tn": [3],
            },
            "parameters": {
                "score_thresholds": [0.5],
                "label": {"key": "color", "value": "red"},
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": [0],
                "fp": [1],
                "fn": [1],
                "tn": [4],
            },
            "parameters": {
                "score_thresholds": [0.5],
                "label": {"key": "color", "value": "blue"},
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": [0],
                "fp": [0],
                "fn": [1],
                "tn": [5],
            },
            "parameters": {
                "score_thresholds": [0.5],
                "label": {"key": "color", "value": "black"},
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics
