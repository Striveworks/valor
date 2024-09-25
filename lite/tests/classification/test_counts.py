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
            [0, 0, 0, 1.0, 1],  # tp
            [0, 0, 1, 0.0, 0],  # tn
            [0, 0, 2, 0.0, 0],  # tn
            [0, 0, 3, 0.0, 0],  # tn
            # datum 1
            [1, 0, 0, 0.0, 0],  # fn
            [1, 0, 1, 0.0, 0],  # tn
            [1, 0, 2, 1.0, 1],  # fp
            [1, 0, 3, 0.0, 0],  # tn
            # datum 2
            [2, 3, 0, 0.0, 0],  # tn
            [2, 3, 1, 0.0, 0],  # tn
            [2, 3, 2, 0.0, 0],  # tn
            [2, 3, 3, 0.3, 1],  # fn for score threshold > 0.3
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
        hardmax=False,
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


def test_counts_basic(classifications_basic: list[Classification]):
    loader = DataLoader()
    loader.add_data(classifications_basic)
    evaluator = loader.finalize()

    assert evaluator.metadata == {
        "n_datums": 3,
        "n_groundtruths": 3,
        "n_predictions": 12,
        "n_labels": 4,
        "ignored_prediction_labels": [
            ("class", "1"),
            ("class", "2"),
        ],
        "missing_prediction_labels": [],
    }

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


def test_counts_unit(
    classifications_from_api_unit_tests: list[Classification],
):

    loader = DataLoader()
    loader.add_data(classifications_from_api_unit_tests)
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
                "label": {"key": "class", "value": "0"},
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": [1],
                "fp": [3],
                "fn": [0],
                "tn": [2],
            },
            "parameters": {
                "score_thresholds": [0.5],
                "label": {"key": "class", "value": "1"},
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
                "label": {"key": "class", "value": "2"},
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_counts_with_example(
    classifications_two_categeories: list[Classification],
):

    loader = DataLoader()
    loader.add_data(classifications_two_categeories)
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


def test_counts_with_image_example(
    classifications_image_example: list[Classification],
):
    loader = DataLoader()
    loader.add_data(classifications_image_example)
    evaluator = loader.finalize()

    assert evaluator.metadata == {
        "n_datums": 3,
        "n_groundtruths": 4,
        "n_predictions": 6,
        "n_labels": 8,
        "ignored_prediction_labels": [
            ("k4", "v1"),
            ("k4", "v8"),
            ("k5", "v1"),
            ("k4", "v5"),
            ("k3", "v1"),
        ],
        "missing_prediction_labels": [
            ("k5", "v5"),
            ("k3", "v3"),
        ],
    }

    metrics = evaluator.evaluate()

    # test Counts
    actual_metrics = [m.to_dict() for m in metrics[MetricType.Counts]]
    expected_metrics = [
        # k3
        {
            "type": "Counts",
            "value": {
                "tp": [0],
                "fp": [1],
                "fn": [0],
                "tn": [0],
            },
            "parameters": {
                "score_thresholds": [0.0],
                "label": {"key": "k3", "value": "v1"},
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": [0],
                "fp": [0],
                "fn": [1],
                "tn": [0],
            },
            "parameters": {
                "score_thresholds": [0.0],
                "label": {"key": "k3", "value": "v3"},
            },
        },
        # k4
        {
            "type": "Counts",
            "value": {
                "tp": [0],
                "fp": [0],
                "fn": [0],
                "tn": [1],
            },
            "parameters": {
                "score_thresholds": [0.0],
                "label": {"key": "k4", "value": "v1"},
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": [1],
                "fp": [0],
                "fn": [1],
                "tn": [0],
            },
            "parameters": {
                "score_thresholds": [0.0],
                "label": {"key": "k4", "value": "v4"},
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": [0],
                "fp": [0],
                "fn": [0],
                "tn": [1],
            },
            "parameters": {
                "score_thresholds": [0.0],
                "label": {"key": "k4", "value": "v5"},
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": [0],
                "fp": [1],
                "fn": [0],
                "tn": [0],
            },
            "parameters": {
                "score_thresholds": [0.0],
                "label": {"key": "k4", "value": "v8"},
            },
        },
        # k5
        {
            "type": "Counts",
            "value": {
                "tp": [0],
                "fp": [1],
                "fn": [0],
                "tn": [0],
            },
            "parameters": {
                "score_thresholds": [0.0],
                "label": {"key": "k5", "value": "v1"},
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": [0],
                "fp": [0],
                "fn": [1],
                "tn": [0],
            },
            "parameters": {
                "score_thresholds": [0.0],
                "label": {"key": "k5", "value": "v5"},
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_counts_with_tabular_example(
    classifications_tabular_example: list[Classification],
):
    loader = DataLoader()
    loader.add_data(classifications_tabular_example)
    evaluator = loader.finalize()

    assert evaluator.metadata == {
        "n_datums": 10,
        "n_groundtruths": 10,
        "n_predictions": 30,
        "n_labels": 3,
        "ignored_prediction_labels": [],
        "missing_prediction_labels": [],
    }

    metrics = evaluator.evaluate()

    # test Counts
    actual_metrics = [m.to_dict() for m in metrics[MetricType.Counts]]
    expected_metrics = [
        {
            "type": "Counts",
            "value": {
                "tp": [3],
                "fp": [3],
                "fn": [0],
                "tn": [4],
            },
            "parameters": {
                "score_thresholds": [0.0],
                "label": {"key": "class", "value": "0"},
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": [2],
                "fp": [1],
                "fn": [4],
                "tn": [3],
            },
            "parameters": {
                "score_thresholds": [0.0],
                "label": {"key": "class", "value": "1"},
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": [0],
                "fp": [1],
                "fn": [1],
                "tn": [8],
            },
            "parameters": {
                "score_thresholds": [0.0],
                "label": {"key": "class", "value": "2"},
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_counts_mutliclass(
    classifications_multiclass: list[Classification],
):
    loader = DataLoader()
    loader.add_data(classifications_multiclass)
    evaluator = loader.finalize()

    assert evaluator.metadata == {
        "ignored_prediction_labels": [],
        "missing_prediction_labels": [],
        "n_datums": 5,
        "n_groundtruths": 5,
        "n_labels": 3,
        "n_predictions": 15,
    }

    metrics = evaluator.evaluate(score_thresholds=[0.05, 0.1, 0.3, 0.85])

    # test Counts
    actual_metrics = [m.to_dict() for m in metrics[MetricType.Counts]]
    expected_metrics = [
        {
            "value": {
                "tp": [1, 1, 1, 0],
                "fp": [0, 0, 0, 0],
                "fn": [1, 1, 1, 2],
                "tn": [3, 3, 3, 3],
            },
            "parameters": {
                "score_thresholds": [0.05, 0.1, 0.3, 0.85],
                "label": {"key": "class_label", "value": "cat"},
            },
            "type": "Counts",
        },
        {
            "value": {
                "tp": [1, 1, 1, 0],
                "fp": [0, 0, 0, 0],
                "fn": [0, 0, 0, 1],
                "tn": [4, 4, 4, 4],
            },
            "parameters": {
                "score_thresholds": [0.05, 0.1, 0.3, 0.85],
                "label": {"key": "class_label", "value": "dog"},
            },
            "type": "Counts",
        },
        {
            "value": {
                "tp": [2, 2, 2, 0],
                "fp": [1, 1, 1, 0],
                "fn": [0, 0, 0, 2],
                "tn": [2, 2, 2, 3],
            },
            "parameters": {
                "score_thresholds": [0.05, 0.1, 0.3, 0.85],
                "label": {"key": "class_label", "value": "bee"},
            },
            "type": "Counts",
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_counts_true_negatives_check(
    classifications_multiclass_true_negatives_check: list[Classification],
):
    loader = DataLoader()
    loader.add_data(classifications_multiclass_true_negatives_check)
    evaluator = loader.finalize()

    assert evaluator.metadata == {
        "ignored_prediction_labels": [
            ("k1", "bee"),
            ("k1", "cat"),
            ("k2", "milk"),
            ("k2", "flour"),
        ],
        "missing_prediction_labels": [],
        "n_datums": 2,
        "n_groundtruths": 2,
        "n_labels": 6,
        "n_predictions": 6,
    }

    metrics = evaluator.evaluate(
        score_thresholds=[0.05, 0.15, 0.95],
    )

    # test Counts
    actual_metrics = [m.to_dict() for m in metrics[MetricType.Counts]]
    expected_metrics = [
        {
            "value": {
                "tp": [0, 0, 0],
                "fp": [0, 0, 0],
                "fn": [1, 1, 1],
                "tn": [0, 0, 0],
            },
            "parameters": {
                "score_thresholds": [0.05, 0.15, 0.95],
                "label": {"key": "k1", "value": "ant"},
            },
            "type": "Counts",
        },
        {
            "value": {
                "tp": [0, 0, 0],
                "fp": [1, 1, 0],
                "fn": [0, 0, 0],
                "tn": [0, 0, 1],
            },
            "parameters": {
                "score_thresholds": [0.05, 0.15, 0.95],
                "label": {"key": "k1", "value": "bee"},
            },
            "type": "Counts",
        },
        {
            "value": {
                "tp": [0, 0, 0],
                "fp": [0, 0, 0],
                "fn": [0, 0, 0],
                "tn": [1, 1, 1],
            },
            "parameters": {
                "score_thresholds": [0.05, 0.15, 0.95],
                "label": {"key": "k1", "value": "cat"},
            },
            "type": "Counts",
        },
        {
            "value": {
                "tp": [0, 0, 0],
                "fp": [0, 0, 0],
                "fn": [1, 1, 1],
                "tn": [0, 0, 0],
            },
            "parameters": {
                "score_thresholds": [0.05, 0.15, 0.95],
                "label": {"key": "k2", "value": "egg"},
            },
            "type": "Counts",
        },
        {
            "value": {
                "tp": [0, 0, 0],
                "fp": [1, 1, 0],
                "fn": [0, 0, 0],
                "tn": [0, 0, 1],
            },
            "parameters": {
                "score_thresholds": [0.05, 0.15, 0.95],
                "label": {"key": "k2", "value": "milk"},
            },
            "type": "Counts",
        },
        {
            "value": {
                "tp": [0, 0, 0],
                "fp": [0, 0, 0],
                "fn": [0, 0, 0],
                "tn": [1, 1, 1],
            },
            "parameters": {
                "score_thresholds": [0.05, 0.15, 0.95],
                "label": {"key": "k2", "value": "flour"},
            },
            "type": "Counts",
        },
    ]
    for m in actual_metrics:
        import json

        print(json.dumps(m, indent=4))
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_counts_zero_count_check(
    classifications_multiclass_zero_count: list[Classification],
):

    loader = DataLoader()
    loader.add_data(classifications_multiclass_zero_count)
    evaluator = loader.finalize()

    assert evaluator.metadata == {
        "ignored_prediction_labels": [
            ("k", "bee"),
            ("k", "cat"),
        ],
        "missing_prediction_labels": [],
        "n_datums": 1,
        "n_groundtruths": 1,
        "n_labels": 3,
        "n_predictions": 3,
    }

    metrics = evaluator.evaluate(score_thresholds=[0.05, 0.2, 0.95])

    # test Counts
    actual_metrics = [m.to_dict() for m in metrics[MetricType.Counts]]
    expected_metrics = [
        {
            "value": {
                "tp": [0, 0, 0],
                "fp": [0, 0, 0],
                "fn": [1, 1, 1],
                "tn": [0, 0, 0],
            },
            "parameters": {
                "score_thresholds": [0.05, 0.2, 0.95],
                "label": {"key": "k", "value": "ant"},
            },
            "type": "Counts",
        },
        {
            "value": {
                "tp": [0, 0, 0],
                "fp": [1, 1, 0],
                "fn": [0, 0, 0],
                "tn": [0, 0, 1],
            },
            "parameters": {
                "score_thresholds": [0.05, 0.2, 0.95],
                "label": {"key": "k", "value": "bee"},
            },
            "type": "Counts",
        },
        {
            "value": {
                "tp": [0, 0, 0],
                "fp": [0, 0, 0],
                "fn": [0, 0, 0],
                "tn": [1, 1, 1],
            },
            "parameters": {
                "score_thresholds": [0.05, 0.2, 0.95],
                "label": {"key": "k", "value": "cat"},
            },
            "type": "Counts",
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics
