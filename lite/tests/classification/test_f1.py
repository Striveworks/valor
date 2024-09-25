import numpy as np
from valor_lite.classification import (
    Classification,
    DataLoader,
    MetricType,
    compute_metrics,
)


def test_f1_score_computation():

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

    (_, _, _, _, f1_score, _, _,) = compute_metrics(
        data=data,
        label_metadata=label_metadata,
        score_thresholds=score_thresholds,
        n_datums=3,
        hardmax=False,
    )

    # score threshold, label, count metric
    assert f1_score.shape == (2, 4)

    # score >= 0.25
    assert f1_score[0][0] == 2 / 3
    assert f1_score[0][1] == 0.0
    assert f1_score[0][2] == 0.0
    assert f1_score[0][3] == 1.0
    # score >= 0.75
    assert f1_score[1][0] == 2 / 3
    assert f1_score[1][1] == 0.0
    assert f1_score[1][2] == 0.0
    assert f1_score[1][3] == 0.0


def test_f1_score_basic(classifications_basic: list[Classification]):
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

    # test F1
    actual_metrics = [m.to_dict() for m in metrics[MetricType.F1]]
    expected_metrics = [
        {
            "type": "F1",
            "value": [2 / 3, 2 / 3],
            "parameters": {
                "score_thresholds": [0.25, 0.75],
                "label": {"key": "class", "value": "0"},
            },
        },
        {
            "type": "F1",
            "value": [1.0, 0.0],
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


def test_f1_score_with_example(
    classifications_two_categeories: list[Classification],
):

    loader = DataLoader()
    loader.add_data(classifications_two_categeories)
    evaluator = loader.finalize()

    metrics = evaluator.evaluate(score_thresholds=[0.5])

    # test F1
    actual_metrics = [m.to_dict() for m in metrics[MetricType.F1]]
    expected_metrics = [
        {
            "type": "F1",
            "value": [0.5],
            "parameters": {
                "score_thresholds": [0.5],
                "label": {"key": "animal", "value": "bird"},
            },
        },
        {
            "type": "F1",
            "value": [0.0],
            "parameters": {
                "score_thresholds": [0.5],
                "label": {"key": "animal", "value": "dog"},
            },
        },
        {
            "type": "F1",
            "value": [0.5],
            "parameters": {
                "score_thresholds": [0.5],
                "label": {"key": "animal", "value": "cat"},
            },
        },
        {
            "type": "F1",
            "value": [0.5],
            "parameters": {
                "score_thresholds": [0.5],
                "label": {"key": "color", "value": "white"},
            },
        },
        {
            "type": "F1",
            "value": [0.5],
            "parameters": {
                "score_thresholds": [0.5],
                "label": {"key": "color", "value": "red"},
            },
        },
        {
            "type": "F1",
            "value": [0.0],
            "parameters": {
                "score_thresholds": [0.5],
                "label": {"key": "color", "value": "blue"},
            },
        },
        {
            "type": "F1",
            "value": [0.0],
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


def test_f1_score_with_image_example(
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

    # test F1
    actual_metrics = [m.to_dict() for m in metrics[MetricType.F1]]
    expected_metrics = [
        {
            "type": "F1",
            "value": [0.6666666666666666],
            "parameters": {
                "score_thresholds": [0.0],
                "label": {"key": "k4", "value": "v4"},
            },
        },
        {
            "type": "F1",
            "value": [0.0],
            "parameters": {
                "score_thresholds": [0.0],
                "label": {"key": "k5", "value": "v5"},
            },
        },
        {
            "type": "F1",
            "value": [0.0],
            "parameters": {
                "score_thresholds": [0.0],
                "label": {"key": "k3", "value": "v3"},
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_f1_score_with_tabular_example(
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

    # test F1
    actual_metrics = [m.to_dict() for m in metrics[MetricType.F1]]
    expected_metrics = [
        {
            "type": "F1",
            "value": [0.6666666666666666],
            "parameters": {
                "score_thresholds": [0.0],
                "label": {"key": "class", "value": "0"},
            },
        },
        {
            "type": "F1",
            "value": [0.4444444444444444],
            "parameters": {
                "score_thresholds": [0.0],
                "label": {"key": "class", "value": "1"},
            },
        },
        {
            "type": "F1",
            "value": [0.0],
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
