import numpy as np
from valor_lite.classification import (
    Classification,
    DataLoader,
    MetricType,
    compute_metrics,
)


def test_accuracy_computation():

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

    (_, _, _, accuracy, _, _, _) = compute_metrics(
        data=data,
        label_metadata=label_metadata,
        score_thresholds=score_thresholds,
        n_datums=3,
        hardmax=False,
    )

    # score threshold, label, count metric
    assert accuracy.shape == (2, 4)

    # score >= 0.25
    assert accuracy[0][0] == 2 / 3
    assert accuracy[0][1] == 1.0
    assert accuracy[0][2] == 2 / 3
    assert accuracy[0][3] == 1.0
    # score >= 0.75
    assert accuracy[1][0] == 2 / 3
    assert accuracy[1][1] == 1.0
    assert accuracy[1][2] == 2 / 3
    assert accuracy[1][3] == 2 / 3


def test_accuracy_basic(basic_classifications: list[Classification]):
    loader = DataLoader()
    loader.add_data(basic_classifications)
    evaluator = loader.finalize()

    assert evaluator.metadata == {
        "n_datums": 3,
        "n_groundtruths": 3,
        "n_predictions": 12,
        "n_labels": 4,
        "ignored_prediction_labels": ["1", "2"],
        "missing_prediction_labels": [],
    }

    metrics = evaluator.evaluate(score_thresholds=[0.25, 0.75], as_dict=True)

    actual_metrics = [m for m in metrics[MetricType.Accuracy]]
    expected_metrics = [
        {
            "type": "Accuracy",
            "value": [2 / 3, 2 / 3],
            "parameters": {
                "score_thresholds": [0.25, 0.75],
                "hardmax": True,
                "label": "0",
            },
        },
        {
            "type": "Accuracy",
            "value": [1.0, 2 / 3],
            "parameters": {
                "score_thresholds": [0.25, 0.75],
                "hardmax": True,
                "label": "3",
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_accuracy_with_animal_example(
    classifications_animal_example: list[Classification],
):

    loader = DataLoader()
    loader.add_data(classifications_animal_example)
    evaluator = loader.finalize()

    metrics = evaluator.evaluate(score_thresholds=[0.5], as_dict=True)

    actual_metrics = [m for m in metrics[MetricType.Accuracy]]
    expected_metrics = [
        {
            "type": "Accuracy",
            "value": [2.0 / 3.0],
            "parameters": {
                "score_thresholds": [0.5],
                "hardmax": True,
                "label": "bird",
            },
        },
        {
            "type": "Accuracy",
            "value": [0.5],
            "parameters": {
                "score_thresholds": [0.5],
                "hardmax": True,
                "label": "dog",
            },
        },
        {
            "type": "Accuracy",
            "value": [2 / 3],
            "parameters": {
                "score_thresholds": [0.5],
                "hardmax": True,
                "label": "cat",
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_accuracy_color_example(
    classifications_color_example: list[Classification],
):

    loader = DataLoader()
    loader.add_data(classifications_color_example)
    evaluator = loader.finalize()

    metrics = evaluator.evaluate(score_thresholds=[0.5], as_dict=True)

    actual_metrics = [m for m in metrics[MetricType.Accuracy]]
    expected_metrics = [
        {
            "type": "Accuracy",
            "value": [2 / 3],
            "parameters": {
                "score_thresholds": [0.5],
                "hardmax": True,
                "label": "white",
            },
        },
        {
            "type": "Accuracy",
            "value": [2 / 3],
            "parameters": {
                "score_thresholds": [0.5],
                "hardmax": True,
                "label": "red",
            },
        },
        {
            "type": "Accuracy",
            "value": [2 / 3],
            "parameters": {
                "score_thresholds": [0.5],
                "hardmax": True,
                "label": "blue",
            },
        },
        {
            "type": "Accuracy",
            "value": [5 / 6],
            "parameters": {
                "score_thresholds": [0.5],
                "hardmax": True,
                "label": "black",
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_accuracy_with_image_example(
    classifications_image_example: list[Classification],
):
    loader = DataLoader()
    loader.add_data(classifications_image_example)
    evaluator = loader.finalize()

    assert evaluator.metadata == {
        "n_datums": 2,
        "n_groundtruths": 2,
        "n_predictions": 4,
        "n_labels": 4,
        "ignored_prediction_labels": ["v1", "v8", "v5"],
        "missing_prediction_labels": [],
    }

    metrics = evaluator.evaluate(as_dict=True)

    actual_metrics = [m for m in metrics[MetricType.Accuracy]]
    expected_metrics = [
        {
            "type": "Accuracy",
            "value": [0.5],
            "parameters": {
                "score_thresholds": [0.0],
                "hardmax": True,
                "label": "v4",
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_accuracy_with_tabular_example(
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

    metrics = evaluator.evaluate(as_dict=True)

    actual_metrics = [m for m in metrics[MetricType.Accuracy]]
    expected_metrics = [
        {
            "type": "Accuracy",
            "value": [0.7],
            "parameters": {
                "score_thresholds": [0.0],
                "hardmax": True,
                "label": "0",
            },
        },
        {
            "type": "Accuracy",
            "value": [0.5],
            "parameters": {
                "score_thresholds": [0.0],
                "hardmax": True,
                "label": "1",
            },
        },
        {
            "type": "Accuracy",
            "value": [0.8],
            "parameters": {
                "score_thresholds": [0.0],
                "hardmax": True,
                "label": "2",
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics
