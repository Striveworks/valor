import numpy as np
from valor_lite.classification import (
    Classification,
    DataLoader,
    MetricType,
    compute_metrics,
)


def test_precision_computation():

    # groundtruth, prediction, score, hardmax
    data = np.array(
        [
            # datum 0
            [0, 0, 0, 1.0, 0],  # tp
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

    (_, precision, _, _, _, _, _) = compute_metrics(
        data=data,
        label_metadata=label_metadata,
        score_thresholds=score_thresholds,
        n_datums=3,
        hardmax=False,
    )

    # score threshold, label, count metric
    assert precision.shape == (2, 4)

    # score >= 0.25
    assert precision[0][0] == 1.0
    assert precision[0][1] == 0.0
    assert precision[0][2] == 0.0
    assert precision[0][3] == 1.0
    # score >= 0.75
    assert precision[1][0] == 1.0
    assert precision[1][1] == 0.0
    assert precision[1][2] == 0.0
    assert precision[1][3] == 0.0


def test_precision_basic(basic_classifications: list[Classification]):
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

    metrics = evaluator.evaluate(
        score_thresholds=[0.25, 0.75],
        as_dict=True,
    )

    actual_metrics = [m for m in metrics[MetricType.Precision]]
    expected_metrics = [
        {
            "type": "Precision",
            "value": [1.0, 1.0],
            "parameters": {
                "score_thresholds": [0.25, 0.75],
                "hardmax": True,
                "label": "0",
            },
        },
        {
            "type": "Precision",
            "value": [1.0, 0.0],
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


def test_precision_with_animal_example(
    classifications_animal_example: list[Classification],
):
    loader = DataLoader()
    loader.add_data(classifications_animal_example)
    evaluator = loader.finalize()

    metrics = evaluator.evaluate(
        score_thresholds=[0.0, 0.5],
        as_dict=True,
    )

    actual_metrics = [m for m in metrics[MetricType.Precision]]
    expected_metrics = [
        {
            "type": "Precision",
            "value": [1.0, 1.0],
            "parameters": {
                "score_thresholds": [0.0, 0.5],
                "hardmax": True,
                "label": "bird",
            },
        },
        {
            "type": "Precision",
            "value": [0.0, 0.0],
            "parameters": {
                "score_thresholds": [0.0, 0.5],
                "hardmax": True,
                "label": "dog",
            },
        },
        {
            "type": "Precision",
            "value": [0.25, 1 / 3],
            "parameters": {
                "score_thresholds": [0.0, 0.5],
                "hardmax": True,
                "label": "cat",
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_precision_with_color_example(
    classifications_color_example: list[Classification],
):
    loader = DataLoader()
    loader.add_data(classifications_color_example)
    evaluator = loader.finalize()

    metrics = evaluator.evaluate(
        score_thresholds=[0.0, 0.5],
        as_dict=True,
    )

    actual_metrics = [m for m in metrics[MetricType.Precision]]
    expected_metrics = [
        {
            "type": "Precision",
            "value": [0.5, 0.5],
            "parameters": {
                "score_thresholds": [0.0, 0.5],
                "hardmax": True,
                "label": "white",
            },
        },
        {
            "type": "Precision",
            "value": [2 / 3, 0.5],
            "parameters": {
                "score_thresholds": [0.0, 0.5],
                "hardmax": True,
                "label": "red",
            },
        },
        {
            "type": "Precision",
            "value": [0.0, 0.0],
            "parameters": {
                "score_thresholds": [0.0, 0.5],
                "hardmax": True,
                "label": "blue",
            },
        },
        {
            "type": "Precision",
            "value": [0.0, 0.0],
            "parameters": {
                "score_thresholds": [0.0, 0.5],
                "hardmax": True,
                "label": "black",
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_precision_with_image_example(
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

    actual_metrics = [m for m in metrics[MetricType.Precision]]
    expected_metrics = [
        {
            "type": "Precision",
            "value": [1.0],
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


def test_precision_with_tabular_example(
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

    actual_metrics = [m for m in metrics[MetricType.Precision]]
    expected_metrics = [
        {
            "type": "Precision",
            "value": [0.5],
            "parameters": {
                "score_thresholds": [0.0],
                "hardmax": True,
                "label": "0",
            },
        },
        {
            "type": "Precision",
            "value": [0.6666666666666666],
            "parameters": {
                "score_thresholds": [0.0],
                "hardmax": True,
                "label": "1",
            },
        },
        {
            "type": "Precision",
            "value": [0.0],
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
