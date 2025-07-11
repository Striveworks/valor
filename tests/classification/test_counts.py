import numpy as np

from valor_lite.classification import (
    Classification,
    DataLoader,
    MetricType,
    compute_precision_recall_rocauc,
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

    (counts, _, _, _, _, _, _) = compute_precision_recall_rocauc(
        detailed_pairs=data,
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


def test_counts_basic(basic_classifications: list[Classification]):
    loader = DataLoader()
    loader.add_data(basic_classifications)
    evaluator = loader.finalize()

    assert evaluator.ignored_prediction_labels == ["1", "2"]
    assert evaluator.missing_prediction_labels == []
    assert evaluator.metadata.to_dict() == {
        "number_of_datums": 3,
        "number_of_ground_truths": 3,
        "number_of_predictions": 12,
        "number_of_labels": 4,
    }

    metrics = evaluator.evaluate(
        score_thresholds=[0.25, 0.75],
    )

    actual_metrics = [m.to_dict() for m in metrics[MetricType.Counts]]
    expected_metrics = [
        # score >= 0.25
        {
            "type": "Counts",
            "value": {
                "tp": 1,
                "fp": 0,
                "fn": 1,
                "tn": 1,
            },
            "parameters": {
                "score_threshold": 0.25,
                "hardmax": True,
                "label": "0",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 3,
            },
            "parameters": {
                "score_threshold": 0.25,
                "hardmax": True,
                "label": "1",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 0,
                "fp": 1,
                "fn": 0,
                "tn": 2,
            },
            "parameters": {
                "score_threshold": 0.25,
                "hardmax": True,
                "label": "2",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 1,
                "fp": 0,
                "fn": 0,
                "tn": 2,
            },
            "parameters": {
                "score_threshold": 0.25,
                "hardmax": True,
                "label": "3",
            },
        },
        # score >= 0.75
        {
            "type": "Counts",
            "value": {
                "tp": 1,
                "fp": 0,
                "fn": 1,
                "tn": 1,
            },
            "parameters": {
                "score_threshold": 0.75,
                "hardmax": True,
                "label": "0",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 3,
            },
            "parameters": {
                "score_threshold": 0.75,
                "hardmax": True,
                "label": "1",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 0,
                "fp": 1,
                "fn": 0,
                "tn": 2,
            },
            "parameters": {
                "score_threshold": 0.75,
                "hardmax": True,
                "label": "2",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 1,
                "tn": 2,
            },
            "parameters": {
                "score_threshold": 0.75,
                "hardmax": True,
                "label": "3",
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

    metrics = evaluator.evaluate(
        score_thresholds=[0.5],
    )

    actual_metrics = [m.to_dict() for m in metrics[MetricType.Counts]]
    expected_metrics = [
        {
            "type": "Counts",
            "value": {
                "tp": 1,
                "fp": 0,
                "fn": 2,
                "tn": 3,
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": True,
                "label": "0",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 1,
                "fp": 3,
                "fn": 0,
                "tn": 2,
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": True,
                "label": "1",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 0,
                "fp": 1,
                "fn": 2,
                "tn": 3,
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": True,
                "label": "2",
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_counts_with_animal_example(
    classifications_animal_example: list[Classification],
):

    loader = DataLoader()
    loader.add_data(classifications_animal_example)
    evaluator = loader.finalize()

    metrics = evaluator.evaluate(
        score_thresholds=[0.05, 0.5, 0.95],
    )

    actual_metrics = [m.to_dict() for m in metrics[MetricType.Counts]]
    expected_metrics = [
        # score >= 0.05
        {
            "type": "Counts",
            "value": {
                "tp": 1,
                "fp": 0,
                "fn": 2,
                "tn": 3,
            },
            "parameters": {
                "score_threshold": 0.05,
                "hardmax": True,
                "label": "bird",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 0,
                "fp": 1,
                "fn": 2,
                "tn": 3,
            },
            "parameters": {
                "score_threshold": 0.05,
                "hardmax": True,
                "label": "dog",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 1,
                "fp": 3,
                "fn": 0,
                "tn": 2,
            },
            "parameters": {
                "score_threshold": 0.05,
                "hardmax": True,
                "label": "cat",
            },
        },
        # score >= 0.5
        {
            "type": "Counts",
            "value": {
                "tp": 1,
                "fp": 0,
                "fn": 2,
                "tn": 3,
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": True,
                "label": "bird",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 0,
                "fp": 1,
                "fn": 2,
                "tn": 3,
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": True,
                "label": "dog",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 1,
                "fp": 2,
                "fn": 0,
                "tn": 3,
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": True,
                "label": "cat",
            },
        },
        # score >= 0.95
        {
            "type": "Counts",
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 3,
                "tn": 3,
            },
            "parameters": {
                "score_threshold": 0.95,
                "hardmax": True,
                "label": "bird",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 2,
                "tn": 4,
            },
            "parameters": {
                "score_threshold": 0.95,
                "hardmax": True,
                "label": "dog",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 1,
                "fp": 0,
                "fn": 0,
                "tn": 5,
            },
            "parameters": {
                "score_threshold": 0.95,
                "hardmax": True,
                "label": "cat",
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_counts_with_color_example(
    classifications_color_example: list[Classification],
):

    loader = DataLoader()
    loader.add_data(classifications_color_example)
    evaluator = loader.finalize()

    metrics = evaluator.evaluate(
        score_thresholds=[0.05, 0.5, 0.95],
    )

    actual_metrics = [m.to_dict() for m in metrics[MetricType.Counts]]
    expected_metrics = [
        # score >= 0.05
        {
            "type": "Counts",
            "value": {
                "tp": 1,
                "fp": 1,
                "fn": 1,
                "tn": 3,
            },
            "parameters": {
                "score_threshold": 0.05,
                "hardmax": True,
                "label": "white",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 2,
                "fp": 1,
                "fn": 0,
                "tn": 3,
            },
            "parameters": {
                "score_threshold": 0.05,
                "hardmax": True,
                "label": "red",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 0,
                "fp": 1,
                "fn": 1,
                "tn": 4,
            },
            "parameters": {
                "score_threshold": 0.05,
                "hardmax": True,
                "label": "blue",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 1,
                "tn": 5,
            },
            "parameters": {
                "score_threshold": 0.05,
                "hardmax": True,
                "label": "black",
            },
        },
        # score >= 0.5
        {
            "type": "Counts",
            "value": {
                "tp": 1,
                "fp": 1,
                "fn": 1,
                "tn": 3,
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": True,
                "label": "white",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 1,
                "fp": 1,
                "fn": 1,
                "tn": 3,
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": True,
                "label": "red",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 0,
                "fp": 1,
                "fn": 1,
                "tn": 4,
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": True,
                "label": "blue",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 1,
                "tn": 5,
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": True,
                "label": "black",
            },
        },
        # score >= 0.95
        {
            "type": "Counts",
            "value": {
                "tp": 0,
                "fp": 1,
                "fn": 2,
                "tn": 3,
            },
            "parameters": {
                "score_threshold": 0.95,
                "hardmax": True,
                "label": "white",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 2,
                "tn": 4,
            },
            "parameters": {
                "score_threshold": 0.95,
                "hardmax": True,
                "label": "red",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 1,
                "tn": 5,
            },
            "parameters": {
                "score_threshold": 0.95,
                "hardmax": True,
                "label": "blue",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 1,
                "tn": 5,
            },
            "parameters": {
                "score_threshold": 0.95,
                "hardmax": True,
                "label": "black",
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

    assert evaluator.ignored_prediction_labels == ["v1", "v8", "v5"]
    assert evaluator.missing_prediction_labels == []
    assert evaluator.metadata.to_dict() == {
        "number_of_datums": 2,
        "number_of_ground_truths": 2,
        "number_of_predictions": 4,
        "number_of_labels": 4,
    }
    metrics = evaluator.evaluate()

    actual_metrics = [m.to_dict() for m in metrics[MetricType.Counts]]
    expected_metrics = [
        {
            "type": "Counts",
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 1,
            },
            "parameters": {
                "score_threshold": 0.0,
                "hardmax": True,
                "label": "v1",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 1,
                "fp": 0,
                "fn": 1,
                "tn": 0,
            },
            "parameters": {
                "score_threshold": 0.0,
                "hardmax": True,
                "label": "v4",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 1,
            },
            "parameters": {
                "score_threshold": 0.0,
                "hardmax": True,
                "label": "v5",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 0,
                "fp": 1,
                "fn": 0,
                "tn": 0,
            },
            "parameters": {
                "score_threshold": 0.0,
                "hardmax": True,
                "label": "v8",
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

    assert evaluator.ignored_prediction_labels == []
    assert evaluator.missing_prediction_labels == []
    assert evaluator.metadata.to_dict() == {
        "number_of_datums": 10,
        "number_of_ground_truths": 10,
        "number_of_predictions": 30,
        "number_of_labels": 3,
    }

    metrics = evaluator.evaluate()

    actual_metrics = [m.to_dict() for m in metrics[MetricType.Counts]]
    expected_metrics = [
        {
            "type": "Counts",
            "value": {
                "tp": 3,
                "fp": 3,
                "fn": 0,
                "tn": 4,
            },
            "parameters": {
                "score_threshold": 0.0,
                "hardmax": True,
                "label": "0",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 2,
                "fp": 1,
                "fn": 4,
                "tn": 3,
            },
            "parameters": {
                "score_threshold": 0.0,
                "hardmax": True,
                "label": "1",
            },
        },
        {
            "type": "Counts",
            "value": {
                "tp": 0,
                "fp": 1,
                "fn": 1,
                "tn": 8,
            },
            "parameters": {
                "score_threshold": 0.0,
                "hardmax": True,
                "label": "2",
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_counts_multiclass(
    classifications_multiclass: list[Classification],
):
    loader = DataLoader()
    loader.add_data(classifications_multiclass)
    evaluator = loader.finalize()

    assert evaluator.ignored_prediction_labels == []
    assert evaluator.missing_prediction_labels == []
    assert evaluator.metadata.to_dict() == {
        "number_of_datums": 5,
        "number_of_ground_truths": 5,
        "number_of_labels": 3,
        "number_of_predictions": 15,
    }

    metrics = evaluator.evaluate(
        score_thresholds=[0.05, 0.1, 0.3, 0.85],
    )

    actual_metrics = [m.to_dict() for m in metrics[MetricType.Counts]]
    expected_metrics = [
        # score >= 0.05
        {
            "value": {
                "tp": 1,
                "fp": 0,
                "fn": 1,
                "tn": 3,
            },
            "parameters": {
                "score_threshold": 0.05,
                "hardmax": True,
                "label": "cat",
            },
            "type": "Counts",
        },
        {
            "value": {
                "tp": 1,
                "fp": 0,
                "fn": 0,
                "tn": 4,
            },
            "parameters": {
                "score_threshold": 0.05,
                "hardmax": True,
                "label": "dog",
            },
            "type": "Counts",
        },
        {
            "value": {
                "tp": 2,
                "fp": 1,
                "fn": 0,
                "tn": 2,
            },
            "parameters": {
                "score_threshold": 0.05,
                "hardmax": True,
                "label": "bee",
            },
            "type": "Counts",
        },
        # score >= 0.1
        {
            "value": {
                "tp": 1,
                "fp": 0,
                "fn": 1,
                "tn": 3,
            },
            "parameters": {
                "score_threshold": 0.1,
                "hardmax": True,
                "label": "cat",
            },
            "type": "Counts",
        },
        {
            "value": {
                "tp": 1,
                "fp": 0,
                "fn": 0,
                "tn": 4,
            },
            "parameters": {
                "score_threshold": 0.1,
                "hardmax": True,
                "label": "dog",
            },
            "type": "Counts",
        },
        {
            "value": {
                "tp": 2,
                "fp": 1,
                "fn": 0,
                "tn": 2,
            },
            "parameters": {
                "score_threshold": 0.1,
                "hardmax": True,
                "label": "bee",
            },
            "type": "Counts",
        },
        # score >= 0.3
        {
            "value": {
                "tp": 1,
                "fp": 0,
                "fn": 1,
                "tn": 3,
            },
            "parameters": {
                "score_threshold": 0.3,
                "hardmax": True,
                "label": "cat",
            },
            "type": "Counts",
        },
        {
            "value": {
                "tp": 1,
                "fp": 0,
                "fn": 0,
                "tn": 4,
            },
            "parameters": {
                "score_threshold": 0.3,
                "hardmax": True,
                "label": "dog",
            },
            "type": "Counts",
        },
        {
            "value": {
                "tp": 2,
                "fp": 1,
                "fn": 0,
                "tn": 2,
            },
            "parameters": {
                "score_threshold": 0.3,
                "hardmax": True,
                "label": "bee",
            },
            "type": "Counts",
        },
        # score >= 0.85
        {
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 2,
                "tn": 3,
            },
            "parameters": {
                "score_threshold": 0.85,
                "hardmax": True,
                "label": "cat",
            },
            "type": "Counts",
        },
        {
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 1,
                "tn": 4,
            },
            "parameters": {
                "score_threshold": 0.85,
                "hardmax": True,
                "label": "dog",
            },
            "type": "Counts",
        },
        {
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 2,
                "tn": 3,
            },
            "parameters": {
                "score_threshold": 0.85,
                "hardmax": True,
                "label": "bee",
            },
            "type": "Counts",
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_counts_true_negatives_check_animals(
    classifications_multiclass_true_negatives_check: list[Classification],
):
    loader = DataLoader()
    loader.add_data(classifications_multiclass_true_negatives_check)
    evaluator = loader.finalize()

    assert evaluator.ignored_prediction_labels == ["bee", "cat"]
    assert evaluator.missing_prediction_labels == []
    assert evaluator.metadata.to_dict() == {
        "number_of_datums": 1,
        "number_of_ground_truths": 1,
        "number_of_predictions": 3,
        "number_of_labels": 3,
    }
    metrics = evaluator.evaluate(
        score_thresholds=[0.05, 0.15, 0.95],
    )

    actual_metrics = [m.to_dict() for m in metrics[MetricType.Counts]]
    expected_metrics = [
        # score >= 0.05
        {
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 1,
                "tn": 0,
            },
            "parameters": {
                "score_threshold": 0.05,
                "hardmax": True,
                "label": "ant",
            },
            "type": "Counts",
        },
        {
            "value": {
                "tp": 0,
                "fp": 1,
                "fn": 0,
                "tn": 0,
            },
            "parameters": {
                "score_threshold": 0.05,
                "hardmax": True,
                "label": "bee",
            },
            "type": "Counts",
        },
        {
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 1,
            },
            "parameters": {
                "score_threshold": 0.05,
                "hardmax": True,
                "label": "cat",
            },
            "type": "Counts",
        },
        # score >= 0.15
        {
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 1,
                "tn": 0,
            },
            "parameters": {
                "score_threshold": 0.15,
                "hardmax": True,
                "label": "ant",
            },
            "type": "Counts",
        },
        {
            "value": {
                "tp": 0,
                "fp": 1,
                "fn": 0,
                "tn": 0,
            },
            "parameters": {
                "score_threshold": 0.15,
                "hardmax": True,
                "label": "bee",
            },
            "type": "Counts",
        },
        {
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 1,
            },
            "parameters": {
                "score_threshold": 0.15,
                "hardmax": True,
                "label": "cat",
            },
            "type": "Counts",
        },
        # score >= 0.95
        {
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 1,
                "tn": 0,
            },
            "parameters": {
                "score_threshold": 0.95,
                "hardmax": True,
                "label": "ant",
            },
            "type": "Counts",
        },
        {
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 1,
            },
            "parameters": {
                "score_threshold": 0.95,
                "hardmax": True,
                "label": "bee",
            },
            "type": "Counts",
        },
        {
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 1,
            },
            "parameters": {
                "score_threshold": 0.95,
                "hardmax": True,
                "label": "cat",
            },
            "type": "Counts",
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_counts_zero_count_check(
    classifications_multiclass_zero_count: list[Classification],
):

    loader = DataLoader()
    loader.add_data(classifications_multiclass_zero_count)
    evaluator = loader.finalize()

    assert evaluator.ignored_prediction_labels == ["bee", "cat"]
    assert evaluator.missing_prediction_labels == []
    assert evaluator.metadata.to_dict() == {
        "number_of_datums": 1,
        "number_of_ground_truths": 1,
        "number_of_labels": 3,
        "number_of_predictions": 3,
    }

    metrics = evaluator.evaluate(
        score_thresholds=[0.05, 0.2, 0.95],
    )

    actual_metrics = [m.to_dict() for m in metrics[MetricType.Counts]]
    expected_metrics = [
        # score >= 0.05
        {
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 1,
                "tn": 0,
            },
            "parameters": {
                "score_threshold": 0.05,
                "hardmax": True,
                "label": "ant",
            },
            "type": "Counts",
        },
        {
            "value": {
                "tp": 0,
                "fp": 1,
                "fn": 0,
                "tn": 0,
            },
            "parameters": {
                "score_threshold": 0.05,
                "hardmax": True,
                "label": "bee",
            },
            "type": "Counts",
        },
        {
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 1,
            },
            "parameters": {
                "score_threshold": 0.05,
                "hardmax": True,
                "label": "cat",
            },
            "type": "Counts",
        },
        # score >= 0.2
        {
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 1,
                "tn": 0,
            },
            "parameters": {
                "score_threshold": 0.2,
                "hardmax": True,
                "label": "ant",
            },
            "type": "Counts",
        },
        {
            "value": {
                "tp": 0,
                "fp": 1,
                "fn": 0,
                "tn": 0,
            },
            "parameters": {
                "score_threshold": 0.2,
                "hardmax": True,
                "label": "bee",
            },
            "type": "Counts",
        },
        {
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 1,
            },
            "parameters": {
                "score_threshold": 0.2,
                "hardmax": True,
                "label": "cat",
            },
            "type": "Counts",
        },
        # score >= 0.95
        {
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 1,
                "tn": 0,
            },
            "parameters": {
                "score_threshold": 0.95,
                "hardmax": True,
                "label": "ant",
            },
            "type": "Counts",
        },
        {
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 1,
            },
            "parameters": {
                "score_threshold": 0.95,
                "hardmax": True,
                "label": "bee",
            },
            "type": "Counts",
        },
        {
            "value": {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 1,
            },
            "parameters": {
                "score_threshold": 0.95,
                "hardmax": True,
                "label": "cat",
            },
            "type": "Counts",
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics
