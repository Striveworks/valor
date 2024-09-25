import numpy as np
from valor_lite.classification import (
    Classification,
    DataLoader,
    MetricType,
    compute_metrics,
)


def test_confusion_matrix_computation():

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

    (_, _, _, _, _, _, _, confusion) = compute_metrics(
        data=data,
        label_metadata=label_metadata,
        score_thresholds=score_thresholds,
        n_datums=3,
        hardmax=False,
    )

    # score threshold, prediction label, groundtruth label
    assert confusion.shape == (2, 4, 4)

    # label 0
    # score >= 0.25
    assert confusion[0][0][0] == 1
    assert confusion[0][0][1] == 0
    assert confusion[0][0][2] == 0
    assert confusion[0][0][3] == 0
    # score >= 0.75
    assert confusion[1][0][0] == 1
    assert confusion[1][0][1] == 0
    assert confusion[1][0][2] == 0
    assert confusion[1][0][3] == 0

    # label 1
    # score >= 0.25
    assert confusion[0][1][0] == 0
    assert confusion[0][1][1] == 0
    assert confusion[0][1][2] == 0
    assert confusion[0][1][3] == 0
    # score >= 0.75
    assert confusion[1][1][0] == 0
    assert confusion[1][1][1] == 0
    assert confusion[1][1][2] == 0
    assert confusion[1][1][3] == 0

    # label 2
    # score >= 0.25
    assert confusion[0][2][0] == 1
    assert confusion[0][2][1] == 0
    assert confusion[0][2][2] == 0
    assert confusion[0][2][3] == 0
    # score >= 0.75
    assert confusion[1][2][0] == 1
    assert confusion[1][2][1] == 0
    assert confusion[1][2][2] == 0
    assert confusion[1][2][3] == 0

    # label 3
    # score >= 0.25
    assert confusion[0][3][0] == 0
    assert confusion[0][3][1] == 0
    assert confusion[0][3][2] == 0
    assert confusion[0][3][3] == 1
    # score >= 0.75
    assert confusion[1][3][0] == 0
    assert confusion[1][3][1] == 0
    assert confusion[1][3][2] == 0
    assert confusion[1][3][3] == 0


def test_confusion_matrix_basic(classifications_basic: list[Classification]):
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

    # test ConfusionMatrix
    actual_metrics = [m.to_dict() for m in metrics[MetricType.ConfusionMatrix]]
    expected_metrics = [
        {
            "type": "ConfusionMatrix",
            "value": {
                "0": {"0": 1, "1": 0, "2": 0, "3": 0},
                "1": {"0": 0, "1": 0, "2": 0, "3": 0},
                "2": {"0": 1, "1": 0, "2": 0, "3": 0},
                "3": {"0": 0, "1": 0, "2": 0, "3": 1},
            },
            "parameters": {
                "score_threshold": 0.25,
                "hardmax": True,
                "label_key": "class",
            },
        },
        {
            "type": "ConfusionMatrix",
            "value": {
                "0": {"0": 1, "1": 0, "2": 0, "3": 0},
                "1": {"0": 0, "1": 0, "2": 0, "3": 0},
                "2": {"0": 1, "1": 0, "2": 0, "3": 0},
                "3": {"0": 0, "1": 0, "2": 0, "3": 0},
            },
            "parameters": {
                "score_threshold": 0.75,
                "hardmax": True,
                "label_key": "class",
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_confusion_matrix_api_unit_test(
    classifications_from_api_unit_tests: list[Classification],
):

    loader = DataLoader()
    loader.add_data(classifications_from_api_unit_tests)
    evaluator = loader.finalize()

    metrics = evaluator.evaluate(score_thresholds=[0.5])

    # test ConfusionMatrix
    actual_metrics = [m.to_dict() for m in metrics[MetricType.ConfusionMatrix]]
    expected_metrics = [
        {
            "type": "ConfusionMatrix",
            "value": {
                "0": {"0": 1, "1": 0, "2": 0},
                "1": {"0": 1, "1": 1, "2": 2},
                "2": {"0": 1, "1": 0, "2": 0},
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": True,
                "label_key": "class",
            },
        }
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_confusion_matrix_with_example(
    classifications_two_categeories: list[Classification],
):

    loader = DataLoader()
    loader.add_data(classifications_two_categeories)
    evaluator = loader.finalize()

    metrics = evaluator.evaluate(score_thresholds=[0.5])

    # test ConfusionMatrix
    actual_metrics = [m.to_dict() for m in metrics[MetricType.ConfusionMatrix]]
    expected_metrics = [
        {
            "type": "ConfusionMatrix",
            "value": {
                "bird": {"bird": 1, "dog": 0, "cat": 0},
                "dog": {"bird": 1, "dog": 0, "cat": 0},
                "cat": {"bird": 1, "dog": 1, "cat": 1},
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": True,
                "label_key": "animal",
            },
        },
        {
            "type": "ConfusionMatrix",
            "value": {
                "white": {"white": 1, "red": 0, "blue": 1, "black": 0},
                "red": {"white": 0, "red": 1, "blue": 0, "black": 1},
                "blue": {"white": 1, "red": 0, "blue": 0, "black": 0},
                "black": {"white": 0, "red": 0, "blue": 0, "black": 0},
            },
            "parameters": {
                "score_threshold": 0.5,
                "hardmax": True,
                "label_key": "color",
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_confusion_matrix_with_image_example(
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

    # test ConfusionMatrix
    actual_metrics = [m.to_dict() for m in metrics[MetricType.ConfusionMatrix]]
    expected_metrics = [
        # k3
        {
            "type": "ConfusionMatrix",
            "value": {
                "v1": {
                    "v1": 0,
                    "v3": 1,
                },
                "v3": {
                    "v1": 0,
                    "v3": 0,
                },
            },
            "parameters": {
                "score_threshold": 0.0,
                "hardmax": True,
                "label_key": "k3",
            },
        },
        # k4
        {
            "type": "ConfusionMatrix",
            "value": {
                "v1": {
                    "v1": 0,
                    "v4": 0,
                    "v5": 0,
                    "v8": 0,
                },
                "v4": {
                    "v1": 0,
                    "v4": 1,
                    "v5": 0,
                    "v8": 0,
                },
                "v5": {
                    "v1": 0,
                    "v4": 0,
                    "v5": 0,
                    "v8": 0,
                },
                "v8": {
                    "v1": 0,
                    "v4": 1,
                    "v5": 0,
                    "v8": 0,
                },
            },
            "parameters": {
                "score_threshold": 0.0,
                "hardmax": True,
                "label_key": "k4",
            },
        },
        # k5
        {
            "type": "ConfusionMatrix",
            "value": {
                "v1": {
                    "v1": 0,
                    "v5": 1,
                },
                "v5": {
                    "v1": 0,
                    "v5": 0,
                },
            },
            "parameters": {
                "score_threshold": 0.0,
                "hardmax": True,
                "label_key": "k5",
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_confusion_matrix_with_tabular_example(
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

    # test ConfusionMatrix
    actual_metrics = [m.to_dict() for m in metrics[MetricType.ConfusionMatrix]]
    expected_metrics = [
        {
            "type": "ConfusionMatrix",
            "value": {
                "0": {"0": 3, "1": 3, "2": 0},
                "1": {"0": 0, "1": 2, "2": 1},
                "2": {"0": 0, "1": 1, "2": 0},
            },
            "parameters": {
                "score_threshold": 0.0,
                "hardmax": True,
                "label_key": "class",
            },
        }
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_confusion_matrix_mutliclass(
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

    # test ConfusionMatrix
    actual_metrics = [m.to_dict() for m in metrics[MetricType.ConfusionMatrix]]
    expected_metrics = [
        {
            "type": "ConfusionMatrix",
            "value": {
                "cat": {"cat": 1, "dog": 0, "bee": 0},
                "dog": {"cat": 0, "dog": 1, "bee": 0},
                "bee": {"cat": 1, "dog": 0, "bee": 2},
            },
            "parameters": {
                "score_threshold": 0.05,
                "hardmax": True,
                "label_key": "class_label",
            },
        },
        {
            "type": "ConfusionMatrix",
            "value": {
                "cat": {"cat": 1, "dog": 0, "bee": 0},
                "dog": {"cat": 0, "dog": 1, "bee": 0},
                "bee": {"cat": 1, "dog": 0, "bee": 2},
            },
            "parameters": {
                "score_threshold": 0.1,
                "hardmax": True,
                "label_key": "class_label",
            },
        },
        {
            "type": "ConfusionMatrix",
            "value": {
                "cat": {"cat": 1, "dog": 0, "bee": 0},
                "dog": {"cat": 0, "dog": 1, "bee": 0},
                "bee": {"cat": 1, "dog": 0, "bee": 2},
            },
            "parameters": {
                "score_threshold": 0.3,
                "hardmax": True,
                "label_key": "class_label",
            },
        },
        {
            "type": "ConfusionMatrix",
            "value": {
                "cat": {"cat": 0, "dog": 0, "bee": 0},
                "dog": {"cat": 0, "dog": 0, "bee": 0},
                "bee": {"cat": 0, "dog": 0, "bee": 0},
            },
            "parameters": {
                "score_threshold": 0.85,
                "hardmax": True,
                "label_key": "class_label",
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_confusion_matrix_true_negatives_check(
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
        score_thresholds=[0.47, 0.49],
    )

    # test ConfusionMatrix
    actual_metrics = [m.to_dict() for m in metrics[MetricType.ConfusionMatrix]]
    expected_metrics = [
        {
            "type": "ConfusionMatrix",
            "value": {
                "ant": {"ant": 0, "bee": 0, "cat": 0},
                "bee": {"ant": 1, "bee": 0, "cat": 0},
                "cat": {"ant": 0, "bee": 0, "cat": 0},
            },
            "parameters": {
                "score_threshold": 0.47,
                "hardmax": True,
                "label_key": "k1",
            },
        },
        {
            "type": "ConfusionMatrix",
            "value": {
                "egg": {"egg": 0, "milk": 0, "flour": 0},
                "milk": {"egg": 1, "milk": 0, "flour": 0},
                "flour": {"egg": 0, "milk": 0, "flour": 0},
            },
            "parameters": {
                "score_threshold": 0.47,
                "hardmax": True,
                "label_key": "k2",
            },
        },
        {
            "type": "ConfusionMatrix",
            "value": {
                "ant": {"ant": 0, "bee": 0, "cat": 0},
                "bee": {"ant": 0, "bee": 0, "cat": 0},
                "cat": {"ant": 0, "bee": 0, "cat": 0},
            },
            "parameters": {
                "score_threshold": 0.49,
                "hardmax": True,
                "label_key": "k1",
            },
        },
        {
            "type": "ConfusionMatrix",
            "value": {
                "egg": {"egg": 0, "milk": 0, "flour": 0},
                "milk": {"egg": 0, "milk": 0, "flour": 0},
                "flour": {"egg": 0, "milk": 0, "flour": 0},
            },
            "parameters": {
                "score_threshold": 0.49,
                "hardmax": True,
                "label_key": "k2",
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_confusion_matrix_zero_count_check(
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

    metrics = evaluator.evaluate(score_thresholds=[0.47, 0.49])

    # test ConfusionMatrix
    actual_metrics = [m.to_dict() for m in metrics[MetricType.ConfusionMatrix]]
    expected_metrics = [
        {
            "type": "ConfusionMatrix",
            "value": {
                "ant": {"ant": 0, "bee": 0, "cat": 0},
                "bee": {"ant": 1, "bee": 0, "cat": 0},
                "cat": {"ant": 0, "bee": 0, "cat": 0},
            },
            "parameters": {
                "score_threshold": 0.47,
                "hardmax": True,
                "label_key": "k",
            },
        },
        {
            "type": "ConfusionMatrix",
            "value": {
                "ant": {"ant": 0, "bee": 0, "cat": 0},
                "bee": {"ant": 0, "bee": 0, "cat": 0},
                "cat": {"ant": 0, "bee": 0, "cat": 0},
            },
            "parameters": {
                "score_threshold": 0.49,
                "hardmax": True,
                "label_key": "k",
            },
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics
