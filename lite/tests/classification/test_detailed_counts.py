import numpy as np
from valor_lite.classification import Classification, DataLoader
from valor_lite.classification.computation import compute_detailed_counts


def test_compute_detailed_counts():

    # groundtruth, prediction, score
    data = np.array(
        [
            # datum 0
            [0, 0, 0, 1.0],  # tp
            [0, 0, 1, 0.0],  # tn
            [0, 0, 2, 0.0],  # tn
            [0, 0, 3, 0.0],  # tn
            # datum 1
            [1, 0, 0, 0.0],  # fn
            [1, 0, 1, 0.0],  # tn
            [1, 0, 2, 1.0],  # fp
            [1, 0, 3, 0.0],  # tn
            # datum 2
            [2, 3, 0, 0.0],  # tn
            [2, 3, 1, 0.0],  # tn
            [2, 3, 2, 0.0],  # tn
            [2, 3, 3, 0.3],  # fn for score threshold > 0.3
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

    detailed_counts = compute_detailed_counts(
        data=data,
        label_metadata=label_metadata,
        score_thresholds=score_thresholds,
        n_samples=0,
    )

    assert (
        detailed_counts
        == np.array(
            [
                [
                    #    tp  fp  fn  fn  tn
                    [1.0, 0.0, 1.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 3.0],
                    [0.0, 1.0, 0.0, 0.0, 2.0],
                    [1.0, 0.0, 0.0, 0.0, 2.0],
                ],
                [
                    [1.0, 0.0, 1.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 3.0],
                    [0.0, 1.0, 0.0, 0.0, 2.0],
                    [0.0, 0.0, 0.0, 1.0, 2.0],
                ],
            ]
        )
    ).all()


def test_detailed_counts_basic(classifications_basic: list[Classification]):
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

    metrics = evaluator.compute_detailed_counts(
        score_thresholds=[0.25, 0.75],
        n_samples=1,
    )

    # test DetailedCounts
    actual_metrics = [m.to_dict() for m in metrics]
    expected_metrics = [
        {
            "value": {
                "tp": [1, 1],
                "fp_misclassification": [0, 0],
                "fn_misclassification": [1, 1],
                "fn_missing_prediction": [0, 0],
                "tn": [1, 1],
                "tp_examples": [["uid0"], ["uid0"]],
                "fp_misclassification_examples": [[], []],
                "fn_misclassification_examples": [["uid1"], ["uid1"]],
                "fn_missing_prediction_examples": [[], []],
                "tn_examples": [["uid2"], ["uid2"]],
            },
            "parameters": {
                "score_thresholds": [0.25, 0.75],
                "label": {"key": "class", "value": "0"},
            },
            "type": "DetailedCounts",
        },
        {
            "value": {
                "tp": [0, 0],
                "fp_misclassification": [0, 0],
                "fn_misclassification": [0, 0],
                "fn_missing_prediction": [0, 0],
                "tn": [3, 3],
                "tp_examples": [[], []],
                "fp_misclassification_examples": [[], []],
                "fn_misclassification_examples": [[], []],
                "fn_missing_prediction_examples": [[], []],
                "tn_examples": [["uid0"], ["uid0"]],
            },
            "parameters": {
                "score_thresholds": [0.25, 0.75],
                "label": {"key": "class", "value": "1"},
            },
            "type": "DetailedCounts",
        },
        {
            "value": {
                "tp": [0, 0],
                "fp_misclassification": [1, 1],
                "fn_misclassification": [0, 0],
                "fn_missing_prediction": [0, 0],
                "tn": [2, 2],
                "tp_examples": [[], []],
                "fp_misclassification_examples": [["uid1"], ["uid1"]],
                "fn_misclassification_examples": [[], []],
                "fn_missing_prediction_examples": [[], []],
                "tn_examples": [["uid0"], ["uid0"]],
            },
            "parameters": {
                "score_thresholds": [0.25, 0.75],
                "label": {"key": "class", "value": "2"},
            },
            "type": "DetailedCounts",
        },
        {
            "value": {
                "tp": [1, 0],
                "fp_misclassification": [0, 0],
                "fn_misclassification": [0, 0],
                "fn_missing_prediction": [0, 1],
                "tn": [2, 2],
                "tp_examples": [["uid2"], []],
                "fp_misclassification_examples": [[], []],
                "fn_misclassification_examples": [[], []],
                "fn_missing_prediction_examples": [[], ["uid2"]],
                "tn_examples": [["uid0"], ["uid0"]],
            },
            "parameters": {
                "score_thresholds": [0.25, 0.75],
                "label": {"key": "class", "value": "3"},
            },
            "type": "DetailedCounts",
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_detailed_counts_unit(
    classifications_from_api_unit_tests: list[Classification],
):

    loader = DataLoader()
    loader.add_data(classifications_from_api_unit_tests)
    evaluator = loader.finalize()

    metrics = evaluator.compute_detailed_counts(score_thresholds=[0.5])

    # test DetailedCounts
    actual_metrics = [m.to_dict() for m in metrics]
    expected_metrics = [
        {
            "value": {
                "tp": [1],
                "fp_misclassification": [0],
                "fn_misclassification": [2],
                "fn_missing_prediction": [0],
                "tn": [3],
                "tp_examples": [[]],
                "fp_misclassification_examples": [[]],
                "fn_misclassification_examples": [[]],
                "fn_missing_prediction_examples": [[]],
                "tn_examples": [[]],
            },
            "parameters": {
                "score_thresholds": [0.5],
                "label": {"key": "class", "value": "0"},
            },
            "type": "DetailedCounts",
        },
        {
            "value": {
                "tp": [1],
                "fp_misclassification": [3],
                "fn_misclassification": [0],
                "fn_missing_prediction": [0],
                "tn": [2],
                "tp_examples": [[]],
                "fp_misclassification_examples": [[]],
                "fn_misclassification_examples": [[]],
                "fn_missing_prediction_examples": [[]],
                "tn_examples": [[]],
            },
            "parameters": {
                "score_thresholds": [0.5],
                "label": {"key": "class", "value": "1"},
            },
            "type": "DetailedCounts",
        },
        {
            "value": {
                "tp": [0],
                "fp_misclassification": [1],
                "fn_misclassification": [2],
                "fn_missing_prediction": [0],
                "tn": [3],
                "tp_examples": [[]],
                "fp_misclassification_examples": [[]],
                "fn_misclassification_examples": [[]],
                "fn_missing_prediction_examples": [[]],
                "tn_examples": [[]],
            },
            "parameters": {
                "score_thresholds": [0.5],
                "label": {"key": "class", "value": "2"},
            },
            "type": "DetailedCounts",
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_detailed_counts_with_example(
    classifications_two_categeories: list[Classification],
):

    loader = DataLoader()
    loader.add_data(classifications_two_categeories)
    evaluator = loader.finalize()

    metrics = evaluator.compute_detailed_counts(
        score_thresholds=[0.5],
        n_samples=6,
    )

    # test DetailedCounts
    actual_metrics = [m.to_dict() for m in metrics]
    expected_metrics = [
        {
            "value": {
                "tp": [1],
                "fp_misclassification": [0],
                "fn_misclassification": [2],
                "fn_missing_prediction": [0],
                "tn": [3],
                "tp_examples": [["uid0"]],
                "fp_misclassification_examples": [[]],
                "fn_misclassification_examples": [["uid2", "uid3"]],
                "fn_missing_prediction_examples": [[]],
                "tn_examples": [["uid5", "uid1", "uid4"]],
            },
            "parameters": {
                "score_thresholds": [0.5],
                "label": {"key": "animal", "value": "bird"},
            },
            "type": "DetailedCounts",
        },
        {
            "value": {
                "tp": [0],
                "fp_misclassification": [1],
                "fn_misclassification": [2],
                "fn_missing_prediction": [0],
                "tn": [3],
                "tp_examples": [[]],
                "fp_misclassification_examples": [["uid3"]],
                "fn_misclassification_examples": [["uid5", "uid1"]],
                "fn_missing_prediction_examples": [[]],
                "tn_examples": [["uid0", "uid2", "uid4"]],
            },
            "parameters": {
                "score_thresholds": [0.5],
                "label": {"key": "animal", "value": "dog"},
            },
            "type": "DetailedCounts",
        },
        {
            "value": {
                "tp": [1],
                "fp_misclassification": [2],
                "fn_misclassification": [0],
                "fn_missing_prediction": [0],
                "tn": [3],
                "tp_examples": [["uid4"]],
                "fp_misclassification_examples": [["uid1", "uid2"]],
                "fn_misclassification_examples": [[]],
                "fn_missing_prediction_examples": [[]],
                "tn_examples": [["uid5", "uid0", "uid3"]],
            },
            "parameters": {
                "score_thresholds": [0.5],
                "label": {"key": "animal", "value": "cat"},
            },
            "type": "DetailedCounts",
        },
        {
            "value": {
                "tp": [1],
                "fp_misclassification": [1],
                "fn_misclassification": [1],
                "fn_missing_prediction": [0],
                "tn": [3],
                "tp_examples": [["uid0"]],
                "fp_misclassification_examples": [["uid3"]],
                "fn_misclassification_examples": [["uid1"]],
                "fn_missing_prediction_examples": [[]],
                "tn_examples": [["uid2", "uid5", "uid4"]],
            },
            "parameters": {
                "score_thresholds": [0.5],
                "label": {"key": "color", "value": "white"},
            },
            "type": "DetailedCounts",
        },
        {
            "value": {
                "tp": [1],
                "fp_misclassification": [1],
                "fn_misclassification": [1],
                "fn_missing_prediction": [0],
                "tn": [3],
                "tp_examples": [["uid5"]],
                "fp_misclassification_examples": [["uid4"]],
                "fn_misclassification_examples": [["uid2"]],
                "fn_missing_prediction_examples": [[]],
                "tn_examples": [["uid0", "uid1", "uid3"]],
            },
            "parameters": {
                "score_thresholds": [0.5],
                "label": {"key": "color", "value": "red"},
            },
            "type": "DetailedCounts",
        },
        {
            "value": {
                "tp": [0],
                "fp_misclassification": [1],
                "fn_misclassification": [1],
                "fn_missing_prediction": [0],
                "tn": [4],
                "tp_examples": [[]],
                "fp_misclassification_examples": [["uid1"]],
                "fn_misclassification_examples": [["uid3"]],
                "fn_missing_prediction_examples": [[]],
                "tn_examples": [["uid0", "uid4", "uid2", "uid5"]],
            },
            "parameters": {
                "score_thresholds": [0.5],
                "label": {"key": "color", "value": "blue"},
            },
            "type": "DetailedCounts",
        },
        {
            "value": {
                "tp": [0],
                "fp_misclassification": [0],
                "fn_misclassification": [1],
                "fn_missing_prediction": [0],
                "tn": [5],
                "tp_examples": [[]],
                "fp_misclassification_examples": [[]],
                "fn_misclassification_examples": [["uid4"]],
                "fn_missing_prediction_examples": [[]],
                "tn_examples": [["uid2", "uid1", "uid0", "uid5", "uid3"]],
            },
            "parameters": {
                "score_thresholds": [0.5],
                "label": {"key": "color", "value": "black"},
            },
            "type": "DetailedCounts",
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_detailed_counts_mutliclass(
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

    metrics = evaluator.compute_detailed_counts(
        score_thresholds=[0.05, 0.1, 0.3, 0.85],
        n_samples=5,
    )

    # test DetailedCounts
    actual_metrics = [m.to_dict() for m in metrics]
    expected_metrics = [
        {
            "value": {
                "tp": [2, 2, 1, 0],
                "fp_misclassification": [3, 3, 2, 0],
                "fn_misclassification": [0, 0, 1, 0],
                "fn_missing_prediction": [0, 0, 0, 2],
                "tn": [0, 0, 1, 3],
                "tp_examples": [
                    ["uid0", "uid2"],
                    ["uid0", "uid2"],
                    ["uid0"],
                    [],
                ],
                "fp_misclassification_examples": [
                    ["uid1", "uid4", "uid3"],
                    ["uid1", "uid4", "uid3"],
                    ["uid1", "uid4"],
                    [],
                ],
                "fn_misclassification_examples": [[], [], ["uid2"], []],
                "fn_missing_prediction_examples": [
                    [],
                    [],
                    [],
                    ["uid0", "uid2"],
                ],
                "tn_examples": [[], [], ["uid3"], ["uid1", "uid4", "uid3"]],
            },
            "parameters": {
                "score_thresholds": [0.05, 0.1, 0.3, 0.85],
                "label": {"key": "class_label", "value": "cat"},
            },
            "type": "DetailedCounts",
        },
        {
            "value": {
                "tp": [1, 1, 1, 0],
                "fp_misclassification": [4, 4, 2, 0],
                "fn_misclassification": [0, 0, 0, 0],
                "fn_missing_prediction": [0, 0, 0, 1],
                "tn": [0, 0, 2, 4],
                "tp_examples": [["uid4"], ["uid4"], ["uid4"], []],
                "fp_misclassification_examples": [
                    ["uid2", "uid0", "uid3", "uid1"],
                    ["uid2", "uid0", "uid3", "uid1"],
                    ["uid2", "uid0"],
                    [],
                ],
                "fn_misclassification_examples": [[], [], [], []],
                "fn_missing_prediction_examples": [[], [], [], ["uid4"]],
                "tn_examples": [
                    [],
                    [],
                    ["uid3", "uid1"],
                    ["uid2", "uid0", "uid3", "uid1"],
                ],
            },
            "parameters": {
                "score_thresholds": [0.05, 0.1, 0.3, 0.85],
                "label": {"key": "class_label", "value": "dog"},
            },
            "type": "DetailedCounts",
        },
        {
            "value": {
                "tp": [2, 2, 2, 0],
                "fp_misclassification": [3, 2, 1, 0],
                "fn_misclassification": [0, 0, 0, 0],
                "fn_missing_prediction": [0, 0, 0, 2],
                "tn": [0, 1, 2, 3],
                "tp_examples": [
                    ["uid3", "uid1"],
                    ["uid3", "uid1"],
                    ["uid3", "uid1"],
                    [],
                ],
                "fp_misclassification_examples": [
                    ["uid2", "uid0", "uid4"],
                    ["uid2", "uid0"],
                    ["uid2"],
                    [],
                ],
                "fn_misclassification_examples": [[], [], [], []],
                "fn_missing_prediction_examples": [
                    [],
                    [],
                    [],
                    ["uid3", "uid1"],
                ],
                "tn_examples": [
                    [],
                    ["uid4"],
                    ["uid0", "uid4"],
                    ["uid2", "uid0", "uid4"],
                ],
            },
            "parameters": {
                "score_thresholds": [0.05, 0.1, 0.3, 0.85],
                "label": {"key": "class_label", "value": "bee"},
            },
            "type": "DetailedCounts",
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_detailed_counts_true_negatives_check(
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

    metrics = evaluator.compute_detailed_counts(
        score_thresholds=[0.05, 0.15, 0.95],
        n_samples=6,
    )

    # test DetailedCounts
    actual_metrics = [m.to_dict() for m in metrics]
    expected_metrics = [
        {
            "value": {
                "tp": [1, 1, 0],
                "fp_misclassification": [0, 0, 0],
                "fn_misclassification": [0, 0, 0],
                "fn_missing_prediction": [0, 0, 1],
                "tn": [0, 0, 0],
                "tp_examples": [["uid1"], ["uid1"], []],
                "fp_misclassification_examples": [[], [], []],
                "fn_misclassification_examples": [[], [], []],
                "fn_missing_prediction_examples": [[], [], ["uid1"]],
                "tn_examples": [[], [], []],
            },
            "parameters": {
                "score_thresholds": [0.05, 0.15, 0.95],
                "label": {"key": "k1", "value": "ant"},
            },
            "type": "DetailedCounts",
        },
        {
            "value": {
                "tp": [0, 0, 0],
                "fp_misclassification": [1, 1, 0],
                "fn_misclassification": [0, 0, 0],
                "fn_missing_prediction": [0, 0, 0],
                "tn": [0, 0, 1],
                "tp_examples": [[], [], []],
                "fp_misclassification_examples": [["uid1"], ["uid1"], []],
                "fn_misclassification_examples": [[], [], []],
                "fn_missing_prediction_examples": [[], [], []],
                "tn_examples": [[], [], ["uid1"]],
            },
            "parameters": {
                "score_thresholds": [0.05, 0.15, 0.95],
                "label": {"key": "k1", "value": "bee"},
            },
            "type": "DetailedCounts",
        },
        {
            "value": {
                "tp": [0, 0, 0],
                "fp_misclassification": [1, 1, 0],
                "fn_misclassification": [0, 0, 0],
                "fn_missing_prediction": [0, 0, 0],
                "tn": [0, 0, 1],
                "tp_examples": [[], [], []],
                "fp_misclassification_examples": [["uid1"], ["uid1"], []],
                "fn_misclassification_examples": [[], [], []],
                "fn_missing_prediction_examples": [[], [], []],
                "tn_examples": [[], [], ["uid1"]],
            },
            "parameters": {
                "score_thresholds": [0.05, 0.15, 0.95],
                "label": {"key": "k1", "value": "cat"},
            },
            "type": "DetailedCounts",
        },
        {
            "value": {
                "tp": [1, 1, 0],
                "fp_misclassification": [0, 0, 0],
                "fn_misclassification": [0, 0, 0],
                "fn_missing_prediction": [0, 0, 1],
                "tn": [0, 0, 0],
                "tp_examples": [["uid2"], ["uid2"], []],
                "fp_misclassification_examples": [[], [], []],
                "fn_misclassification_examples": [[], [], []],
                "fn_missing_prediction_examples": [[], [], ["uid2"]],
                "tn_examples": [[], [], []],
            },
            "parameters": {
                "score_thresholds": [0.05, 0.15, 0.95],
                "label": {"key": "k2", "value": "egg"},
            },
            "type": "DetailedCounts",
        },
        {
            "value": {
                "tp": [0, 0, 0],
                "fp_misclassification": [1, 1, 0],
                "fn_misclassification": [0, 0, 0],
                "fn_missing_prediction": [0, 0, 0],
                "tn": [0, 0, 1],
                "tp_examples": [[], [], []],
                "fp_misclassification_examples": [["uid2"], ["uid2"], []],
                "fn_misclassification_examples": [[], [], []],
                "fn_missing_prediction_examples": [[], [], []],
                "tn_examples": [[], [], ["uid2"]],
            },
            "parameters": {
                "score_thresholds": [0.05, 0.15, 0.95],
                "label": {"key": "k2", "value": "milk"},
            },
            "type": "DetailedCounts",
        },
        {
            "value": {
                "tp": [0, 0, 0],
                "fp_misclassification": [1, 1, 0],
                "fn_misclassification": [0, 0, 0],
                "fn_missing_prediction": [0, 0, 0],
                "tn": [0, 0, 1],
                "tp_examples": [[], [], []],
                "fp_misclassification_examples": [["uid2"], ["uid2"], []],
                "fn_misclassification_examples": [[], [], []],
                "fn_missing_prediction_examples": [[], [], []],
                "tn_examples": [[], [], ["uid2"]],
            },
            "parameters": {
                "score_thresholds": [0.05, 0.15, 0.95],
                "label": {"key": "k2", "value": "flour"},
            },
            "type": "DetailedCounts",
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics


def test_detailed_counts_zero_count_check(
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

    metrics = evaluator.compute_detailed_counts(
        score_thresholds=[0.05, 0.2, 0.95],
        n_samples=0,
    )

    # test DetailedCounts
    actual_metrics = [m.to_dict() for m in metrics]
    expected_metrics = [
        {
            "value": {
                "tp": [1, 0, 0],
                "fp_misclassification": [0, 0, 0],
                "fn_misclassification": [0, 1, 0],
                "fn_missing_prediction": [0, 0, 1],
                "tn": [0, 0, 0],
                "tp_examples": [[], [], []],
                "fp_misclassification_examples": [[], [], []],
                "fn_misclassification_examples": [[], [], []],
                "fn_missing_prediction_examples": [[], [], []],
                "tn_examples": [[], [], []],
            },
            "parameters": {
                "score_thresholds": [0.05, 0.2, 0.95],
                "label": {"key": "k", "value": "ant"},
            },
            "type": "DetailedCounts",
        },
        {
            "value": {
                "tp": [0, 0, 0],
                "fp_misclassification": [1, 1, 0],
                "fn_misclassification": [0, 0, 0],
                "fn_missing_prediction": [0, 0, 0],
                "tn": [0, 0, 1],
                "tp_examples": [[], [], []],
                "fp_misclassification_examples": [[], [], []],
                "fn_misclassification_examples": [[], [], []],
                "fn_missing_prediction_examples": [[], [], []],
                "tn_examples": [[], [], []],
            },
            "parameters": {
                "score_thresholds": [0.05, 0.2, 0.95],
                "label": {"key": "k", "value": "bee"},
            },
            "type": "DetailedCounts",
        },
        {
            "value": {
                "tp": [0, 0, 0],
                "fp_misclassification": [1, 1, 0],
                "fn_misclassification": [0, 0, 0],
                "fn_missing_prediction": [0, 0, 0],
                "tn": [0, 0, 1],
                "tp_examples": [[], [], []],
                "fp_misclassification_examples": [[], [], []],
                "fn_misclassification_examples": [[], [], []],
                "fn_missing_prediction_examples": [[], [], []],
                "tn_examples": [[], [], []],
            },
            "parameters": {
                "score_thresholds": [0.05, 0.2, 0.95],
                "label": {"key": "k", "value": "cat"},
            },
            "type": "DetailedCounts",
        },
    ]
    for m in actual_metrics:
        assert m in expected_metrics
    for m in expected_metrics:
        assert m in actual_metrics
