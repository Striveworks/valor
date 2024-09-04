import pytest
from valor_core import enums


@pytest.fixture
def evaluate_classification_with_label_maps_expected():
    cat_expected_metrics = [
        {"type": "Accuracy", "parameters": {"label_key": "k3"}, "value": 0.0},
        {"type": "ROCAUC", "parameters": {"label_key": "k3"}, "value": 1.0},
        {
            "type": "Precision",
            "value": 0.0,
            "label": {"key": "k3", "value": "v1"},
        },
        {
            "type": "Recall",
            "value": 0.0,
            "label": {"key": "k3", "value": "v1"},
        },
        {"type": "F1", "value": 0.0, "label": {"key": "k3", "value": "v1"}},
        {
            "type": "Precision",
            "value": 0.0,
            "label": {"key": "k3", "value": "v3"},
        },
        {
            "type": "Recall",
            "value": 0.0,
            "label": {"key": "k3", "value": "v3"},
        },
        {"type": "F1", "value": 0.0, "label": {"key": "k3", "value": "v3"}},
        {"type": "Accuracy", "parameters": {"label_key": "k5"}, "value": 0.0},
        {"type": "ROCAUC", "parameters": {"label_key": "k5"}, "value": 1.0},
        {
            "type": "Precision",
            "value": 0.0,
            "label": {"key": "k5", "value": "v5"},
        },
        {
            "type": "Recall",
            "value": 0.0,
            "label": {"key": "k5", "value": "v5"},
        },
        {"type": "F1", "value": 0.0, "label": {"key": "k5", "value": "v5"}},
        {
            "type": "Precision",
            "value": 0.0,
            "label": {"key": "k5", "value": "v1"},
        },
        {
            "type": "Recall",
            "value": 0.0,
            "label": {"key": "k5", "value": "v1"},
        },
        {"type": "F1", "value": 0.0, "label": {"key": "k5", "value": "v1"}},
        {
            "type": "Accuracy",
            "parameters": {"label_key": "special_class"},
            "value": 1.0,
        },
        {
            "type": "ROCAUC",
            "parameters": {"label_key": "special_class"},
            "value": 1.0,
        },
        {
            "type": "Precision",
            "value": 1.0,
            "label": {"key": "special_class", "value": "cat_type1"},
        },
        {
            "type": "Recall",
            "value": 1.0,
            "label": {"key": "special_class", "value": "cat_type1"},
        },
        {
            "type": "F1",
            "value": 1.0,
            "label": {"key": "special_class", "value": "cat_type1"},
        },
        {"type": "Accuracy", "parameters": {"label_key": "k4"}, "value": 0.5},
        {
            "type": "ROCAUC",
            "parameters": {
                "label_key": "k4",
            },
            "value": 1.0,
        },
        {
            "type": "Precision",
            "value": -1.0,
            "label": {"key": "k4", "value": "v5"},
        },
        {
            "type": "Recall",
            "value": -1.0,
            "label": {"key": "k4", "value": "v5"},
        },
        {"type": "F1", "value": -1.0, "label": {"key": "k4", "value": "v5"}},
        {
            "type": "Precision",
            "value": -1.0,
            "label": {"key": "k4", "value": "v1"},
        },
        {
            "type": "Recall",
            "value": -1.0,
            "label": {"key": "k4", "value": "v1"},
        },
        {"type": "F1", "value": -1.0, "label": {"key": "k4", "value": "v1"}},
        {
            "type": "Precision",
            "value": 1.0,
            "label": {"key": "k4", "value": "v4"},
        },
        {
            "type": "Recall",
            "value": 0.5,
            "label": {"key": "k4", "value": "v4"},
        },
        {
            "type": "F1",
            "value": 0.6666666666666666,
            "label": {"key": "k4", "value": "v4"},
        },
        {
            "type": "Precision",
            "value": 0.0,
            "label": {"key": "k4", "value": "v8"},
        },
        {
            "type": "Recall",
            "value": 0.0,
            "label": {"key": "k4", "value": "v8"},
        },
        {"type": "F1", "value": 0.0, "label": {"key": "k4", "value": "v8"}},
    ]

    cat_expected_cm = [
        {
            "label_key": "special_class",
            "entries": [
                {
                    "prediction": "cat_type1",
                    "groundtruth": "cat_type1",
                    "count": 3,
                }
            ],
        }
        # other label keys not included for testing purposes
    ]

    pr_expected_values = {
        # k3
        (0, "k3", "v1", "0.1", "fp"): 1,
        (0, "k3", "v1", "0.1", "tn"): 2,
        (0, "k3", "v3", "0.1", "fn"): 1,
        (0, "k3", "v3", "0.1", "tn"): 2,
        (0, "k3", "v3", "0.1", "accuracy"): 2 / 3,
        (0, "k3", "v3", "0.1", "precision"): -1,
        (0, "k3", "v3", "0.1", "recall"): 0,
        (0, "k3", "v3", "0.1", "f1_score"): -1,
        # k4
        (1, "k4", "v1", "0.1", "fp"): 1,
        (1, "k4", "v1", "0.1", "tn"): 2,
        (1, "k4", "v4", "0.1", "fn"): 1,
        (1, "k4", "v4", "0.1", "tn"): 1,
        (1, "k4", "v4", "0.1", "tp"): 1,
        (1, "k4", "v4", "0.9", "tp"): 0,
        (1, "k4", "v4", "0.9", "tn"): 1,
        (1, "k4", "v4", "0.9", "fn"): 2,
        (1, "k4", "v5", "0.1", "fp"): 1,
        (1, "k4", "v5", "0.1", "tn"): 2,
        (1, "k4", "v5", "0.3", "fp"): 0,
        (1, "k4", "v5", "0.3", "tn"): 3,
        (1, "k4", "v8", "0.1", "tn"): 2,
        (1, "k4", "v8", "0.6", "fp"): 0,
        (1, "k4", "v8", "0.6", "tn"): 3,
        # k5
        (2, "k5", "v1", "0.1", "fp"): 1,
        (2, "k5", "v1", "0.1", "tn"): 2,
        (2, "k5", "v5", "0.1", "fn"): 1,
        (
            2,
            "k5",
            "v5",
            "0.1",
            "tn",
        ): 2,
        (2, "k5", "v1", "0.1", "accuracy"): 2 / 3,
        (2, "k5", "v1", "0.1", "precision"): 0,
        (2, "k5", "v1", "0.1", "recall"): -1,
        (2, "k5", "v1", "0.1", "f1_score"): -1,
        # special_class
        (3, "special_class", "cat_type1", "0.1", "tp"): 3,
        (3, "special_class", "cat_type1", "0.1", "tn"): 0,
        (3, "special_class", "cat_type1", "0.95", "tp"): 3,
    }

    detailed_pr_expected_answers = {
        # k3
        (0, "v1", "0.1", "tp"): {"all": 0, "total": 0},
        (0, "v1", "0.1", "fp"): {
            "misclassifications": 1,
            "total": 1,
        },
        (0, "v1", "0.1", "tn"): {"all": 2, "total": 2},
        (0, "v1", "0.1", "fn"): {
            "no_predictions": 0,
            "misclassifications": 0,
            "total": 0,
        },
        # k4
        (1, "v1", "0.1", "tp"): {"all": 0, "total": 0},
        (1, "v1", "0.1", "fp"): {
            "misclassifications": 1,
            "total": 1,
        },
        (1, "v1", "0.1", "tn"): {"all": 2, "total": 2},
        (1, "v1", "0.1", "fn"): {
            "no_predictions": 0,
            "misclassifications": 0,
            "total": 0,
        },
        (1, "v4", "0.1", "fn"): {
            "no_predictions": 0,
            "misclassifications": 1,
            "total": 1,
        },
        (1, "v8", "0.1", "tn"): {"all": 2, "total": 2},
    }

    return (
        cat_expected_metrics,
        cat_expected_cm,
        pr_expected_values,
        detailed_pr_expected_answers,
    )


@pytest.fixture
def evaluate_image_clf_expected():
    expected_metrics = [
        {"type": "Accuracy", "parameters": {"label_key": "k4"}, "value": 0.5},
        {
            "type": "ROCAUC",
            "parameters": {"label_key": "k4"},
            "value": 1.0,
        },
        {
            "type": "Precision",
            "value": 1.0,  # no false predictions
            "label": {"key": "k4", "value": "v4"},
        },
        {
            "type": "Recall",
            "value": 0.5,  # img5 had the correct prediction, but not img6
            "label": {"key": "k4", "value": "v4"},
        },
        {
            "type": "F1",
            "value": 0.6666666666666666,
            "label": {"key": "k4", "value": "v4"},
        },
        {
            "type": "Precision",
            "value": 0.0,
            "label": {"key": "k4", "value": "v8"},
        },
        {
            "type": "Recall",
            "value": 0.0,
            "label": {"key": "k4", "value": "v8"},
        },
        {"type": "F1", "value": 0.0, "label": {"key": "k4", "value": "v8"}},
        {
            "type": "Precision",
            "value": -1.0,
            "label": {"key": "k4", "value": "v5"},
        },
        {
            "type": "Recall",
            "value": -1.0,
            "label": {"key": "k4", "value": "v5"},
        },
        {"type": "F1", "value": -1.0, "label": {"key": "k4", "value": "v5"}},
        {
            "type": "Precision",
            "value": -1.0,  # this value is -1 (not 0) because this label is never used anywhere; (k4, v8) has the higher score for img5, so it's chosen over (k4, v1)
            "label": {"key": "k4", "value": "v1"},
        },
        {
            "type": "Recall",
            "value": -1.0,
            "label": {"key": "k4", "value": "v1"},
        },
        {"type": "F1", "value": -1.0, "label": {"key": "k4", "value": "v1"}},
        {"type": "Accuracy", "parameters": {"label_key": "k5"}, "value": 0.0},
        {
            "type": "ROCAUC",
            "parameters": {"label_key": "k5"},
            "value": 1.0,
        },
        {
            "type": "Precision",
            "value": 0.0,
            "label": {"key": "k5", "value": "v1"},
        },
        {
            "type": "Recall",
            "value": 0.0,
            "label": {"key": "k5", "value": "v1"},
        },
        {"type": "F1", "value": 0.0, "label": {"key": "k5", "value": "v1"}},
        {
            "type": "Precision",
            "value": 0.0,
            "label": {"key": "k5", "value": "v5"},
        },
        {
            "type": "Recall",
            "value": 0.0,
            "label": {"key": "k5", "value": "v5"},
        },
        {"type": "F1", "value": 0.0, "label": {"key": "k5", "value": "v5"}},
        {"type": "Accuracy", "parameters": {"label_key": "k3"}, "value": 0.0},
        {"type": "ROCAUC", "parameters": {"label_key": "k3"}, "value": 1.0},
        {
            "type": "Precision",
            "value": 0.0,
            "label": {"key": "k3", "value": "v1"},
        },
        {
            "type": "Recall",
            "value": 0.0,
            "label": {"key": "k3", "value": "v1"},
        },
        {"type": "F1", "value": 0.0, "label": {"key": "k3", "value": "v1"}},
        {
            "type": "Precision",
            "value": 0.0,
            "label": {"key": "k3", "value": "v3"},
        },
        {
            "type": "Recall",
            "value": 0.0,
            "label": {"key": "k3", "value": "v3"},
        },
        {"type": "F1", "value": 0.0, "label": {"key": "k3", "value": "v3"}},
    ]

    expected_confusion_matrices = [
        {
            "label_key": "k5",
            "entries": [{"prediction": "v1", "groundtruth": "v5", "count": 1}],
        },
        {
            "label_key": "k4",
            "entries": [
                {"prediction": "v4", "groundtruth": "v4", "count": 1},
                {"prediction": "v8", "groundtruth": "v4", "count": 1},
            ],
        },
        {
            "label_key": "k3",
            "entries": [{"prediction": "v1", "groundtruth": "v3", "count": 1}],
        },
    ]

    return expected_metrics, expected_confusion_matrices


@pytest.fixture
def evaluate_tabular_clf_expected():

    expected_metrics = [
        {
            "type": "Accuracy",
            "parameters": {"label_key": "class"},
            "value": 0.5,
        },
        {
            "type": "ROCAUC",
            "parameters": {"label_key": "class"},
            "value": 0.7685185185185185,
        },
        {
            "type": "Precision",
            "value": 0.6666666666666666,
            "label": {"key": "class", "value": "1"},
        },
        {
            "type": "Recall",
            "value": 0.3333333333333333,
            "label": {"key": "class", "value": "1"},
        },
        {
            "type": "F1",
            "value": 0.4444444444444444,
            "label": {"key": "class", "value": "1"},
        },
        {
            "type": "Precision",
            "value": 0.0,
            "label": {"key": "class", "value": "2"},
        },
        {
            "type": "Recall",
            "value": 0.0,
            "label": {"key": "class", "value": "2"},
        },
        {"type": "F1", "value": 0.0, "label": {"key": "class", "value": "2"}},
        {
            "type": "Precision",
            "value": 0.5,
            "label": {"key": "class", "value": "0"},
        },
        {
            "type": "Recall",
            "value": 1.0,
            "label": {"key": "class", "value": "0"},
        },
        {
            "type": "F1",
            "value": 0.6666666666666666,
            "label": {"key": "class", "value": "0"},
        },
    ]

    expected_confusion_matrix = {
        "label_key": "class",
        "entries": [
            {"prediction": "0", "groundtruth": "0", "count": 3},
            {"prediction": "0", "groundtruth": "1", "count": 3},
            {"prediction": "1", "groundtruth": "1", "count": 2},
            {"prediction": "1", "groundtruth": "2", "count": 1},
            {"prediction": "2", "groundtruth": "1", "count": 1},
        ],
    }

    return expected_metrics, expected_confusion_matrix


@pytest.fixture
def evaluate_classification_model_with_no_predictions_expected():

    expected_metrics = [
        {"type": "Accuracy", "parameters": {"label_key": "k5"}, "value": 0.0},
        {"type": "ROCAUC", "parameters": {"label_key": "k5"}, "value": 0.0},
        {
            "type": "Precision",
            "value": 0.0,
            "label": {"key": "k5", "value": "v5"},
        },
        {
            "type": "Recall",
            "value": 0.0,
            "label": {"key": "k5", "value": "v5"},
        },
        {"type": "F1", "value": 0.0, "label": {"key": "k5", "value": "v5"}},
        {"type": "Accuracy", "parameters": {"label_key": "k4"}, "value": 0.0},
        {"type": "ROCAUC", "parameters": {"label_key": "k4"}, "value": 0.0},
        {
            "type": "Precision",
            "value": 0.0,
            "label": {"key": "k4", "value": "v4"},
        },
        {
            "type": "Recall",
            "value": 0.0,
            "label": {"key": "k4", "value": "v4"},
        },
        {"type": "F1", "value": 0.0, "label": {"key": "k4", "value": "v4"}},
        {"type": "Accuracy", "parameters": {"label_key": "k3"}, "value": 0.0},
        {"type": "ROCAUC", "parameters": {"label_key": "k3"}, "value": 0.0},
        {
            "type": "Precision",
            "value": 0.0,
            "label": {"key": "k3", "value": "v3"},
        },
        {
            "type": "Recall",
            "value": 0.0,
            "label": {"key": "k3", "value": "v3"},
        },
        {"type": "F1", "value": 0.0, "label": {"key": "k3", "value": "v3"}},
    ]

    return expected_metrics


@pytest.fixture
def compute_confusion_matrix_at_label_key_using_label_map_expected():

    expected_entries = [
        {
            "label_key": "class",
            "entries": [
                {"prediction": "avian", "groundtruth": "avian", "count": 1},
                {"prediction": "mammal", "groundtruth": "avian", "count": 2},
                {"prediction": "mammal", "groundtruth": "mammal", "count": 3},
            ],
        },
        {
            "label_key": "color",
            "entries": [
                {"prediction": "blue", "groundtruth": "white", "count": 1},
                {"prediction": "red", "groundtruth": "black", "count": 1},
                {"prediction": "red", "groundtruth": "red", "count": 2},
                {"prediction": "white", "groundtruth": "blue", "count": 1},
                {"prediction": "white", "groundtruth": "white", "count": 1},
            ],
        },
    ]

    return expected_entries


@pytest.fixture
def rocauc_with_label_map_expected():

    expected_metrics = [
        {
            "parameters": {"label_key": "animal"},
            "value": 0.8009259259259259,
            "type": "ROCAUC",
        },
        {
            "parameters": {"label_key": "color"},
            "value": 0.43125,
            "type": "ROCAUC",
        },
    ]

    return expected_metrics


@pytest.fixture
def compute_classification_expected():

    expected_metrics = [
        {
            "label": {"key": "animal", "value": "bird"},
            "value": 1.0,
            "type": "Precision",
        },
        {
            "label": {"key": "animal", "value": "bird"},
            "value": 0.3333333333333333,
            "type": "Recall",
        },
        {
            "label": {"key": "animal", "value": "bird"},
            "value": 0.5,
            "type": "F1",
        },
        {
            "label": {"key": "animal", "value": "cat"},
            "value": 0.25,
            "type": "Precision",
        },
        {
            "label": {"key": "animal", "value": "cat"},
            "value": 1.0,
            "type": "Recall",
        },
        {
            "label": {"key": "animal", "value": "cat"},
            "value": 0.4,
            "type": "F1",
        },
        {
            "label": {"key": "animal", "value": "dog"},
            "value": 0.0,
            "type": "Precision",
        },
        {
            "label": {"key": "animal", "value": "dog"},
            "value": 0.0,
            "type": "Recall",
        },
        {
            "label": {"key": "animal", "value": "dog"},
            "value": 0.0,
            "type": "F1",
        },
        {
            "label": {"key": "color", "value": "blue"},
            "value": 0.0,
            "type": "Precision",
        },
        {
            "label": {"key": "color", "value": "blue"},
            "value": 0.0,
            "type": "Recall",
        },
        {
            "label": {"key": "color", "value": "blue"},
            "value": 0.0,
            "type": "F1",
        },
        {
            "label": {"key": "color", "value": "red"},
            "value": 0.6666666666666666,
            "type": "Precision",
        },
        {
            "label": {"key": "color", "value": "red"},
            "value": 1.0,
            "type": "Recall",
        },
        {
            "label": {"key": "color", "value": "red"},
            "value": 0.8,
            "type": "F1",
        },
        {
            "label": {"key": "color", "value": "white"},
            "value": 0.5,
            "type": "Precision",
        },
        {
            "label": {"key": "color", "value": "white"},
            "value": 0.5,
            "type": "Recall",
        },
        {
            "label": {"key": "color", "value": "white"},
            "value": 0.5,
            "type": "F1",
        },
        {
            "label": {"key": "color", "value": "black"},
            "value": 0.0,
            "type": "Precision",
        },
        {
            "label": {"key": "color", "value": "black"},
            "value": 0.0,
            "type": "Recall",
        },
        {
            "label": {"key": "color", "value": "black"},
            "value": 0.0,
            "type": "F1",
        },
        {
            "parameters": {"label_key": "animal"},
            "value": 0.3333333333333333,
            "type": "Accuracy",
        },
        {
            "parameters": {"label_key": "color"},
            "value": 0.5,
            "type": "Accuracy",
        },
        {
            "parameters": {"label_key": "animal"},
            "value": 0.8009259259259259,
            "type": "ROCAUC",
        },
        {
            "parameters": {"label_key": "color"},
            "value": 0.43125,
            "type": "ROCAUC",
        },
    ]
    expected_pr_curves = {
        # bird
        ("bird", 0.05, "tp"): 3,
        ("bird", 0.05, "fp"): 1,
        ("bird", 0.05, "tn"): 2,
        ("bird", 0.05, "fn"): 0,
        ("bird", 0.3, "tp"): 1,
        ("bird", 0.3, "fn"): 2,
        ("bird", 0.3, "fp"): 0,
        ("bird", 0.3, "tn"): 3,
        ("bird", 0.65, "fn"): 3,
        ("bird", 0.65, "tn"): 3,
        ("bird", 0.65, "tp"): 0,
        ("bird", 0.65, "fp"): 0,
        # dog
        ("dog", 0.05, "tp"): 2,
        ("dog", 0.05, "fp"): 3,
        ("dog", 0.05, "tn"): 1,
        ("dog", 0.05, "fn"): 0,
        ("dog", 0.45, "fn"): 2,
        ("dog", 0.45, "fp"): 1,
        ("dog", 0.45, "tn"): 3,
        ("dog", 0.45, "tp"): 0,
        ("dog", 0.8, "fn"): 2,
        ("dog", 0.8, "fp"): 0,
        ("dog", 0.8, "tn"): 4,
        ("dog", 0.8, "tp"): 0,
        # cat
        ("cat", 0.05, "tp"): 1,
        ("cat", 0.05, "tn"): 0,
        ("cat", 0.05, "fp"): 5,
        ("cat", 0.05, "fn"): 0,
        ("cat", 0.95, "tp"): 1,
        ("cat", 0.95, "fp"): 0,
        ("cat", 0.95, "tn"): 5,
        ("cat", 0.95, "fn"): 0,
    }
    expected_detailed_pr_curves = {
        # bird
        ("bird", 0.05, "tp"): {"all": 3, "total": 3},
        ("bird", 0.05, "fp"): {
            "misclassifications": 1,
            "total": 1,
        },
        ("bird", 0.05, "tn"): {"all": 2, "total": 2},
        ("bird", 0.05, "fn"): {
            "no_predictions": 0,
            "misclassifications": 0,
            "total": 0,
        },
        # dog
        ("dog", 0.05, "tp"): {"all": 2, "total": 2},
        ("dog", 0.05, "fp"): {
            "misclassifications": 3,
            "total": 3,
        },
        ("dog", 0.05, "tn"): {"all": 1, "total": 1},
        ("dog", 0.8, "fn"): {
            "no_predictions": 1,
            "misclassifications": 1,
            "total": 2,
        },
        # cat
        ("cat", 0.05, "tp"): {"all": 1, "total": 1},
        ("cat", 0.05, "fp"): {
            "misclassifications": 5,
            "total": 5,
        },
        ("cat", 0.05, "tn"): {"all": 0, "total": 0},
        ("cat", 0.8, "fn"): {
            "no_predictions": 0,
            "misclassifications": 0,
            "total": 0,
        },
    }
    expected_cm = [
        {
            "label_key": "animal",
            "entries": [
                {"prediction": "bird", "groundtruth": "bird", "count": 1},
                {"prediction": "cat", "groundtruth": "bird", "count": 1},
                {"prediction": "cat", "groundtruth": "cat", "count": 1},
                {"prediction": "cat", "groundtruth": "dog", "count": 2},
                {"prediction": "dog", "groundtruth": "bird", "count": 1},
            ],
        },
        {
            "label_key": "color",
            "entries": [
                {"prediction": "blue", "groundtruth": "white", "count": 1},
                {"prediction": "red", "groundtruth": "black", "count": 1},
                {"prediction": "red", "groundtruth": "red", "count": 2},
                {"prediction": "white", "groundtruth": "blue", "count": 1},
                {"prediction": "white", "groundtruth": "white", "count": 1},
            ],
        },
    ]

    return (
        expected_metrics,
        expected_cm,
        expected_pr_curves,
        expected_detailed_pr_curves,
    )


@pytest.fixture
def test_pr_curves_multiple_predictions_per_groundtruth_expected():
    expected_outputs = {
        "bee": {
            0.05: {
                "tp": 2.0,
                "fp": 3.0,
                "fn": 0.0,
                "tn": 0.0,
            },
            0.55: {
                "tp": 1.0,
                "fp": 0.0,
                "fn": 1.0,
                "tn": 3.0,
            },
            0.95: {
                "tp": 0.0,
                "fp": 0.0,
                "fn": 2.0,
                "tn": 3.0,
            },
        },
        "cat": {
            0.05: {
                "tp": 2.0,
                "fp": 3.0,
                "fn": 0.0,
                "tn": 0.0,
            },
            0.4: {
                "tp": 1.0,
                "fp": 1.0,
                "fn": 1.0,
                "tn": 2.0,
            },
            0.95: {
                "tp": 0.0,
                "fp": 0.0,
                "fn": 2.0,
                "tn": 3.0,
            },
        },
        "dog": {
            0.05: {
                "tp": 1.0,
                "fp": 4.0,
                "fn": 0.0,
                "tn": 0.0,
            },
            0.55: {
                "tp": 1.0,
                "fp": 0.0,
                "fn": 0.0,
                "tn": 4.0,
            },
            0.95: {
                "tp": 0.0,
                "fp": 0.0,
                "fn": 1.0,
                "tn": 4.0,
            },
        },
    }

    return expected_outputs


@pytest.fixture
def evaluate_detection_expected():

    expected_metrics = [
        {
            "label": {"key": "k2", "value": "v2"},
            "parameters": {"iou": 0.1},
            "value": 0.0,
            "type": "AP",
        },
        {
            "label": {"key": "k2", "value": "v2"},
            "parameters": {"iou": 0.6},
            "value": 0.0,
            "type": "AP",
        },
        {
            "label": {"key": "k1", "value": "v1"},
            "parameters": {"iou": 0.1},
            "value": 0.504950495049505,
            "type": "AP",
        },
        {
            "label": {"key": "k1", "value": "v1"},
            "parameters": {"iou": 0.6},
            "value": 0.504950495049505,
            "type": "AP",
        },
        {
            "parameters": {"label_key": "k1", "iou": 0.1},
            "value": 0.504950495049505,
            "type": "mAP",
        },
        {
            "parameters": {"label_key": "k2", "iou": 0.1},
            "value": 0.0,
            "type": "mAP",
        },
        {
            "parameters": {"label_key": "k1", "iou": 0.6},
            "value": 0.504950495049505,
            "type": "mAP",
        },
        {
            "parameters": {"label_key": "k2", "iou": 0.6},
            "value": 0.0,
            "type": "mAP",
        },
        {
            "label": {"key": "k2", "value": "v2"},
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.0,
            "type": "APAveragedOverIOUs",
        },
        {
            "label": {"key": "k1", "value": "v1"},
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.504950495049505,
            "type": "APAveragedOverIOUs",
        },
        {
            "parameters": {"label_key": "k1", "ious": [0.1, 0.6]},
            "value": 0.504950495049505,
            "type": "mAPAveragedOverIOUs",
        },
        {
            "parameters": {"label_key": "k2", "ious": [0.1, 0.6]},
            "value": 0.0,
            "type": "mAPAveragedOverIOUs",
        },
        {
            "label": {"key": "k2", "value": "v2"},
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.0,
            "type": "AR",
        },
        {
            "label": {"key": "k1", "value": "v1"},
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.5,
            "type": "AR",
        },
        {
            "parameters": {"label_key": "k1", "ious": [0.1, 0.6]},
            "value": 0.5,
            "type": "mAR",
        },
        {
            "parameters": {"label_key": "k2", "ious": [0.1, 0.6]},
            "value": 0.0,
            "type": "mAR",
        },
    ]

    expected_metadata = {
        "parameters": {
            "label_map": {},
            "metrics_to_return": [
                enums.MetricType.AP,
                enums.MetricType.AR,
                enums.MetricType.mAP,
                enums.MetricType.APAveragedOverIOUs,
                enums.MetricType.mAR,
                enums.MetricType.mAPAveragedOverIOUs,
            ],
            "iou_thresholds_to_compute": [0.1, 0.6],
            "iou_thresholds_to_return": [0.1, 0.6],
            "recall_score_threshold": 0.0,
            "pr_curve_iou_threshold": 0.5,
            "pr_curve_max_examples": 1,
            "convert_annotations_to_type": None,
        },
        "confusion_matrices": [],
        "ignored_pred_labels": [],
        "missing_pred_labels": [],
    }

    return expected_metrics, expected_metadata


@pytest.fixture
def evaluate_detection_with_label_maps_expected():

    baseline_expected_metrics = [
        {
            "type": "AP",
            "parameters": {"iou": 0.1},
            "value": 0.504950495049505,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.6},
            "value": 0.504950495049505,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.1},
            "value": 0.0,
            "label": {"key": "k2", "value": "v2"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.6},
            "value": 0.0,
            "label": {"key": "k2", "value": "v2"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.1},
            "value": 0.0,
            "label": {"key": "class_name", "value": "maine coon cat"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.6},
            "value": 0.0,
            "label": {"key": "class_name", "value": "maine coon cat"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.1},
            "value": 0.0,
            "label": {"key": "class", "value": "british shorthair"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.6},
            "value": 0.0,
            "label": {"key": "class", "value": "british shorthair"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.1},
            "value": 0.0,
            "label": {"key": "class", "value": "siamese cat"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.6},
            "value": 0.0,
            "label": {"key": "class", "value": "siamese cat"},
        },
        {
            "type": "AR",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.5,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "AR",
            "parameters": {"ious": [0.1, 0.6]},
            "value": -1.0,
            "label": {"key": "class_name", "value": "cat"},
        },
        {
            "type": "AR",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.0,
            "label": {"key": "k2", "value": "v2"},
        },
        {
            "type": "AR",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.0,
            "label": {"key": "class_name", "value": "maine coon cat"},
        },
        {
            "type": "AR",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.0,
            "label": {"key": "class", "value": "british shorthair"},
        },
        {
            "type": "AR",
            "parameters": {"ious": [0.1, 0.6]},
            "value": -1.0,
            "label": {"key": "class", "value": "cat"},
        },
        {
            "type": "AR",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.0,
            "label": {"key": "class", "value": "siamese cat"},
        },
        {
            "type": "mAR",
            "parameters": {"ious": [0.1, 0.6], "label_key": "k1"},
            "value": 0.5,
        },
        {
            "type": "mAR",
            "parameters": {"ious": [0.1, 0.6], "label_key": "k2"},
            "value": 0.0,
        },
        {
            "type": "mAR",
            "parameters": {"ious": [0.1, 0.6], "label_key": "class_name"},
            "value": 0.0,
        },
        {
            "type": "mAR",
            "parameters": {"ious": [0.1, 0.6], "label_key": "class"},
            "value": 0.0,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.1, "label_key": "class"},
            "value": 0.0,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.6, "label_key": "class"},
            "value": 0.0,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.1, "label_key": "class_name"},
            "value": 0.0,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.6, "label_key": "class_name"},
            "value": 0.0,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.1, "label_key": "k1"},
            "value": 0.504950495049505,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.6, "label_key": "k1"},
            "value": 0.504950495049505,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.1, "label_key": "k2"},
            "value": 0.0,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.6, "label_key": "k2"},
            "value": 0.0,
        },
        {
            "type": "APAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.504950495049505,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "APAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.0,
            "label": {"key": "k2", "value": "v2"},
        },
        {
            "type": "APAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.0,
            "label": {"key": "class_name", "value": "maine coon cat"},
        },
        {
            "type": "APAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.0,
            "label": {"key": "class", "value": "british shorthair"},
        },
        {
            "type": "APAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.0,
            "label": {"key": "class", "value": "siamese cat"},
        },
        {
            "type": "mAPAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6], "label_key": "class"},
            "value": 0.0,
        },
        {
            "type": "mAPAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6], "label_key": "class_name"},
            "value": 0.0,
        },
        {
            "type": "mAPAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6], "label_key": "k1"},
            "value": 0.504950495049505,
        },
        {
            "type": "mAPAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6], "label_key": "k2"},
            "value": 0.0,
        },
    ]

    baseline_pr_expected_answers = {
        # class
        (
            0,
            "class",
            "cat",
            "0.1",
            "fp",
        ): 1,
        (0, "class", "cat", "0.4", "fp"): 0,
        (0, "class", "siamese cat", "0.1", "fn"): 1,
        (0, "class", "british shorthair", "0.1", "fn"): 1,
        # class_name
        (1, "class_name", "cat", "0.1", "fp"): 1,
        (1, "class_name", "maine coon cat", "0.1", "fn"): 1,
        # k1
        (2, "k1", "v1", "0.1", "fn"): 1,
        (2, "k1", "v1", "0.1", "tp"): 1,
        (2, "k1", "v1", "0.4", "fn"): 2,
        # k2
        (3, "k2", "v2", "0.1", "fn"): 1,
        (3, "k2", "v2", "0.1", "fp"): 1,
    }

    baseline_detailed_pr_expected_answers = {
        # class
        (0, "cat", "0.1", "fp"): {
            "hallucinations": 1,
            "misclassifications": 0,
            "total": 1,
        },
        (0, "cat", "0.4", "fp"): {
            "hallucinations": 0,
            "misclassifications": 0,
            "total": 0,
        },
        (0, "british shorthair", "0.1", "fn"): {
            "no_predictions": 1,
            "misclassifications": 0,
            "total": 1,
        },
        # class_name
        (1, "cat", "0.4", "fp"): {
            "hallucinations": 1,
            "misclassifications": 0,
            "total": 1,
        },
        (1, "maine coon cat", "0.1", "fn"): {
            "no_predictions": 1,
            "misclassifications": 0,
            "total": 1,
        },
        # k1
        (2, "v1", "0.1", "fn"): {
            "no_predictions": 1,
            "misclassifications": 0,
            "total": 1,
        },
        (2, "v1", "0.4", "fn"): {
            "no_predictions": 2,
            "misclassifications": 0,
            "total": 2,
        },
        (2, "v1", "0.1", "tp"): {"all": 1, "total": 1},
        # k2
        (3, "v2", "0.1", "fn"): {
            "no_predictions": 1,
            "misclassifications": 0,
            "total": 1,
        },
        (3, "v2", "0.1", "fp"): {
            "hallucinations": 1,
            "misclassifications": 0,
            "total": 1,
        },
    }

    cat_expected_metrics = [
        {
            "type": "AP",
            "parameters": {"iou": 0.1},
            "value": 0.33663366336633666,
            "label": {"key": "class", "value": "cat"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.1},
            "value": 0.504950495049505,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.1},
            "value": 0.0,
            "label": {"key": "k2", "value": "v2"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.6},
            "value": 0.33663366336633666,
            "label": {"key": "class", "value": "cat"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.6},
            "value": 0.504950495049505,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.6},
            "value": 0.0,
            "label": {"key": "k2", "value": "v2"},
        },
        {
            "type": "AR",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.5,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "AR",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.3333333333333333,
            "label": {"key": "class", "value": "cat"},
        },
        {
            "type": "AR",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.0,
            "label": {"key": "k2", "value": "v2"},
        },
        {
            "type": "AR",
            "parameters": {"ious": [0.1, 0.6]},
            "value": -1.0,
            "label": {"key": "class_name", "value": "cat"},
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.1, "label_key": "class"},
            "value": 0.33663366336633666,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.1, "label_key": "k1"},
            "value": 0.504950495049505,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.1, "label_key": "k2"},
            "value": 0.0,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.6, "label_key": "class"},
            "value": 0.33663366336633666,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.6, "label_key": "k1"},
            "value": 0.504950495049505,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.6, "label_key": "k2"},
            "value": 0.0,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.1, "label_key": "class"},
            "value": 0.33663366336633666,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.1, "label_key": "k1"},
            "value": 0.504950495049505,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.1, "label_key": "k2"},
            "value": 0.0,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.6, "label_key": "class"},
            "value": 0.33663366336633666,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.6, "label_key": "k1"},
            "value": 0.504950495049505,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.6, "label_key": "k2"},
            "value": 0.0,
        },
        {
            "type": "mAR",
            "parameters": {"ious": [0.1, 0.6], "label_key": "class"},
            "value": 0.3333333333333333,
        },
        {
            "type": "mAR",
            "parameters": {"ious": [0.1, 0.6], "label_key": "k1"},
            "value": 0.5,
        },
        {
            "type": "mAR",
            "parameters": {"ious": [0.1, 0.6], "label_key": "class_name"},
            "value": -1.0,
        },
        {
            "type": "mAR",
            "parameters": {"ious": [0.1, 0.6], "label_key": "k2"},
            "value": 0.0,
        },
        {
            "type": "APAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.33663366336633666,
            "label": {"key": "class", "value": "cat"},
        },
        {
            "type": "APAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.504950495049505,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "APAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.0,
            "label": {"key": "k2", "value": "v2"},
        },
        {
            "type": "mAPAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6], "label_key": "k1"},
            "value": 0.504950495049505,
        },
        {
            "type": "mAPAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6], "label_key": "class"},
            "value": 0.33663366336633666,
        },
        {
            "type": "mAPAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6], "label_key": "k2"},
            "value": 0.0,
        },
    ]

    foo_expected_metrics = [
        {
            "type": "AP",
            "parameters": {"iou": 0.1},
            "value": 0.6633663366336634,
            "label": {"key": "foo", "value": "bar"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.1},
            "value": 0.0,
            "label": {"key": "k2", "value": "v2"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.1},
            "value": 0.504950495049505,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "AR",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.6666666666666666,
            "label": {"key": "foo", "value": "bar"},
        },
        {
            "type": "AR",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.5,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.6, "label_key": "foo"},
            "value": 0.6633663366336634,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.6, "label_key": "k2"},
            "value": 0.0,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.6, "label_key": "k1"},
            "value": 0.504950495049505,
        },
        {
            "type": "mAR",
            "parameters": {"ious": [0.1, 0.6], "label_key": "k2"},
            "value": 0.0,
        },
        {
            "type": "APAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.6633663366336634,
            "label": {"key": "foo", "value": "bar"},
        },
        {
            "type": "APAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.504950495049505,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "mAPAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6], "label_key": "k2"},
            "value": 0.0,
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.6},
            "value": 0.6633663366336634,
            "label": {"key": "foo", "value": "bar"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.6},
            "value": 0.0,
            "label": {"key": "k2", "value": "v2"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.6},
            "value": 0.504950495049505,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "AR",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.0,
            "label": {"key": "k2", "value": "v2"},
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.1, "label_key": "foo"},
            "value": 0.6633663366336634,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.1, "label_key": "k2"},
            "value": 0.0,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.1, "label_key": "k1"},
            "value": 0.504950495049505,
        },
        {
            "type": "mAR",
            "parameters": {"ious": [0.1, 0.6], "label_key": "foo"},
            "value": 0.6666666666666666,
        },
        {
            "type": "mAR",
            "parameters": {"ious": [0.1, 0.6], "label_key": "k1"},
            "value": 0.5,
        },
        {
            "type": "APAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.0,
            "label": {"key": "k2", "value": "v2"},
        },
        {
            "type": "mAPAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6], "label_key": "foo"},
            "value": 0.6633663366336634,
        },
        {
            "type": "mAPAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6], "label_key": "k1"},
            "value": 0.504950495049505,
        },
    ]

    foo_expected_metrics_with_higher_score_threshold = [
        {
            "type": "AP",
            "parameters": {"iou": 0.1},
            "value": 0.6633663366336634,
            "label": {"key": "foo", "value": "bar"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.1},
            "value": 0.504950495049505,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.1},
            "value": 0.0,
            "label": {"key": "k2", "value": "v2"},
        },
        {
            "type": "AR",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.3333333333333333,  # two missed groundtruth on the first image, and 1 hit for the second image
            "label": {"key": "foo", "value": "bar"},
        },
        {
            "type": "AR",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.0,
            "label": {"key": "k2", "value": "v2"},
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.6, "label_key": "foo"},
            "value": 0.6633663366336634,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.6, "label_key": "k1"},
            "value": 0.504950495049505,
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.6},
            "value": 0.6633663366336634,
            "label": {"key": "foo", "value": "bar"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.6},
            "value": 0.504950495049505,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.6},
            "value": 0.0,
            "label": {"key": "k2", "value": "v2"},
        },
        {
            "type": "AR",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.0,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.1, "label_key": "foo"},
            "value": 0.6633663366336634,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.1, "label_key": "k1"},
            "value": 0.504950495049505,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.1, "label_key": "k2"},
            "value": 0.0,
        },
        {
            "type": "mAP",
            "parameters": {"iou": 0.6, "label_key": "k2"},
            "value": 0.0,
        },
        {
            "type": "mAR",
            "parameters": {"ious": [0.1, 0.6], "label_key": "foo"},
            "value": 0.3333333333333333,
        },
        {
            "type": "mAR",
            "parameters": {"ious": [0.1, 0.6], "label_key": "k1"},
            "value": 0.0,
        },
        {
            "type": "APAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.6633663366336634,
            "label": {"key": "foo", "value": "bar"},
        },
        {
            "type": "APAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.0,
            "label": {"key": "k2", "value": "v2"},
        },
        {
            "type": "mAPAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6], "label_key": "k1"},
            "value": 0.504950495049505,
        },
        {
            "type": "mAR",
            "parameters": {"ious": [0.1, 0.6], "label_key": "k2"},
            "value": 0.0,
        },
        {
            "type": "APAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6]},
            "value": 0.504950495049505,
            "label": {"key": "k1", "value": "v1"},
        },
        {
            "type": "mAPAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6], "label_key": "foo"},
            "value": 0.6633663366336634,
        },
        {
            "type": "mAPAveragedOverIOUs",
            "parameters": {"ious": [0.1, 0.6], "label_key": "k2"},
            "value": 0.0,
        },
    ]

    foo_pr_expected_answers = {
        # foo
        (0, "foo", "bar", "0.1", "fn"): 1,  # missed rect3
        (0, "foo", "bar", "0.1", "tp"): 2,
        (0, "foo", "bar", "0.4", "fn"): 2,
        (0, "foo", "bar", "0.4", "tp"): 1,
        # k1
        (1, "k1", "v1", "0.1", "fn"): 1,
        (1, "k1", "v1", "0.1", "tp"): 1,
        (1, "k1", "v1", "0.4", "fn"): 2,
        # k2
        (2, "k2", "v2", "0.1", "fn"): 1,
        (2, "k2", "v2", "0.1", "fp"): 1,
    }

    return (
        baseline_expected_metrics,
        baseline_pr_expected_answers,
        baseline_detailed_pr_expected_answers,
        cat_expected_metrics,
        foo_expected_metrics,
        foo_pr_expected_answers,
        foo_expected_metrics_with_higher_score_threshold,
    )


@pytest.fixture
def evaluate_detection_false_negatives_AP_of_1():
    return {
        "type": "AP",
        "parameters": {"iou": 0.5},
        "value": 1.0,
        "label": {"key": "key", "value": "value"},
    }


@pytest.fixture
def evaluate_detection_false_negatives_AP_of_point_5():
    return {
        "type": "AP",
        "parameters": {"iou": 0.5},
        "value": 0.5,
        "label": {"key": "key", "value": "value"},
    }


@pytest.fixture
def evaluate_detection_false_negatives_AP_of_0():
    return {
        "type": "AP",
        "parameters": {"iou": 0.5},
        "value": 0,
        "label": {"key": "key", "value": "other value"},
    }


@pytest.fixture
def detailed_precision_recall_curve_outputs():
    expected_outputs = {
        ("v1", "0.3", "tp", "total"): 1,
        ("v1", "0.55", "tp", "total"): 0,
        ("v1", "0.55", "fn", "total"): 1,
        ("v1", "0.55", "fn", "observations", "no_predictions", "count"): 1,
        ("v1", "0.05", "fn", "total"): 0,
        ("v1", "0.05", "fp", "total"): 0,
        (
            "missed_detection",
            "0.05",
            "fn",
            "observations",
            "no_predictions",
            "count",
        ): 1,
        (
            "missed_detection",
            "0.95",
            "fn",
            "observations",
            "no_predictions",
            "count",
        ): 1,
        ("missed_detection", "0.05", "tp", "total"): 0,
        ("missed_detection", "0.05", "fp", "total"): 0,
        ("v2", "0.3", "fn", "observations", "no_predictions", "count"): 1,
        ("v2", "0.35", "fn", "observations", "no_predictions", "count"): 1,
        ("v2", "0.05", "tp", "total"): 0,
        ("v2", "0.05", "fp", "total"): 0,
        ("not_v2", "0.05", "fp", "observations", "hallucinations", "count"): 1,
        (
            "not_v2",
            "0.05",
            "fp",
            "observations",
            "misclassifications",
            "count",
        ): 0,
        ("not_v2", "0.05", "tp", "total"): 0,
        ("not_v2", "0.05", "fn", "total"): 0,
        (
            "hallucination",
            "0.05",
            "fp",
            "observations",
            "hallucinations",
            "count",
        ): 1,
        (
            "hallucination",
            "0.35",
            "fp",
            "observations",
            "hallucinations",
            "count",
        ): 0,
        ("hallucination", "0.05", "tp", "total"): 0,
        ("hallucination", "0.05", "fn", "total"): 0,
        ("low_iou", "0.3", "fn", "observations", "no_predictions", "count"): 1,
        (
            "low_iou",
            "0.95",
            "fn",
            "observations",
            "no_predictions",
            "count",
        ): 1,
        ("low_iou", "0.3", "fp", "observations", "hallucinations", "count"): 1,
        (
            "low_iou",
            "0.55",
            "fp",
            "observations",
            "hallucinations",
            "count",
        ): 0,
    }

    lower_threshold_expected_outputs = {
        ("v2", "0.3", "fn", "observations", "misclassifications", "count"): 1,
        ("v2", "0.3", "fn", "observations", "no_predictions", "count"): 0,
        ("v2", "0.35", "fn", "observations", "misclassifications", "count"): 0,
        ("v2", "0.35", "fn", "observations", "no_predictions", "count"): 1,
        ("v2", "0.05", "tp", "total"): 0,
        ("v2", "0.05", "fp", "total"): 0,
        ("not_v2", "0.05", "fp", "observations", "hallucinations", "count"): 0,
        (
            "not_v2",
            "0.05",
            "fp",
            "observations",
            "misclassifications",
            "count",
        ): 1,
        ("not_v2", "0.05", "tp", "total"): 0,
        ("not_v2", "0.05", "fn", "total"): 0,
    }

    return expected_outputs, lower_threshold_expected_outputs


@pytest.fixture
def evaluate_detection_model_with_no_predictions_output():

    expected_metrics = [
        {
            "label": {
                "key": "k2",
                "value": "v2",
            },
            "parameters": {
                "iou": 0.5,
            },
            "type": "AP",
            "value": 0.0,
        },
        {
            "label": {
                "key": "k2",
                "value": "v2",
            },
            "parameters": {
                "iou": 0.75,
            },
            "type": "AP",
            "value": 0.0,
        },
        {
            "label": {
                "key": "k1",
                "value": "v1",
            },
            "parameters": {
                "iou": 0.5,
            },
            "type": "AP",
            "value": 0.0,
        },
        {
            "label": {
                "key": "k1",
                "value": "v1",
            },
            "parameters": {
                "iou": 0.75,
            },
            "type": "AP",
            "value": 0.0,
        },
        {
            "label": {
                "key": "k2",
                "value": "v2",
            },
            "parameters": {
                "ious": [
                    0.5,
                    0.55,
                    0.6,
                    0.65,
                    0.7,
                    0.75,
                    0.8,
                    0.85,
                    0.9,
                    0.95,
                ],
            },
            "type": "AR",
            "value": 0.0,
        },
        {
            "label": {
                "key": "k1",
                "value": "v1",
            },
            "parameters": {
                "ious": [
                    0.5,
                    0.55,
                    0.6,
                    0.65,
                    0.7,
                    0.75,
                    0.8,
                    0.85,
                    0.9,
                    0.95,
                ],
            },
            "type": "AR",
            "value": 0.0,
        },
        {
            "parameters": {
                "iou": 0.5,
                "label_key": "k2",
            },
            "type": "mAP",
            "value": 0.0,
        },
        {
            "parameters": {
                "iou": 0.75,
                "label_key": "k2",
            },
            "type": "mAP",
            "value": 0.0,
        },
        {
            "parameters": {
                "iou": 0.5,
                "label_key": "k1",
            },
            "type": "mAP",
            "value": 0.0,
        },
        {
            "parameters": {
                "iou": 0.75,
                "label_key": "k1",
            },
            "type": "mAP",
            "value": 0.0,
        },
        {
            "parameters": {
                "ious": [
                    0.5,
                    0.55,
                    0.6,
                    0.65,
                    0.7,
                    0.75,
                    0.8,
                    0.85,
                    0.9,
                    0.95,
                ],
                "label_key": "k2",
            },
            "type": "mAR",
            "value": 0.0,
        },
        {
            "parameters": {
                "ious": [
                    0.5,
                    0.55,
                    0.6,
                    0.65,
                    0.7,
                    0.75,
                    0.8,
                    0.85,
                    0.9,
                    0.95,
                ],
                "label_key": "k1",
            },
            "type": "mAR",
            "value": 0.0,
        },
        {
            "label": {
                "key": "k2",
                "value": "v2",
            },
            "parameters": {
                "ious": [
                    0.5,
                    0.55,
                    0.6,
                    0.65,
                    0.7,
                    0.75,
                    0.8,
                    0.85,
                    0.9,
                    0.95,
                ],
            },
            "type": "APAveragedOverIOUs",
            "value": 0.0,
        },
        {
            "label": {
                "key": "k1",
                "value": "v1",
            },
            "parameters": {
                "ious": [
                    0.5,
                    0.55,
                    0.6,
                    0.65,
                    0.7,
                    0.75,
                    0.8,
                    0.85,
                    0.9,
                    0.95,
                ],
            },
            "type": "APAveragedOverIOUs",
            "value": 0.0,
        },
        {
            "parameters": {
                "ious": [
                    0.5,
                    0.55,
                    0.6,
                    0.65,
                    0.7,
                    0.75,
                    0.8,
                    0.85,
                    0.9,
                    0.95,
                ],
                "label_key": "k2",
            },
            "type": "mAPAveragedOverIOUs",
            "value": 0.0,
        },
        {
            "parameters": {
                "ious": [
                    0.5,
                    0.55,
                    0.6,
                    0.65,
                    0.7,
                    0.75,
                    0.8,
                    0.85,
                    0.9,
                    0.95,
                ],
                "label_key": "k1",
            },
            "type": "mAPAveragedOverIOUs",
            "value": 0.0,
        },
    ]

    return expected_metrics


@pytest.fixture
def evaluate_detection_functional_test_outputs():
    # cf with torch metrics/pycocotools results listed here:
    # https://github.com/Lightning-AI/metrics/blob/107dbfd5fb158b7ae6d76281df44bd94c836bfce/tests/unittests/detection/test_map.py#L231
    expected_metrics = [
        {
            "label": {"key": "class", "value": "0"},
            "parameters": {"iou": 0.5},
            "value": 1.0,
            "type": "AP",
        },
        {
            "label": {"key": "class", "value": "0"},
            "parameters": {"iou": 0.75},
            "value": 0.723,
            "type": "AP",
        },
        {
            "label": {"key": "class", "value": "2"},
            "parameters": {"iou": 0.5},
            "value": 0.505,
            "type": "AP",
        },
        {
            "label": {"key": "class", "value": "2"},
            "parameters": {"iou": 0.75},
            "value": 0.505,
            "type": "AP",
        },
        {
            "label": {"key": "class", "value": "49"},
            "parameters": {"iou": 0.5},
            "value": 0.791,
            "type": "AP",
        },
        {
            "label": {"key": "class", "value": "49"},
            "parameters": {"iou": 0.75},
            "value": 0.576,
            "type": "AP",
        },
        {
            "label": {"key": "class", "value": "1"},
            "parameters": {"iou": 0.5},
            "value": 1.0,
            "type": "AP",
        },
        {
            "label": {"key": "class", "value": "1"},
            "parameters": {"iou": 0.75},
            "value": 1.0,
            "type": "AP",
        },
        {
            "label": {"key": "class", "value": "4"},
            "parameters": {"iou": 0.5},
            "value": 1.0,
            "type": "AP",
        },
        {
            "label": {"key": "class", "value": "4"},
            "parameters": {"iou": 0.75},
            "value": 1.0,
            "type": "AP",
        },
        {
            "parameters": {"label_key": "class", "iou": 0.5},
            "value": 0.859,
            "type": "mAP",
        },
        {
            "parameters": {"label_key": "class", "iou": 0.75},
            "value": 0.761,
            "type": "mAP",
        },
        {
            "label": {"key": "class", "value": "0"},
            "parameters": {
                "ious": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            },
            "value": 0.725,
            "type": "APAveragedOverIOUs",
        },
        {
            "label": {"key": "class", "value": "2"},
            "parameters": {
                "ious": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            },
            "value": 0.454,
            "type": "APAveragedOverIOUs",
        },
        {
            "label": {"key": "class", "value": "49"},
            "parameters": {
                "ious": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            },
            "value": 0.556,
            "type": "APAveragedOverIOUs",
        },
        {
            "label": {"key": "class", "value": "1"},
            "parameters": {
                "ious": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            },
            "value": 0.8,
            "type": "APAveragedOverIOUs",
        },
        {
            "label": {"key": "class", "value": "4"},
            "parameters": {
                "ious": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            },
            "value": 0.65,
            "type": "APAveragedOverIOUs",
        },
        {
            "parameters": {
                "label_key": "class",
                "ious": [
                    0.5,
                    0.55,
                    0.6,
                    0.65,
                    0.7,
                    0.75,
                    0.8,
                    0.85,
                    0.9,
                    0.95,
                ],
            },
            "value": 0.637,
            "type": "mAPAveragedOverIOUs",
        },
        {
            "label": {"key": "class", "value": "0"},
            "parameters": {
                "ious": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            },
            "value": 0.78,
            "type": "AR",
        },
        {
            "label": {"key": "class", "value": "2"},
            "parameters": {
                "ious": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            },
            "value": 0.45,
            "type": "AR",
        },
        {
            "label": {"key": "class", "value": "49"},
            "parameters": {
                "ious": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            },
            "value": 0.58,
            "type": "AR",
        },
        {
            "label": {"key": "class", "value": "3"},
            "parameters": {
                "ious": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            },
            "value": -1.0,
            "type": "AR",
        },
        {
            "label": {"key": "class", "value": "1"},
            "parameters": {
                "ious": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            },
            "value": 0.8,
            "type": "AR",
        },
        {
            "label": {"key": "class", "value": "4"},
            "parameters": {
                "ious": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            },
            "value": 0.65,
            "type": "AR",
        },
        {
            "parameters": {
                "label_key": "class",
                "ious": [
                    0.5,
                    0.55,
                    0.6,
                    0.65,
                    0.7,
                    0.75,
                    0.8,
                    0.85,
                    0.9,
                    0.95,
                ],
            },
            "value": 0.652,
            "type": "mAR",
        },
    ]

    pr_expected_answers = {
        # (class, 4)
        ("class", "4", 0.05, "tp"): 2,
        ("class", "4", 0.05, "fn"): 0,
        ("class", "4", 0.25, "tp"): 1,
        ("class", "4", 0.25, "fn"): 1,
        ("class", "4", 0.55, "tp"): 0,
        ("class", "4", 0.55, "fn"): 2,
        # (class, 2)
        ("class", "2", 0.05, "tp"): 1,
        ("class", "2", 0.05, "fn"): 1,
        ("class", "2", 0.75, "tp"): 0,
        ("class", "2", 0.75, "fn"): 2,
        # (class, 49)
        ("class", "49", 0.05, "tp"): 8,
        ("class", "49", 0.3, "tp"): 5,
        ("class", "49", 0.5, "tp"): 4,
        ("class", "49", 0.85, "tp"): 1,
        # (class, 3)
        ("class", "3", 0.05, "tp"): 0,
        ("class", "3", 0.05, "fp"): 1,
        # (class, 1)
        ("class", "1", 0.05, "tp"): 1,
        ("class", "1", 0.35, "tp"): 0,
        # (class, 0)
        ("class", "0", 0.05, "tp"): 5,
        ("class", "0", 0.5, "tp"): 3,
        ("class", "0", 0.95, "tp"): 1,
        ("class", "0", 0.95, "fn"): 4,
    }

    detailed_pr_expected_answers = {
        # (class, 4)
        ("4", 0.05, "tp"): {"all": 2, "total": 2},
        ("4", 0.05, "fn"): {
            "no_predictions": 0,
            "misclassifications": 0,
            "total": 0,
        },
        # (class, 2)
        ("2", 0.05, "tp"): {"all": 1, "total": 1},
        ("2", 0.05, "fn"): {
            "no_predictions": 0,
            "misclassifications": 1,
            "total": 1,
        },
        ("2", 0.75, "tp"): {"all": 0, "total": 0},
        ("2", 0.75, "fn"): {
            "no_predictions": 2,
            "misclassifications": 0,
            "total": 2,
        },
        # (class, 49)
        ("49", 0.05, "tp"): {"all": 9, "total": 9},
        # (class, 3)
        ("3", 0.05, "tp"): {"all": 0, "total": 0},
        ("3", 0.05, "fp"): {
            "hallucinations": 0,
            "misclassifications": 1,
            "total": 1,
        },
        # (class, 1)
        ("1", 0.05, "tp"): {"all": 1, "total": 1},
        ("1", 0.8, "fn"): {
            "no_predictions": 1,
            "misclassifications": 0,
            "total": 1,
        },
        # (class, 0)
        ("0", 0.05, "tp"): {"all": 5, "total": 5},
        ("0", 0.95, "fn"): {
            "no_predictions": 4,
            "misclassifications": 0,
            "total": 4,
        },
    }

    higher_iou_threshold_pr_expected_answers = {
        # (class, 4)
        ("class", "4", 0.05, "tp"): 0,
        ("class", "4", 0.05, "fn"): 2,
        # (class, 2)
        ("class", "2", 0.05, "tp"): 1,
        ("class", "2", 0.05, "fn"): 1,
        ("class", "2", 0.75, "tp"): 0,
        ("class", "2", 0.75, "fn"): 2,
        # (class, 49)
        ("class", "49", 0.05, "tp"): 2,
        ("class", "49", 0.3, "tp"): 2,
        ("class", "49", 0.5, "tp"): 2,
        ("class", "49", 0.85, "tp"): 1,
        # (class, 3)
        ("class", "3", 0.05, "tp"): 0,
        ("class", "3", 0.05, "fp"): 1,
        # (class, 1)
        ("class", "1", 0.05, "tp"): 0,
        ("class", "1", 0.05, "fn"): 1,
        # (class, 0)
        ("class", "0", 0.05, "tp"): 1,
        ("class", "0", 0.5, "tp"): 0,
        ("class", "0", 0.95, "fn"): 5,
    }

    higher_iou_threshold_detailed_pr_expected_answers = {
        # (class, 4)
        ("4", 0.05, "tp"): {"all": 0, "total": 0},
        ("4", 0.05, "fn"): {
            "no_predictions": 2,  # below IOU threshold of .9
            "misclassifications": 0,
            "total": 2,
        },
        # (class, 2)
        ("2", 0.05, "tp"): {"all": 1, "total": 1},
        ("2", 0.05, "fn"): {
            "no_predictions": 1,
            "misclassifications": 0,
            "total": 1,
        },
        ("2", 0.75, "tp"): {"all": 0, "total": 0},
        ("2", 0.75, "fn"): {
            "no_predictions": 2,
            "misclassifications": 0,
            "total": 2,
        },
        # (class, 49)
        ("49", 0.05, "tp"): {"all": 2, "total": 2},
        # (class, 3)
        ("3", 0.05, "tp"): {"all": 0, "total": 0},
        ("3", 0.05, "fp"): {
            "hallucinations": 1,
            "misclassifications": 0,
            "total": 1,
        },
        # (class, 1)
        ("1", 0.05, "tp"): {"all": 0, "total": 0},
        ("1", 0.8, "fn"): {
            "no_predictions": 1,
            "misclassifications": 0,
            "total": 1,
        },
        # (class, 0)
        ("0", 0.05, "tp"): {"all": 1, "total": 1},
        ("0", 0.95, "fn"): {
            "no_predictions": 5,
            "misclassifications": 0,
            "total": 5,
        },
    }

    return (
        expected_metrics,
        pr_expected_answers,
        detailed_pr_expected_answers,
        higher_iou_threshold_pr_expected_answers,
        higher_iou_threshold_detailed_pr_expected_answers,
    )


@pytest.fixture
def evaluate_detection_functional_test_with_rasters_output():

    expected_metrics = [
        {
            "label": {"key": "class", "value": "label1"},
            "parameters": {"iou": 0.5},
            "value": 1.0,
            "type": "AP",
        },
        {
            "label": {"key": "class", "value": "label1"},
            "parameters": {"iou": 0.75},
            "value": 1.0,
            "type": "AP",
        },
        {
            "label": {"key": "class", "value": "label2"},
            "parameters": {"iou": 0.5},
            "value": 1.0,
            "type": "AP",
        },
        {
            "label": {"key": "class", "value": "label2"},
            "parameters": {"iou": 0.75},
            "value": 1.0,
            "type": "AP",
        },
        {
            "label": {"key": "class", "value": "label3"},
            "parameters": {"iou": 0.5},
            "value": 0.0,
            "type": "AP",
        },
        {
            "label": {"key": "class", "value": "label3"},
            "parameters": {"iou": 0.75},
            "value": 0.0,
            "type": "AP",
        },
        {
            "parameters": {"label_key": "class", "iou": 0.5},
            "value": 0.667,
            "type": "mAP",
        },
        {
            "parameters": {"label_key": "class", "iou": 0.75},
            "value": 0.667,
            "type": "mAP",
        },
        {
            "label": {"key": "class", "value": "label1"},
            "parameters": {
                "ious": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            },
            "value": 1.0,
            "type": "APAveragedOverIOUs",
        },
        {
            "label": {"key": "class", "value": "label2"},
            "parameters": {
                "ious": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            },
            "value": 1.0,
            "type": "APAveragedOverIOUs",
        },
        {
            "label": {"key": "class", "value": "label3"},
            "parameters": {
                "ious": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            },
            "value": 0.0,
            "type": "APAveragedOverIOUs",
        },
        {
            "parameters": {
                "label_key": "class",
                "ious": [
                    0.5,
                    0.55,
                    0.6,
                    0.65,
                    0.7,
                    0.75,
                    0.8,
                    0.85,
                    0.9,
                    0.95,
                ],
            },
            "value": 0.667,
            "type": "mAPAveragedOverIOUs",
        },
        {
            "label": {"key": "class", "value": "label1"},
            "parameters": {
                "ious": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            },
            "value": 1.0,
            "type": "AR",
        },
        {
            "label": {"key": "class", "value": "label4"},
            "parameters": {
                "ious": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            },
            "value": -1.0,
            "type": "AR",
        },
        {
            "label": {"key": "class", "value": "label2"},
            "parameters": {
                "ious": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            },
            "value": 1.0,
            "type": "AR",
        },
        {
            "label": {"key": "class", "value": "label3"},
            "parameters": {
                "ious": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            },
            "value": 0.0,
            "type": "AR",
        },
        {
            "parameters": {
                "label_key": "class",
                "ious": [
                    0.5,
                    0.55,
                    0.6,
                    0.65,
                    0.7,
                    0.75,
                    0.8,
                    0.85,
                    0.9,
                    0.95,
                ],
            },
            "value": 0.667,
            "type": "mAR",
        },
    ]

    pr_expected_answers = {
        ("class", "label1", 0.05, "tp"): 1,
        ("class", "label1", 0.35, "tp"): 0,
        ("class", "label2", 0.05, "tp"): 1,
        ("class", "label2", 0.05, "fp"): 0,
        ("class", "label2", 0.95, "fp"): 0,
        ("class", "label3", 0.05, "tp"): 0,
        ("class", "label3", 0.05, "fn"): 1,
        ("class", "label4", 0.05, "tp"): 0,
        ("class", "label4", 0.05, "fp"): 1,
    }

    return expected_metrics, pr_expected_answers


@pytest.fixture
def evaluate_detection_functional_test_with_rasters_outputs():

    expected_metrics = [
        {
            "label": {"key": "class", "value": "label1"},
            "parameters": {"iou": 0.5},
            "value": 1.0,
            "type": "AP",
        },
        {
            "label": {"key": "class", "value": "label1"},
            "parameters": {"iou": 0.75},
            "value": 1.0,
            "type": "AP",
        },
        {
            "label": {"key": "class", "value": "label2"},
            "parameters": {"iou": 0.5},
            "value": 1.0,
            "type": "AP",
        },
        {
            "label": {"key": "class", "value": "label2"},
            "parameters": {"iou": 0.75},
            "value": 1.0,
            "type": "AP",
        },
        {
            "label": {"key": "class", "value": "label3"},
            "parameters": {"iou": 0.5},
            "value": 0.0,
            "type": "AP",
        },
        {
            "label": {"key": "class", "value": "label3"},
            "parameters": {"iou": 0.75},
            "value": 0.0,
            "type": "AP",
        },
        {
            "parameters": {"label_key": "class", "iou": 0.5},
            "value": 0.667,
            "type": "mAP",
        },
        {
            "parameters": {"label_key": "class", "iou": 0.75},
            "value": 0.667,
            "type": "mAP",
        },
        {
            "label": {"key": "class", "value": "label1"},
            "parameters": {
                "ious": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            },
            "value": 1.0,
            "type": "APAveragedOverIOUs",
        },
        {
            "label": {"key": "class", "value": "label2"},
            "parameters": {
                "ious": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            },
            "value": 1.0,
            "type": "APAveragedOverIOUs",
        },
        {
            "label": {"key": "class", "value": "label3"},
            "parameters": {
                "ious": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            },
            "value": 0.0,
            "type": "APAveragedOverIOUs",
        },
        {
            "parameters": {
                "label_key": "class",
                "ious": [
                    0.5,
                    0.55,
                    0.6,
                    0.65,
                    0.7,
                    0.75,
                    0.8,
                    0.85,
                    0.9,
                    0.95,
                ],
            },
            "value": 0.667,
            "type": "mAPAveragedOverIOUs",
        },
        {
            "label": {"key": "class", "value": "label1"},
            "parameters": {
                "ious": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            },
            "value": 1.0,
            "type": "AR",
        },
        {
            "label": {"key": "class", "value": "label4"},
            "parameters": {
                "ious": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            },
            "value": -1.0,
            "type": "AR",
        },
        {
            "label": {"key": "class", "value": "label2"},
            "parameters": {
                "ious": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            },
            "value": 1.0,
            "type": "AR",
        },
        {
            "label": {"key": "class", "value": "label3"},
            "parameters": {
                "ious": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            },
            "value": 0.0,
            "type": "AR",
        },
        {
            "parameters": {
                "label_key": "class",
                "ious": [
                    0.5,
                    0.55,
                    0.6,
                    0.65,
                    0.7,
                    0.75,
                    0.8,
                    0.85,
                    0.9,
                    0.95,
                ],
            },
            "value": 0.667,
            "type": "mAR",
        },
    ]

    pr_expected_answers = {
        ("class", "label1", 0.05, "tp"): 1,
        ("class", "label1", 0.35, "tp"): 0,
        ("class", "label2", 0.05, "tp"): 1,
        ("class", "label2", 0.05, "fp"): 0,
        ("class", "label2", 0.95, "fp"): 0,
        ("class", "label3", 0.05, "tp"): 0,
        ("class", "label3", 0.05, "fn"): 1,
        ("class", "label4", 0.05, "tp"): 0,
        ("class", "label4", 0.05, "fp"): 1,
    }

    return expected_metrics, pr_expected_answers


@pytest.fixture()
def evaluate_mixed_annotations_output():

    expected = [
        {
            "type": "AP",
            "parameters": {"iou": 0.1},
            "value": 1.0,
            "label": {"key": "key", "value": "value"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.6},
            "value": 1.0,
            "label": {"key": "key", "value": "value"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.1},
            "value": 1.0,
            "label": {"key": "key2", "value": "value"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.6},
            "value": 1.0,
            "label": {"key": "key2", "value": "value"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.1},
            "value": 1.0,
            "label": {"key": "key1", "value": "value"},
        },
        {
            "type": "AP",
            "parameters": {"iou": 0.6},
            "value": 1.0,
            "label": {"key": "key1", "value": "value"},
        },
    ]

    return expected


@pytest.fixture
def detailed_curve_examples_output():

    expected_outputs = {
        ("bee", 0.05, "tp", "all"): {
            ("datum1",),
            ("datum3",),
        },
        (
            "bee",
            0.05,
            "fp",
            "misclassifications",
        ): {("datum2",), ("datum0",), ("datum4",)},
        ("dog", 0.05, "tp", "all"): {("datum4",)},
        (
            "dog",
            0.05,
            "fp",
            "misclassifications",
        ): {("datum2",), ("datum0",), ("datum3",), ("datum1",)},
        ("cat", 0.05, "tp", "all"): {
            ("datum2",),
            ("datum0",),
        },
        (
            "cat",
            0.05,
            "fp",
            "misclassifications",
        ): {("datum3",), ("datum1",), ("datum4",)},
        ("bee", 0.85, "tn", "all"): {
            ("datum0",),
            ("datum2",),
            ("datum4",),
        },
        ("bee", 0.85, "fn", "no_predictions"): {
            ("datum3",),
            ("datum1",),
        },
        ("dog", 0.85, "tn", "all"): {
            ("datum2",),
            ("datum0",),
            ("datum3",),
            ("datum1",),
        },
        ("dog", 0.85, "fn", "no_predictions"): {("datum4",)},
        ("cat", 0.85, "tn", "all"): {
            ("datum1",),
            ("datum3",),
            ("datum4",),
        },
        ("cat", 0.85, "fn", "no_predictions"): {
            ("datum2",),
            ("datum0",),
        },
        # check cases where we shouldn't have any examples since the count is zero
        ("bee", 0.3, "fn", "misclassifications"): set(),
        ("dog", 0.1, "tn", "all"): set(),
    }

    return expected_outputs


@pytest.fixture
def detailed_curve_examples_check_zero_count_examples_output():
    expected_outputs = {
        ("ant", 0.05, "fp", "misclassifications"): 0,
        ("ant", 0.95, "tn", "all"): 0,
        ("bee", 0.2, "fn", "misclassifications"): 0,
        ("cat", 0.2, "fn", "misclassifications"): 0,
    }

    return expected_outputs
