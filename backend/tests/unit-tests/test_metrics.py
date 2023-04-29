import numpy as np
import pytest

from velour_api.metrics.classification import (
    _binary_roc_auc,
    confusion_matrix,
    precision_and_recall_f1_at_class_index_from_confusion_matrix,
    roc_auc,
)
from velour_api.metrics.detection import _match_array


def test__match_array():
    ious = [
        [0.0, 0.004, 0.78],
        [0.65, 0.0, 0.0],
        [0.0, 0.8, 0.051],
        [0.71, 0.0, 0.0],
    ]

    assert _match_array(ious, 1.0) == [None, None, None, None]

    assert _match_array(ious, 0.75) == [2, None, 1, None]

    assert _match_array(ious, 0.7) == [2, None, 1, 0]

    # check that match to groundtruth 0 switches
    assert _match_array(ious, 0.1) == [2, 0, 1, None]

    assert _match_array(ious, 0.0) == [2, 0, 1, None]


def test_confusion_matrix():
    groundtruths = ["bird", "dog", "bird", "bird", "cat", "dog"]
    preds = ["bird", "cat", "cat", "dog", "cat", "cat"]
    expected_output = np.array([[1, 1, 1], [0, 1, 0], [0, 2, 0]])
    np.testing.assert_equal(
        confusion_matrix(groundtruths, preds), expected_output
    )


def test_confusion_matrix_empty_lists():
    groundtruths = []
    preds = []
    expected_output = np.zeros((0, 0), dtype=int)
    np.testing.assert_equal(
        confusion_matrix(groundtruths, preds), expected_output
    )


def test_precision_and_recall_f1_at_class_index_from_confusion_matrix():
    cm = np.array([[1, 1, 1], [0, 1, 0], [0, 2, 0]])

    expected_precisions = [1, 0.25, 0]
    expected_recalls = [1 / 3, 1, 0]
    expected_f1 = [0.5, 0.4, 0]

    for i in range(3):
        (
            p,
            r,
            f1,
        ) = precision_and_recall_f1_at_class_index_from_confusion_matrix(cm, i)
        assert p == expected_precisions[i]
        assert r == expected_recalls[i]
        assert f1 == expected_f1[i]


def test__binary_roc_auc():
    preds = [0.32, 0.63, 0.39, 0.46, 0.16, 0.65, 0.65, 0.03, 0.85, 0.35]
    gts = [False, False, False, True, False, False, False, True, False, True]
    # this agrees with scikit learn:
    # from sklearn.metrics import roc_auc_score
    # roc_auc_score(y_true=y, y_score=y_pred)
    # > 0.23809523809523808
    assert round(_binary_roc_auc(groundtruths=gts, preds=preds), 6) == 0.238095


def test_roc_auc():
    preds = [
        {"c1": 0.3568, "c2": 0.3408, "c3": 0.3024},
        {"c1": 0.3347, "c2": 0.3429, "c3": 0.3224},
        {"c1": 0.3984, "c2": 0.3015, "c3": 0.3001},
        {"c1": 0.3808, "c2": 0.2226, "c3": 0.3966},
        {"c1": 0.4145, "c2": 0.1888, "c3": 0.3967},
    ]
    gts = ["c1", "c3", "c2", "c3", "c1"]
    # this agrees with scikit learn:
    # if y_true = [0, 2, 1, 2, 0]
    # and y_score = [[0.3568 0.3408 0.3024]
    # [0.3347 0.3429 0.3224]
    # [0.3984 0.3015 0.3001]
    # [0.3808 0.2226 0.3966]
    # [0.4145 0.1888 0.3967]]
    # then
    # from sklearn.metrics import roc_auc_score
    # roc_auc_score(y_true=y, y_score=y_pred)
    # > 0.61111111
    assert round(roc_auc(groundtruths=gts, preds=preds), 6) == 0.611111


def test_roc_auc_error():
    """Check we get an error if we give a prediction that doesn't sum to 1"""
    preds = [{"c1": 0.2, "c2": 0.6}, {"c1": 0.3, "c2": 0.7}]
    gts = ["c1", "c1"]

    with pytest.raises(ValueError) as exc_info:
        roc_auc(gts, preds)

    assert "Sum of predictions" in str(exc_info)
