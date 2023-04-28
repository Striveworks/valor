import numpy as np

from velour_api.metrics.classification import (
    confusion_matrix,
    precision_and_recall_f1_at_class_index_from_confusion_matrix,
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
