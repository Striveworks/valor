import numpy as np
import pytest
from numpy.typing import NDArray

from valor_lite.classification.computation import (
    PairClassification,
    compute_accuracy,
    compute_confusion_matrix,
    compute_counts,
    compute_f1_score,
    compute_pair_classifications,
    compute_precision,
    compute_recall,
    compute_rocauc,
)


@pytest.fixture
def counts() -> NDArray[np.uint64]:
    return np.array([])


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

    score_thresholds = np.array([0.25, 0.75], dtype=np.float64)

    counts = compute_counts(
        ids=data[:, (0, 1, 2)].astype(np.int64),
        scores=data[:, 3],
        winners=data[:, 4] > 0.5,
        score_thresholds=score_thresholds,
        n_labels=4,
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

    score_thresholds = np.array([0.25, 0.75], dtype=np.float64)

    counts = compute_counts(
        ids=data[:, (0, 1, 2)].astype(np.int64),
        scores=data[:, 3],
        winners=data[:, 4] > 0.5,
        score_thresholds=score_thresholds,
        n_labels=4,
        hardmax=False,
    )
    precision = compute_precision(counts)

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


def test_recall_computation():

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

    score_thresholds = np.array([0.25, 0.75], dtype=np.float64)

    counts = compute_counts(
        ids=data[:, (0, 1, 2)].astype(np.int64),
        scores=data[:, 3],
        winners=data[:, 4] > 0.5,
        score_thresholds=score_thresholds,
        n_labels=4,
        hardmax=False,
    )
    recall = compute_recall(counts)

    # score threshold, label, count metric
    assert recall.shape == (2, 4)

    # score >= 0.25
    assert recall[0][0] == 0.5
    assert recall[0][1] == 0.0
    assert recall[0][2] == 0.0
    assert recall[0][3] == 1.0
    # score >= 0.75
    assert recall[1][0] == 0.5
    assert recall[1][1] == 0.0
    assert recall[1][2] == 0.0
    assert recall[1][3] == 0.0


def test_compute_accuracy():

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

    score_thresholds = np.array([0.25, 0.75], dtype=np.float64)

    counts = compute_counts(
        ids=data[:, (0, 1, 2)].astype(np.int64),
        scores=data[:, 3],
        winners=data[:, 4] > 0.5,
        score_thresholds=score_thresholds,
        n_labels=4,
        hardmax=False,
    )
    accuracy = compute_accuracy(counts, n_datums=3)

    # score threshold, label, count metric
    assert accuracy.shape == (2,)

    # score >= 0.25
    assert accuracy[0] == 2 / 3
    # score >= 0.75
    assert accuracy[1] == 1 / 3


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

    score_thresholds = np.array([0.25, 0.75], dtype=np.float64)

    counts = compute_counts(
        ids=data[:, (0, 1, 2)].astype(np.int64),
        scores=data[:, 3],
        winners=data[:, 4] > 0.5,
        score_thresholds=score_thresholds,
        n_labels=4,
        hardmax=False,
    )
    precision = compute_precision(counts)
    recall = compute_recall(counts)
    f1_score = compute_f1_score(precision, recall)

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


def test_compute_rocauc_animals():
    """
    Test ROC auc computation. This agrees with scikit-learn: the code (whose data
    comes from classification_test_data)

    animal_gts = ["bird", "dog", "bird", "bird", "cat", "dog"]
    animal_preds = [
        {"bird": 0.6, "dog": 0.2, "cat": 0.2},
        {"cat": 0.9, "dog": 0.1, "bird": 0.0},
        {"cat": 0.8, "dog": 0.05, "bird": 0.15},
        {"dog": 0.75, "cat": 0.1, "bird": 0.15},
        {"cat": 1.0, "dog": 0.0, "bird": 0.0},
        {"cat": 0.4, "dog": 0.4, "bird": 0.2},
    ]

    ```
    from sklearn.metrics import roc_auc_score

    # for the "animal" label key
    y_true = [0, 2, 0, 0, 1, 2]
    y_score = [
        [0.6, 0.2, 0.2],
        [0.0, 0.9, 0.1],
        [0.15, 0.8, 0.05],
        [0.15, 0.1, 0.75],
        [0.0, 1.0, 0.0],
        [0.2, 0.4, 0.4],
    ]
    print(roc_auc_score(y_true, y_score, multi_class="ovr"))
    ```
    outputs ==> 0.8009259259259259
    """

    # groundtruth, prediction, score
    data = np.array(
        [
            # datum 0
            [0, 0, 0, 0.6, 1.0],
            [0, 0, 1, 0.2, 0.0],
            [0, 0, 2, 0.2, 0.0],
            # datum 1
            [1, 2, 0, 0.0, 0.0],
            [1, 2, 1, 0.9, 1.0],
            [1, 2, 2, 0.1, 0.0],
            # datum 2
            [2, 0, 0, 0.15, 0.0],
            [2, 0, 1, 0.8, 1.0],
            [2, 0, 2, 0.05, 0.0],
            # datum 3
            [3, 0, 0, 0.15, 0.0],
            [3, 0, 1, 0.1, 0.0],
            [3, 0, 2, 0.75, 1.0],
            # datum 4
            [4, 1, 0, 0.0, 0.0],
            [4, 1, 1, 1.0, 1.0],
            [4, 1, 2, 0.0, 0.0],
            # datum 5
            [5, 2, 0, 0.2, 0.0],
            [5, 2, 1, 0.4, 1.0],
            [5, 2, 2, 0.4, 0.0],
        ],
        dtype=np.float64,
    )
    indices = np.lexsort([data[:, 2], data[:, 1], data[:, 0], -data[:, 3]])
    data = data[indices]

    n_datums = 6
    n_labels = 3

    # compute ROCAUC and mROCAUC
    rocauc, prev_fp, prev_tp = compute_rocauc(
        ids=data[:, (0, 1, 2)].astype(np.int64),
        scores=data[:, 3],
        gt_count_per_label=np.array([3, 1, 2], dtype=np.uint64),
        pd_count_per_label=np.array([6, 6, 6], dtype=np.uint64),
        n_datums=n_datums,
        n_labels=n_labels,
        prev_cumulative_fp=np.zeros((n_labels, 1), dtype=np.uint64),
        prev_cumulative_tp=np.zeros((n_labels, 1), dtype=np.uint64),
    )

    # test intermediates
    assert (prev_fp == np.array([3, 5, 4])).all()
    assert (prev_tp == np.array([3, 1, 2])).all()

    # test ROCAUC
    assert rocauc[0] == 0.7777777777777778  # (animal, bird)
    assert rocauc[1] == 1.0  # (animal, cat)
    assert rocauc[2] == 0.625  # (animal, dog)

    # test mROCAUC
    mean_rocauc = rocauc.mean()
    assert mean_rocauc == 0.8009259259259259


def test_compute_rocauc_colors():
    """
    Test ROC auc computation. This agrees with scikit-learn: the code (whose data
    comes from classification_test_data)

    color_gts = ["white", "white", "red", "blue", "black", "red"]
    color_preds = [
        {"white": 0.65, "red": 0.1, "blue": 0.2, "black": 0.05},
        {"blue": 0.5, "white": 0.3, "red": 0.0, "black": 0.2},
        {"red": 0.4, "white": 0.2, "blue": 0.1, "black": 0.3},
        {"white": 1.0, "red": 0.0, "blue": 0.0, "black": 0.0},
        {"red": 0.8, "white": 0.0, "blue": 0.2, "black": 0.0},
        {"red": 0.9, "white": 0.06, "blue": 0.01, "black": 0.03},
    ]

    ```
    from sklearn.metrics import roc_auc_score

    # for the "color" label key
    y_true = [3, 3, 2, 1, 0, 2]
    y_score = [
        [0.05, 0.2, 0.1, 0.65],
        [0.2, 0.5, 0.0, 0.3],
        [0.3, 0.1, 0.4, 0.2],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.2, 0.8, 0.0],
        [0.03, 0.01, 0.9, 0.06],
    ]
    print(roc_auc_score(y_true, y_score, multi_class="ovr"))
    ```

    outputs:

    ```
    0.43125
    ```
    """

    # groundtruth, prediction, score
    data = np.array(
        [
            # datum 0
            [0, 3, 0, 0.05, 0.0],
            [0, 3, 1, 0.2, 0.0],
            [0, 3, 2, 0.1, 0.0],
            [0, 3, 3, 0.65, 1.0],
            # datum 1
            [1, 3, 0, 0.2, 0.0],
            [1, 3, 1, 0.5, 1.0],
            [1, 3, 2, 0.0, 0.0],
            [1, 3, 3, 0.3, 0.0],
            # datum 2
            [2, 2, 0, 0.3, 0.0],
            [2, 2, 1, 0.1, 0.0],
            [2, 2, 2, 0.4, 1.0],
            [2, 2, 3, 0.2, 0.0],
            # datum 3
            [3, 1, 0, 0.0, 0.0],
            [3, 1, 1, 0.0, 0.0],
            [3, 1, 2, 0.0, 0.0],
            [3, 1, 3, 1.0, 1.0],
            # datum 4
            [4, 0, 0, 0.0, 0.0],
            [4, 0, 1, 0.2, 0.0],
            [4, 0, 2, 0.8, 1.0],
            [4, 0, 3, 0.0, 0.0],
            # datum 5
            [5, 2, 0, 0.03, 0.0],
            [5, 2, 1, 0.01, 0.0],
            [5, 2, 2, 0.9, 1.0],
            [5, 2, 3, 0.06, 0.0],
        ],
        dtype=np.float64,
    )
    indices = np.lexsort([data[:, 2], data[:, 1], data[:, 0], -data[:, 3]])
    data = data[indices]

    n_datums = 6
    n_labels = 4

    # compute ROCAUC and mROCAUC
    rocauc, prev_fp, prev_tp = compute_rocauc(
        ids=data[:, (0, 1, 2)].astype(np.int64),
        scores=data[:, 3],
        gt_count_per_label=np.array([1, 1, 2, 2], dtype=np.uint64),
        pd_count_per_label=np.array([6, 6, 6, 6], dtype=np.uint64),
        n_datums=n_datums,
        n_labels=n_labels,
        prev_cumulative_fp=np.zeros((n_labels, 1), dtype=np.uint64),
        prev_cumulative_tp=np.zeros((n_labels, 1), dtype=np.uint64),
    )

    # test intermediates
    assert (prev_fp == np.array([5, 5, 4, 4])).all()
    assert (prev_tp == np.array([1, 1, 2, 2])).all()

    # test ROCAUC
    assert rocauc[0] == 0.09999999999999998  # (color, black)
    assert rocauc[1] == 0.0  # (color, blue)
    assert rocauc[2] == 0.875  # (color, red)
    assert rocauc[3] == 0.75  # (color, white)

    # test mROCAUC
    mean_rocauc = rocauc.mean()
    assert mean_rocauc == 0.43125


def test_compute_pair_classifications():

    # groundtruth, prediction, score
    data = np.array(
        [
            # datum 0
            [0, 0, 0, 1.0, 1.0],  # tp
            [0, 0, 1, 0.0, 0.0],  # tn
            [0, 0, 2, 0.0, 0.0],  # tn
            [0, 0, 3, 0.0, 0.0],  # tn
            # datum 1
            [1, 0, 0, 0.0, 0.0],  # fn
            [1, 0, 1, 0.0, 0.0],  # tn
            [1, 0, 2, 1.0, 1.0],  # fp
            [1, 0, 3, 0.0, 0.0],  # tn
            # datum 2
            [2, 3, 0, 0.0, 0.0],  # tn
            [2, 3, 1, 0.0, 0.0],  # tn
            [2, 3, 2, 0.0, 0.0],  # tn
            [2, 3, 3, 0.3, 1.0],  # fn for score threshold > 0.3
        ],
        dtype=np.float64,
    )
    score_thresholds = np.array([0.25, 0.75], dtype=np.float64)

    mask_tp, mask_misclf, mask_unmatched_fn = compute_pair_classifications(
        ids=data[:, (0, 1, 2)].astype(np.int64),
        scores=data[:, 3],
        winners=data[:, 4] > 0.5,
        score_thresholds=score_thresholds,
        hardmax=True,
    )

    result = np.zeros((len(score_thresholds), data.shape[0]), dtype=np.uint8)
    result[mask_tp] = PairClassification.TP
    result[mask_misclf] = PairClassification.FP_FN_MISCLF
    result[mask_unmatched_fn] = PairClassification.FN_UNMATCHED

    assert result.shape == (2, 12)
    assert np.all(
        result
        == np.array(
            [
                [
                    PairClassification.TP,
                    0,
                    0,
                    0,
                    0,
                    0,
                    PairClassification.FP_FN_MISCLF,
                    0,
                    0,
                    0,
                    0,
                    PairClassification.TP,
                ],
                [
                    PairClassification.TP,
                    0,
                    0,
                    0,
                    0,
                    0,
                    PairClassification.FP_FN_MISCLF,
                    0,
                    PairClassification.FN_UNMATCHED,
                    PairClassification.FN_UNMATCHED,
                    PairClassification.FN_UNMATCHED,
                    PairClassification.FN_UNMATCHED,
                ],
            ],
            dtype=np.uint8,
        ),
    )


def test_compute_confusion_matrix():

    # groundtruth, prediction, score
    data = np.array(
        [
            # datum 0
            [0, 0, 0, 1.0, 1.0],  # tp
            [0, 0, 1, 0.0, 0.0],  # tn
            [0, 0, 2, 0.0, 0.0],  # tn
            [0, 0, 3, 0.0, 0.0],  # tn
            # datum 1
            [1, 0, 0, 0.0, 0.0],  # fn
            [1, 0, 1, 0.0, 0.0],  # tn
            [1, 0, 2, 1.0, 1.0],  # fp
            [1, 0, 3, 0.0, 0.0],  # tn
            # datum 2
            [2, 3, 0, 0.0, 0.0],  # tn
            [2, 3, 1, 0.0, 0.0],  # tn
            [2, 3, 2, 0.0, 0.0],  # tn
            [2, 3, 3, 0.3, 1.0],  # fn for score threshold > 0.3
        ],
        dtype=np.float64,
    )
    score_thresholds = np.array([0.25, 0.75], dtype=np.float64)
    n_labels = 4

    mask_tp, mask_misclf, mask_unmatched_fn = compute_pair_classifications(
        ids=data[:, (0, 1, 2)].astype(np.int64),
        scores=data[:, 3],
        winners=data[:, 4] > 0.5,
        score_thresholds=score_thresholds,
        hardmax=True,
    )
    cm, unmatched_gts = compute_confusion_matrix(
        ids=data[:, (0, 1, 2)].astype(np.int64),
        mask_tp=mask_tp,
        mask_fp_fn_misclf=mask_misclf,
        mask_fn_unmatched=mask_unmatched_fn,
        score_thresholds=score_thresholds,
        n_labels=n_labels,
    )

    assert (
        cm
        == np.array(
            [
                [
                    [1, 0, 1, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 1],
                ],
                [
                    [1, 0, 1, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
            ]
        )
    ).all()
    assert (
        unmatched_gts
        == np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 1],
            ]
        )
    ).all()
