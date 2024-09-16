import numpy as np
from numpy.typing import NDArray

# 0 datum index
# 1 gt label index
# 2 pd label index
# 3 score


def compute_metrics(
    data: NDArray[np.floating],
    label_metadata: NDArray[np.int32],
    score_thresholds: NDArray[np.floating],
) -> tuple[
    NDArray[np.int32],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
]:
    """
    Computes classification metrics.

    Parameters
    ----------
    data : NDArray[np.floating]
        A sorted array of classification pairs.
    label_metadata : NDArray[np.int32]
        An array containing metadata related to labels.
    score_thresholds : NDArray[np.floating]
        An array contains score thresholds to compute metrics over.

    Returns
    -------
    NDArray[np.int32]
        TP, FP, FN, TN counts.
    NDArray[np.floating]
        Recall.
    NDArray[np.floating]
        Precision.
    NDArray[np.floating]
        Accuracy
    NDArray[np.floating]
        F1 Score
    """

    n_labels = label_metadata.shape[0]
    n_scores = score_thresholds.shape[0]

    counts = np.zeros((n_scores, n_labels, 4), dtype=np.int32)
    recall = np.zeros((n_scores, n_labels), dtype=np.float64)
    precision = np.zeros_like(recall)
    accuracy = np.zeros_like(recall)
    f1_score = np.zeros_like(recall)

    pd_labels = data[:, 2].astype(int)
    total_count = np.bincount(pd_labels, minlength=n_labels)

    mask_gt_exists = data[:, 1] >= 0.0
    mask_pd_exists = data[:, 2] >= 0.0
    mask_matching_labels = np.isclose(data[:, 1], data[:, 2])
    mask_score_nonzero = ~np.isclose(data[:, 3], 0.0)

    # positive_count = np.bincount(
    #     pd_labels[mask_matching_labels], minlength=n_labels
    # )
    # negative_count = np.bincount(
    #     pd_labels[~mask_matching_labels], minlength=n_labels
    # )

    for score_idx in range(n_scores):
        mask_score_threshold = data[:, 3] >= score_thresholds[score_idx]
        mask_score = mask_score_nonzero & mask_score_threshold

        mask_tp = (
            mask_gt_exists & mask_pd_exists & mask_matching_labels & mask_score
        )
        mask_fp = (
            mask_pd_exists
            & (~mask_gt_exists | ~mask_matching_labels)
            & mask_score
        )
        mask_fn = mask_gt_exists & (
            (mask_pd_exists & mask_matching_labels & ~mask_score)
            | ~mask_pd_exists
        )
        mask_tn = (
            mask_pd_exists
            & mask_gt_exists
            & ~mask_matching_labels
            & ~mask_score
        )

        counts[score_idx][0] = np.bincount(
            pd_labels[mask_tp], minlength=n_labels
        )
        counts[score_idx][1] = np.bincount(
            pd_labels[mask_fp], minlength=n_labels
        )
        counts[score_idx][2] = np.bincount(
            pd_labels[mask_fn], minlength=n_labels
        )
        counts[score_idx][3] = np.bincount(
            pd_labels[mask_tn], minlength=n_labels
        )

    np.divide(
        counts[:, 0],
        (counts[:, 0] + counts[:, 2]),
        where=(counts[:, 0] + counts[:, 2]) > 1e-9,
        out=recall,
    )

    np.divide(
        counts[:, 0],
        (counts[:, 0] + counts[:, 1]),
        where=(counts[:, 0] + counts[:, 1]) > 1e-9,
        out=recall,
    )

    np.divide(
        (counts[:, 0] + counts[:, 3]),
        total_count,
        where=total_count > 1e-9,
        out=accuracy,
    )

    np.divide(
        (2 * precision * recall),
        (precision + recall),
        where=(precision + recall) > 1e-9,
        out=f1_score,
    )

    return (
        counts,
        precision,
        recall,
        accuracy,
        f1_score,
    )


def compute_detailed_pr_curve() -> tuple:
    return tuple()
