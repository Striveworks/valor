import numpy as np
from numpy.typing import NDArray

# 0 datum index
# 1 gt label index
# 2 pd label index
# 3 score


def _compute_rocauc(
    data: NDArray[np.float64],
    label_metadata: NDArray[np.int32],
    n_datums: int,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:

    n_labels = label_metadata.shape[0]
    n_label_keys = np.unique(label_metadata[:, 2]).size

    pd_labels = data[:, 2].astype(int)

    count_labels_per_key = np.bincount(label_metadata[:, 2])
    count_groundtruths_per_key = np.bincount(
        label_metadata[:, 2],
        weights=label_metadata[:, 0],
        minlength=n_label_keys,
    )

    positive_count = label_metadata[:, 0]
    negative_count = (
        count_groundtruths_per_key[label_metadata[:, 2]] - label_metadata[:, 0]
    )

    mask_pd_exists = data[:, 2] >= 0.0
    mask_matching_labels = np.isclose(data[:, 1], data[:, 2])
    mask_true_positives = mask_pd_exists & mask_matching_labels
    mask_false_positives = mask_pd_exists & ~mask_matching_labels

    true_positives = np.zeros((n_labels, n_datums), dtype=np.int32)
    false_positives = np.zeros_like(true_positives)
    scores = np.zeros_like(true_positives, dtype=np.float64)
    for label_idx in range(n_labels):
        true_positives[label_idx] = mask_true_positives[pd_labels == label_idx]
        false_positives[label_idx] = mask_false_positives[
            pd_labels == label_idx
        ]
        scores[label_idx] = data[pd_labels == label_idx, 3]

    cumulative_fp = np.cumsum(false_positives, axis=1)
    cumulative_tp = np.cumsum(true_positives, axis=1)

    fpr = np.zeros_like(true_positives, dtype=np.float64)
    np.divide(
        cumulative_fp,
        negative_count[:, np.newaxis],
        where=negative_count[:, np.newaxis] > 1e-9,
        out=fpr,
    )
    tpr = np.zeros_like(true_positives, dtype=np.float64)
    np.divide(
        cumulative_tp,
        positive_count[:, np.newaxis],
        where=positive_count[:, np.newaxis] > 1e-9,
        out=tpr,
    )

    # sort by -tpr, -score
    indices = np.lexsort((-tpr, -scores), axis=1)
    fpr = np.take_along_axis(fpr, indices, axis=1)
    tpr = np.take_along_axis(tpr, indices, axis=1)

    # running max of tpr
    np.maximum.accumulate(tpr, axis=1, out=tpr)

    # compute rocauc
    rocauc = np.trapz(x=fpr, y=tpr, axis=1)

    # compute mean rocauc
    summed_rocauc = np.bincount(label_metadata[:, 2], weights=rocauc)
    mean_rocauc = np.zeros(n_label_keys, dtype=np.float64)
    np.divide(
        summed_rocauc,
        count_labels_per_key,
        where=count_labels_per_key > 1e-9,
        out=mean_rocauc,
    )

    return (rocauc, mean_rocauc)


def compute_metrics(
    data: NDArray[np.floating],
    label_metadata: NDArray[np.int32],
    score_thresholds: NDArray[np.floating],
    n_datums: int,
) -> tuple[
    NDArray[np.int32],
    NDArray[np.floating],
    NDArray[np.floating],
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
    NDArray[np.floating]
        ROCAUC.
    NDArray[np.floating]
        mROCAUC.
    """

    n_labels = label_metadata.shape[0]
    n_scores = score_thresholds.shape[0]

    counts = np.zeros((n_scores, n_labels, 4), dtype=np.int32)

    pd_labels = data[:, 2].astype(int)

    mask_gt_exists = data[:, 1] >= 0.0
    mask_pd_exists = data[:, 2] >= 0.0
    mask_matching_labels = np.isclose(data[:, 1], data[:, 2])
    mask_score_nonzero = ~np.isclose(data[:, 3], 0.0)

    # calculate reciever-operating-characteristic (ROC) curve
    rocauc, mean_rocauc = _compute_rocauc(
        data=data,
        label_metadata=label_metadata,
        n_datums=n_datums,
    )

    # calculate metrics at each score threshold
    total_count = np.bincount(pd_labels[mask_pd_exists], minlength=n_labels)
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

    recall = np.zeros((n_scores, n_labels), dtype=np.float64)
    np.divide(
        counts[:, 0],
        (counts[:, 0] + counts[:, 2]),
        where=(counts[:, 0] + counts[:, 2]) > 1e-9,
        out=recall,
    )

    precision = np.zeros_like(recall)
    np.divide(
        counts[:, 0],
        (counts[:, 0] + counts[:, 1]),
        where=(counts[:, 0] + counts[:, 1]) > 1e-9,
        out=recall,
    )

    accuracy = np.zeros_like(recall)
    np.divide(
        (counts[:, 0] + counts[:, 3]),
        total_count,
        where=total_count > 1e-9,
        out=accuracy,
    )

    f1_score = np.zeros_like(recall)
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
        rocauc,
        mean_rocauc,
    )


def compute_detailed_pr_curve() -> tuple:
    return tuple()
