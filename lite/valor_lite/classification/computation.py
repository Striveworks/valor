import numpy as np
from numpy.typing import NDArray

# 0 datum index
# 1 label key index
# 2 gt label index
# 3 pd label index
# 4 score


def _compute_rocauc(
    data: NDArray[np.float64],
    n_rows: int,
    gt_labels: NDArray[np.int32],
    pd_labels: NDArray[np.int32],
    mask_pd_exists: NDArray[np.bool_],
    mask_matching_labels: NDArray[np.bool_],
    label_metadata: NDArray[np.int32],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:

    n_labels = label_metadata.shape[0]
    n_label_keys = np.unique(label_metadata[:, 2]).size

    count_labels_per_key = np.bincount(label_metadata[:, 2])
    count_gts_per_key = np.bincount(
        label_metadata[:, 2],
        weights=label_metadata[:, 0],
        minlength=n_label_keys,
    )

    positive_count = label_metadata[:, 0]
    negative_count = (
        count_gts_per_key[label_metadata[:, 2]] - label_metadata[:, 0]
    )

    unique_pd_labels, pd_label_indices = np.unique(
        pd_labels, return_inverse=True
    )
    mask_pd_labels = pd_label_indices[:, np.newaxis] == np.arange(
        unique_pd_labels.size
    )

    rocauc = np.zeros(n_labels, dtype=np.float64)
    for label_idx in range(n_labels):
        n_preds = label_metadata[label_idx, 1]
        if n_preds == 0:
            continue

        mask_prediction = pd_labels == label_idx
        mask_tp = np.zeros(n_preds, dtype=np.bool_)
        mask_tp[mask_matching_labels[mask_prediction]] = True
        mask_fp = ~mask_tp

        cumulative_tp = np.cumsum(mask_tp)
        cumulative_fp = np.cumsum(mask_fp)

        fpr_tpr = np.zeros((2, n_preds), dtype=np.float64)
        np.divide(
            cumulative_fp,
            negative_count[label_idx],
            where=negative_count[label_idx] > 1e-9,
            out=fpr_tpr[0],
        )
        np.divide(
            cumulative_tp,
            positive_count[label_idx],
            where=positive_count[label_idx] > 1e-9,
            out=fpr_tpr[1],
        )

        indices = np.lexsort((-fpr_tpr[1], -data[mask_prediction, -1]), axis=0)
        fpr_tpr = fpr_tpr[:, indices]

        indices = np.concatenate(
            (
                [0],
                (indices + 1),
            ),
            axis=0,
        )
        fpr_tpr = np.concatenate(([[0], [0]], fpr_tpr), axis=1)

        tpr = np.zeros_like(fpr_tpr[1])
        np.maximum.accumulate(fpr_tpr[1], out=tpr)

        print()
        print(label_idx)
        print("fpr", fpr_tpr[0])
        print("tpr", tpr)
        print("rocauc", np.trapz(x=fpr_tpr[0], y=tpr, axis=0))

        rocauc[label_idx] = np.trapz(x=fpr_tpr[0], y=tpr, axis=0)

    # cumulative_fp = np.sum(
    #     np.cumsum(
    #         mask_pd_labels & ~mask_matching_labels[:, np.newaxis], axis=0
    #     ),
    #     axis=1,
    # )
    # cumulative_tp = np.sum(
    #     np.cumsum(
    #         mask_pd_labels & mask_matching_labels[:, np.newaxis], axis=0
    #     ),
    #     axis=1,
    # )

    # print(np.cumsum(
    #         mask_pd_labels & mask_matching_labels[:, np.newaxis], axis=0
    #     ))
    # print(cumulative_tp)

    # fpr_tpr = np.zeros((2, n_rows, n_labels), dtype=np.float64)
    # np.divide(
    #     cumulative_fp,
    #     negative_count,
    #     where=mask_pd_labels & (negative_count > 1e-9)[np.newaxis, :],
    #     out=fpr_tpr[0],
    # )
    # np.divide(
    #     cumulative_tp,
    #     positive_count,
    #     where=mask_pd_labels & (positive_count > 1e-9)[np.newaxis, :],
    #     out=fpr_tpr[1],
    # )

    # # sort by fpr, tpr
    # indices = np.lexsort([fpr_tpr[1], fpr_tpr[0]], axis=0)
    # fpr_tpr[0] = np.take_along_axis(fpr_tpr[0], indices, axis=0)
    # fpr_tpr[1] = np.take_along_axis(fpr_tpr[1], indices, axis=0)

    # # print()

    # print()
    # print(fpr_tpr[:, :, 3].transpose()[-6:])
    # print()
    # print(fpr_tpr[:, :, 4].transpose()[-6:])
    # print()
    # print(fpr_tpr[:, :, 5].transpose()[-6:])
    # print()
    # print(fpr_tpr[:, :, 6].transpose()[-6:])

    # compute rocauc
    # rocauc = np.trapz(x=fpr_tpr[0], y=fpr_tpr[1], axis=0)

    # compute mean rocauc
    summed_rocauc = np.bincount(label_metadata[:, 2], weights=rocauc)
    mean_rocauc = np.zeros(2, dtype=np.float64)
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
) -> tuple[
    NDArray[np.int32],
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
    """

    n_labels = label_metadata.shape[0]
    n_scores = score_thresholds.shape[0]

    counts = np.zeros((n_scores, n_labels, 4), dtype=np.int32)
    recall = np.zeros((n_scores, n_labels), dtype=np.float64)
    precision = np.zeros_like(recall)
    accuracy = np.zeros_like(recall)
    f1_score = np.zeros_like(recall)

    pd_labels = data[:, 2].astype(int)

    mask_gt_exists = data[:, 1] >= 0.0
    mask_pd_exists = data[:, 2] >= 0.0
    mask_matching_labels = np.isclose(data[:, 1], data[:, 2])
    mask_score_nonzero = ~np.isclose(data[:, 3], 0.0)

    # calculate reciever-operating-characteristic (ROC) curve
    rocauc = _compute_rocauc(
        pd_labels=pd_labels,
        mask_gt_exists=mask_gt_exists,
        mask_matching_labels=mask_matching_labels,
        n_labels=n_labels,
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
        rocauc,
    )


def compute_detailed_pr_curve() -> tuple:
    return tuple()
