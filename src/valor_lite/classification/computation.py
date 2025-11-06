from enum import IntFlag

import numpy as np
from numpy.typing import NDArray

import valor_lite.classification.numpy_compatibility as npc


def compute_rocauc(
    rocauc: NDArray[np.float64],
    array: NDArray[np.float64],
    gt_count_per_label: NDArray[np.uint64],
    pd_count_per_label: NDArray[np.uint64],
    n_labels: int,
    accumulated_tp: NDArray[np.uint64],
    accumulated_fp: NDArray[np.uint64],
) -> tuple[NDArray[np.float64], NDArray[np.uint64], NDArray[np.uint64]]:
    """
    Compute ROCAUC.

    Parameters
    ----------
    ids : NDArray[np.int64]
        A sorted array of classification pairs with shape (n_pairs, 3).
            Index 0 - Datum Index
            Index 1 - GroundTruth Label Index
            Index 2 - Prediction Label Index
    scores : NDArray[np.float64]
        A sorted array of classification scores with shape (n_pairs,).
    gt_count_per_label : NDArray[np.uint64]
        The number of ground truth occurences per label.
    pd_count_per_label : NDArray[np.uint64]
        The number of prediction occurences per label.
    n_datums : int
        The number of datums being operated over.
    n_labels : int
        The total number of unqiue labels.

    Returns
    -------
    NDArray[np.float64]
        ROCAUC.
    NDArray[np.uint64]
        Final cumulative sum for FP's. Used as intermediate in chunking operations.
    NDArray[np.uint64]
        Final cumulative sum for TP's. Used as intermediate in chunking operations.
    """
    pd_labels = array[:, 0].astype(np.int64)
    scores = array[:, 1]
    mask_matching_labels = array[:, 2] > 0.5

    positive_count = gt_count_per_label
    negative_count = pd_count_per_label - gt_count_per_label

    print()
    for label_idx in range(n_labels):
        mask_pds = pd_labels == label_idx
        n_masked_pds = mask_pds.sum()
        if pd_count_per_label[label_idx] == 0 or n_masked_pds == 0:
            continue

        true_positives = mask_matching_labels[mask_pds]
        tp_scores = scores[mask_pds]

        distinct_score_indices = np.where(np.diff(tp_scores))[0]
        indices = np.r_[distinct_score_indices, n_masked_pds - 1]
        cumulative_tp = np.cumsum(true_positives, dtype=np.uint64)[indices]
        cumulative_fp = indices + 1 - cumulative_tp

        cumulative_tp += accumulated_tp[label_idx]
        cumulative_fp += accumulated_fp[label_idx]

        cumulative_tp = np.concatenate([accumulated_tp[label_idx:label_idx+1], cumulative_tp])
        cumulative_fp = np.concatenate([accumulated_fp[label_idx:label_idx+1], cumulative_fp])

        accumulated_tp[label_idx] = cumulative_tp[-1]
        accumulated_fp[label_idx] = cumulative_fp[-1]

        fpr = np.zeros_like(cumulative_fp, dtype=np.float64)
        np.divide(
            cumulative_fp,
            negative_count[label_idx],
            where=negative_count[label_idx] > 0,
            out=fpr,
        )
        tpr = np.zeros_like(cumulative_tp, dtype=np.float64)
        np.divide(
            cumulative_tp,
            positive_count[label_idx],
            where=positive_count[label_idx] > 0,
            out=tpr,
        )

        # # sort by -tpr, -score
        # indices = np.lexsort((-tpr, -tp_scores))
        # fpr = fpr[indices]
        # tpr = tpr[indices]

        # running max of tpr
        np.maximum.accumulate(tpr, out=tpr)

        # compute rocauc
        rocauc[label_idx] += npc.trapezoid(x=fpr, y=tpr, axis=0)

        if label_idx == 3:
            print(rocauc[label_idx])
            print(
                f"{'CFP':4}",
                f"{'CTP':4}",
                f"{'FPR':4}",
                f"{'TPR':4}",
                # f"{'SCO':4}",
            )
            for af, at, fr, tr in zip(
                cumulative_fp,
                cumulative_tp,
                fpr, 
                tpr, 
                # tp_scores,
            ):
                print(
                    f"{af:.2f}", 
                    f"{at:.2f}", 
                    f"{fr:.2f}", 
                    f"{tr:.2f}", 
                    # f"{s:.2f}",
                )
            
    return rocauc, accumulated_fp, accumulated_tp


def compute_counts(
    ids: NDArray[np.int64],
    scores: NDArray[np.float64],
    winners: NDArray[np.bool_],
    score_thresholds: NDArray[np.float64],
    hardmax: bool,
    n_labels: int,
) -> NDArray[np.uint64]:
    """
    Computes counts of TP, FP and FN's per label.

    Parameters
    ----------
    ids : NDArray[np.int64]
        A sorted array of classification pairs with shape (n_pairs, 3).
            Index 0 - Datum Index
            Index 1 - GroundTruth Label Index
            Index 2 - Prediction Label Index
    scores : NDArray[np.float64]
        A sorted array of classification scores with shape (n_pairs,).
    winner : NDArray[np.bool_]
        Marks predictions with highest score over a datum.
    score_thresholds : NDArray[np.float64]
        A 1-D array contains score thresholds to compute metrics over.
    hardmax : bool
        Option to only allow a single positive prediction.
    n_labels : int
        The total number of unqiue labels.

    Returns
    -------
    NDArray[np.int32]
        TP, FP, FN, TN counts.
    """
    n_scores = score_thresholds.shape[0]
    counts = np.zeros((n_scores, n_labels, 4), dtype=np.uint64)
    if ids.size == 0:
        return counts

    gt_labels = ids[:, 1]
    pd_labels = ids[:, 2]

    mask_matching_labels = np.isclose(gt_labels, pd_labels)
    mask_score_nonzero = ~np.isclose(scores, 0.0)
    mask_hardmax = winners > 0.5
    mask_valid_gts = gt_labels >= 0
    mask_valid_pds = pd_labels >= 0

    # calculate metrics at various score thresholds
    for score_idx in range(n_scores):
        mask_score_threshold = scores >= score_thresholds[score_idx]
        mask_score = mask_score_nonzero & mask_score_threshold

        if hardmax:
            mask_score &= mask_hardmax

        mask_tp = mask_matching_labels & mask_score
        mask_fp = ~mask_matching_labels & mask_score
        mask_fn = (mask_matching_labels & ~mask_score) | mask_fp
        mask_tn = ~mask_matching_labels & ~mask_score

        mask_fn &= mask_valid_gts
        mask_fp &= mask_valid_pds

        fn = np.unique(ids[mask_fn][:, [0, 1]].astype(int), axis=0)
        tn = np.unique(ids[mask_tn][:, [0, 2]].astype(int), axis=0)

        counts[score_idx, :, 0] = np.bincount(
            pd_labels[mask_tp], minlength=n_labels
        )
        counts[score_idx, :, 1] = np.bincount(
            pd_labels[mask_fp], minlength=n_labels
        )
        counts[score_idx, :, 2] = np.bincount(fn[:, 1], minlength=n_labels)
        counts[score_idx, :, 3] = np.bincount(tn[:, 1], minlength=n_labels)

    return counts


def compute_precision(counts: NDArray[np.uint64]) -> NDArray[np.float64]:
    """
    Compute precision metric using result of compute_counts.
    """
    n_scores, n_labels, _ = counts.shape
    precision = np.zeros((n_scores, n_labels), dtype=np.float64)
    np.divide(
        counts[:, :, 0],
        (counts[:, :, 0] + counts[:, :, 1]),
        where=(counts[:, :, 0] + counts[:, :, 1]) > 0,
        out=precision,
    )
    return precision


def compute_recall(counts: NDArray[np.uint64]) -> NDArray[np.float64]:
    """
    Compute recall metric using result of compute_counts.
    """
    n_scores, n_labels, _ = counts.shape
    recall = np.zeros((n_scores, n_labels), dtype=np.float64)
    np.divide(
        counts[:, :, 0],
        (counts[:, :, 0] + counts[:, :, 2]),
        where=(counts[:, :, 0] + counts[:, :, 2]) > 0,
        out=recall,
    )
    return recall


def compute_f1_score(
    precision: NDArray[np.float64], recall: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Compute f1 metric using result of compute_precision and compute_recall.
    """
    f1_score = np.zeros_like(recall)
    np.divide(
        (2 * precision * recall),
        (precision + recall),
        where=(precision + recall) > 1e-9,
        out=f1_score,
    )
    return f1_score


def compute_accuracy(
    counts: NDArray[np.uint64], n_datums: int
) -> NDArray[np.float64]:
    """
    Compute accuracy metric using result of compute_counts.
    """
    n_scores, _, _ = counts.shape
    accuracy = np.zeros(n_scores, dtype=np.float64)
    if n_datums == 0:
        return accuracy
    np.divide(
        counts[:, :, 0].sum(axis=1),
        n_datums,
        out=accuracy,
    )
    return accuracy


class PairClassification(IntFlag):
    TP = 1 << 0
    FP_FN_MISCLF = 1 << 1
    FN_UNMATCHED = 1 << 2


def compute_pair_classifications(
    ids: NDArray[np.int64],
    scores: NDArray[np.float64],
    winners: NDArray[np.bool_],
    score_thresholds: NDArray[np.float64],
    hardmax: bool,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_]]:
    """
    Classifiy ID pairs as TP, FP or FN.

    Parameters
    ----------
    ids : NDArray[np.int64]
        A sorted array of classification pairs with shape (n_pairs, 3).
            Index 0 - Datum Index
            Index 1 - GroundTruth Label Index
            Index 2 - Prediction Label Index
    scores : NDArray[np.float64]
        A sorted array of classification scores with shape (n_pairs,).
    winner : NDArray[np.bool_]
        Marks predictions with highest score over a datum.
    score_thresholds : NDArray[np.float64]
        A 1-D array containing score thresholds.
    hardmax : bool
        Option to only allow a single positive prediction.

    Returns
    -------
    NDArray[bool]
        True-positive mask.
    NDArray[bool]
        Misclassification FP, FN mask.
    NDArray[bool]
        Unmatched FN mask.
    """
    n_pairs = ids.shape[0]
    n_scores = score_thresholds.shape[0]

    gt_labels = ids[:, 1]
    pd_labels = ids[:, 2]
    groundtruths = ids[:, [0, 1]]

    pair_classifications = np.zeros(
        (n_scores, n_pairs),
        dtype=np.uint8,
    )

    mask_label_match = np.isclose(gt_labels, pd_labels)
    mask_score = scores > 1e-9
    for score_idx in range(n_scores):
        mask_score &= scores >= score_thresholds[score_idx]
        if hardmax:
            mask_score &= winners

        mask_true_positives = mask_label_match & mask_score
        mask_misclassifications = ~mask_label_match & mask_score
        mask_unmatched_groundtruths = ~(
            (
                groundtruths.reshape(-1, 1, 2)
                == groundtruths[mask_score].reshape(1, -1, 2)
            )
            .all(axis=2)
            .any(axis=1)
        )

        # classify pairings
        pair_classifications[score_idx, mask_true_positives] |= np.uint8(
            PairClassification.TP
        )
        pair_classifications[score_idx, mask_misclassifications] |= np.uint8(
            PairClassification.FP_FN_MISCLF
        )
        pair_classifications[
            score_idx, mask_unmatched_groundtruths
        ] |= np.uint8(PairClassification.FN_UNMATCHED)

    mask_tp = np.bitwise_and(pair_classifications, PairClassification.TP) > 0
    mask_fp_fn_misclf = (
        np.bitwise_and(pair_classifications, PairClassification.FP_FN_MISCLF)
        > 0
    )
    mask_fn_unmatched = (
        np.bitwise_and(pair_classifications, PairClassification.FN_UNMATCHED)
        > 0
    )

    return (
        mask_tp,
        mask_fp_fn_misclf,
        mask_fn_unmatched,
    )


def compute_confusion_matrix(
    ids: NDArray[np.int64],
    mask_tp: NDArray[np.bool_],
    mask_fp_fn_misclf: NDArray[np.bool_],
    mask_fn_unmatched: NDArray[np.bool_],
    score_thresholds: NDArray[np.float64],
    n_labels: int,
):
    """
    Compute confusion matrix using output of compute_pair_classifications.
    """
    n_scores = score_thresholds.size

    # initialize arrays
    confusion_matrices = np.zeros(
        (n_scores, n_labels, n_labels), dtype=np.uint64
    )
    unmatched_groundtruths = np.zeros((n_scores, n_labels), dtype=np.uint64)

    mask_matched = mask_tp | mask_fp_fn_misclf
    for score_idx in range(n_scores):
        # matched annotations
        unique_pairs = np.unique(
            ids[np.ix_(mask_matched[score_idx], (0, 1, 2))],  # type: ignore - numpy ix_ typing
            axis=0,
        )
        unique_labels, unique_label_counts = np.unique(
            unique_pairs[:, (1, 2)], axis=0, return_counts=True
        )
        confusion_matrices[
            score_idx, unique_labels[:, 0], unique_labels[:, 1]
        ] = unique_label_counts

        # unmatched groundtruths
        unique_pairs = np.unique(
            ids[np.ix_(mask_fn_unmatched[score_idx], (0, 1))],  # type: ignore - numpy ix_ typing
            axis=0,
        )
        unique_labels, unique_label_counts = np.unique(
            unique_pairs[:, 1], return_counts=True
        )
        unmatched_groundtruths[score_idx, unique_labels] = unique_label_counts

    return confusion_matrices, unmatched_groundtruths
