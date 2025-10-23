from enum import IntFlag, auto

import numpy as np
from numpy.typing import NDArray

import valor_lite.classification.numpy_compatibility as npc


def compute_label_metadata(
    ids: NDArray[np.int32],
    n_labels: int,
) -> NDArray[np.int32]:
    """
    Computes label metadata returning a count of annotations per label.

    Parameters
    ----------
    pairs : NDArray[np.int32]
        Detailed annotation pairings with shape (n_pairs, 3).
            Index 0 - Datum Index
            Index 1 - GroundTruth Label Index
            Index 2 - Prediction Label Index
    n_labels : int
        The total number of unique labels.

    Returns
    -------
    NDArray[np.int32]
        The label metadata array with shape (n_labels, 2).
            Index 0 - Ground truth label count
            Index 1 - Prediction label count
    """
    label_metadata = np.zeros((n_labels, 2), dtype=np.int32)
    ground_truth_pairs = ids[:, (0, 1)]
    ground_truth_pairs = ground_truth_pairs[ground_truth_pairs[:, 1] >= 0]
    unique_pairs = np.unique(ground_truth_pairs, axis=0)
    label_indices, unique_counts = np.unique(
        unique_pairs[:, 1], return_counts=True
    )
    label_metadata[label_indices.astype(np.int32), 0] = unique_counts

    prediction_pairs = ids[:, (0, 2)]
    prediction_pairs = prediction_pairs[prediction_pairs[:, 1] >= 0]
    unique_pairs = np.unique(prediction_pairs, axis=0)
    label_indices, unique_counts = np.unique(
        unique_pairs[:, 1], return_counts=True
    )
    label_metadata[label_indices.astype(np.int32), 1] = unique_counts

    return label_metadata


def filter_cache(
    pairs: NDArray[np.float64],
    datum_mask: NDArray[np.bool_],
    valid_label_indices: NDArray[np.int32] | None,
    n_labels: int,
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    # filter by datum
    pairs = pairs[datum_mask].copy()

    n_rows = pairs.shape[0]
    mask_invalid_groundtruths = np.zeros(n_rows, dtype=np.bool_)
    mask_invalid_predictions = np.zeros_like(mask_invalid_groundtruths)

    # filter labels
    if valid_label_indices is not None:
        mask_invalid_groundtruths[
            ~np.isin(pairs[:, 1], valid_label_indices)
        ] = True
        mask_invalid_predictions[
            ~np.isin(pairs[:, 2], valid_label_indices)
        ] = True

    # filter cache
    if mask_invalid_groundtruths.any():
        invalid_groundtruth_indices = np.where(mask_invalid_groundtruths)[0]
        pairs[invalid_groundtruth_indices[:, None], 1] = np.array([[-1.0]])

    if mask_invalid_predictions.any():
        invalid_prediction_indices = np.where(mask_invalid_predictions)[0]
        pairs[invalid_prediction_indices[:, None], (2, 3, 4)] = np.array(
            [[-1.0, -1.0, -1.0]]
        )

    # filter null pairs
    mask_null_pairs = np.all(
        np.isclose(
            pairs[:, 1:5],
            np.array([-1.0, -1.0, -1.0, -1.0]),
        ),
        axis=1,
    )
    pairs = pairs[~mask_null_pairs]

    pairs = np.unique(pairs, axis=0)
    indices = np.lexsort(
        (
            pairs[:, 1],  # ground truth
            pairs[:, 2],  # prediction
            -pairs[:, 3],  # score
        )
    )
    pairs = pairs[indices]
    label_metadata = compute_label_metadata(
        ids=pairs[:, :3].astype(np.int32),
        n_labels=n_labels,
    )
    return pairs, label_metadata


def compute_rocauc(
    ids: NDArray[np.int64],
    scores: NDArray[np.float64],
    gt_count_per_label: NDArray[np.uint64],
    pd_count_per_label: NDArray[np.uint64],
    n_datums: int,
    n_labels: int,
    prev_cumulative_fp: NDArray[np.uint64],
    prev_cumulative_tp: NDArray[np.uint64],
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
    prev_cumulative_fp : NDArray[np.uint64]
        Previous cumulative FP sum. Used in chunked computations.
    prev_cumulative_tp : NDArray[np.uint64]
        Previous cumulative TP sum. Used in chunked computations.

    Returns
    -------
    NDArray[np.float64]
        ROCAUC.
    NDArray[np.uint64]
        Final cumulative sum for FP's. Used as intermediate in chunking operations.
    NDArray[np.uint64]
        Final cumulative sum for TP's. Used as intermediate in chunking operations.
    """
    gt_labels = ids[:, 1]
    pd_labels = ids[:, 2]
    mask_matching_labels = np.isclose(gt_labels, pd_labels)

    positive_count = gt_count_per_label
    negative_count = pd_count_per_label - gt_count_per_label

    true_positives = np.zeros((n_labels, n_datums), dtype=np.uint64)
    false_positives = np.zeros_like(true_positives)
    tp_scores = np.zeros_like(true_positives, dtype=np.float64)

    for label_idx in range(n_labels):
        if pd_count_per_label[label_idx] == 0:
            continue

        mask_pds = pd_labels == label_idx
        true_positives[label_idx] = mask_matching_labels[mask_pds]
        false_positives[label_idx] = ~mask_matching_labels[mask_pds]
        tp_scores[label_idx] = scores[mask_pds]

    cumulative_fp = np.cumsum(false_positives, axis=1) + prev_cumulative_fp
    cumulative_tp = np.cumsum(true_positives, axis=1) + prev_cumulative_tp

    fpr = np.zeros_like(true_positives, dtype=np.float64)
    np.divide(
        cumulative_fp,
        negative_count[:, np.newaxis],
        where=negative_count[:, np.newaxis] > 0,
        out=fpr,
    )
    tpr = np.zeros_like(true_positives, dtype=np.float64)
    np.divide(
        cumulative_tp,
        positive_count[:, np.newaxis],
        where=positive_count[:, np.newaxis] > 0,
        out=tpr,
    )

    # sort by -tpr, -score
    indices = np.lexsort((-tpr, -tp_scores), axis=1)
    fpr = np.take_along_axis(fpr, indices, axis=1)
    tpr = np.take_along_axis(tpr, indices, axis=1)

    # running max of tpr
    np.maximum.accumulate(tpr, axis=1, out=tpr)

    # compute rocauc
    rocauc = npc.trapezoid(x=fpr, y=tpr, axis=1)

    return rocauc, cumulative_fp[-1], cumulative_tp[-1]


def compute_counts(
    ids: NDArray[np.uint64],
    scores: NDArray[np.float64],
    winners: NDArray[np.bool_],
    score_thresholds: NDArray[np.float64],
    hardmax: bool,
    n_labels: int,
) -> NDArray[np.uint64]:
    """
    Computes classification metrics.

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
    gt_labels = ids[:, 1]
    pd_labels = ids[:, 2]

    mask_matching_labels = np.isclose(gt_labels, pd_labels)
    mask_score_nonzero = ~np.isclose(scores, 0.0)
    mask_hardmax = winners > 0.5

    # calculate metrics at various score thresholds
    counts = np.zeros((n_scores, n_labels, 4), dtype=np.uint64)
    for score_idx in range(n_scores):
        mask_score_threshold = scores >= score_thresholds[score_idx]
        mask_score = mask_score_nonzero & mask_score_threshold

        if hardmax:
            mask_score &= mask_hardmax

        mask_tp = mask_matching_labels & mask_score
        mask_fp = ~mask_matching_labels & mask_score
        mask_fn = (mask_matching_labels & ~mask_score) | mask_fp
        mask_tn = ~mask_matching_labels & ~mask_score

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
    TP = auto()
    FP_FN_MISCLF = auto()
    FN_UNMATCHED = auto()


def compute_pair_classifications(
    ids: NDArray[np.int64],
    scores: NDArray[np.float64],
    winners: NDArray[np.bool_],
    score_thresholds: NDArray[np.float64],
    hardmax: bool,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_]]:
    """
    Compute detailed confusion matrix.

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
    NDArray[uint8]
        Row-wise classification of pairs.
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
    ids: NDArray[np.uint64],
    mask_tp: NDArray[np.bool_],
    mask_fp_fn_misclf: NDArray[np.bool_],
    mask_fn_unmatched: NDArray[np.bool_],
    score_thresholds: NDArray[np.float64],
    n_labels: int,
):
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
            unique_pairs[:, 2], return_counts=True
        )
        unmatched_groundtruths[score_idx, unique_labels] = unique_label_counts

    return confusion_matrices, unmatched_groundtruths
