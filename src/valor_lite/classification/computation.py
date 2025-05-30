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
    detailed_pairs : NDArray[np.int32]
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
    detailed_pairs: NDArray[np.float64],
    datum_mask: NDArray[np.bool_],
    valid_label_indices: NDArray[np.int32] | None,
    n_labels: int,
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    # filter by datum
    detailed_pairs = detailed_pairs[datum_mask].copy()

    n_rows = detailed_pairs.shape[0]
    mask_invalid_groundtruths = np.zeros(n_rows, dtype=np.bool_)
    mask_invalid_predictions = np.zeros_like(mask_invalid_groundtruths)

    # filter labels
    if valid_label_indices is not None:
        mask_invalid_groundtruths[
            ~np.isin(detailed_pairs[:, 1], valid_label_indices)
        ] = True
        mask_invalid_predictions[
            ~np.isin(detailed_pairs[:, 2], valid_label_indices)
        ] = True

    # filter cache
    if mask_invalid_groundtruths.any():
        invalid_groundtruth_indices = np.where(mask_invalid_groundtruths)[0]
        detailed_pairs[invalid_groundtruth_indices[:, None], 1] = np.array(
            [[-1.0]]
        )

    if mask_invalid_predictions.any():
        invalid_prediction_indices = np.where(mask_invalid_predictions)[0]
        detailed_pairs[
            invalid_prediction_indices[:, None], (2, 3, 4)
        ] = np.array([[-1.0, -1.0, -1.0]])

    # filter null pairs
    mask_null_pairs = np.all(
        np.isclose(
            detailed_pairs[:, 1:5],
            np.array([-1.0, -1.0, -1.0, -1.0]),
        ),
        axis=1,
    )
    detailed_pairs = detailed_pairs[~mask_null_pairs]

    detailed_pairs = np.unique(detailed_pairs, axis=0)
    indices = np.lexsort(
        (
            detailed_pairs[:, 1],  # ground truth
            detailed_pairs[:, 2],  # prediction
            -detailed_pairs[:, 3],  # score
        )
    )
    detailed_pairs = detailed_pairs[indices]
    label_metadata = compute_label_metadata(
        ids=detailed_pairs[:, :3].astype(np.int32),
        n_labels=n_labels,
    )
    return detailed_pairs, label_metadata


def _compute_rocauc(
    data: NDArray[np.float64],
    label_metadata: NDArray[np.int32],
    n_datums: int,
    n_labels: int,
    mask_matching_labels: NDArray[np.bool_],
    pd_labels: NDArray[np.int32],
) -> tuple[NDArray[np.float64], float]:
    """
    Compute ROCAUC and mean ROCAUC.
    """
    positive_count = label_metadata[:, 0]
    negative_count = label_metadata[:, 1] - label_metadata[:, 0]

    true_positives = np.zeros((n_labels, n_datums), dtype=np.int32)
    false_positives = np.zeros_like(true_positives)
    scores = np.zeros_like(true_positives, dtype=np.float64)

    for label_idx in range(n_labels):
        if label_metadata[label_idx, 1] == 0:
            continue

        mask_pds = pd_labels == label_idx
        true_positives[label_idx] = mask_matching_labels[mask_pds]
        false_positives[label_idx] = ~mask_matching_labels[mask_pds]
        scores[label_idx] = data[mask_pds, 3]

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
    rocauc = npc.trapezoid(x=fpr, y=tpr, axis=1)

    # compute mean rocauc
    mean_rocauc = rocauc.mean()

    return rocauc, mean_rocauc  # type: ignore[reportReturnType]


def compute_precision_recall_rocauc(
    detailed_pairs: NDArray[np.float64],
    label_metadata: NDArray[np.int32],
    score_thresholds: NDArray[np.float64],
    hardmax: bool,
    n_datums: int,
) -> tuple[
    NDArray[np.int32],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    float,
]:
    """
    Computes classification metrics.

    Parameters
    ----------
    detailed_pairs : NDArray[np.float64]
        A sorted array of classification pairs with shape (n_pairs, 5).
            Index 0 - Datum Index
            Index 1 - GroundTruth Label Index
            Index 2 - Prediction Label Index
            Index 3 - Score
            Index 4 - Hard-Max Score
    label_metadata : NDArray[np.int32]
        An array containing metadata related to labels with shape (n_labels, 2).
            Index 0 - GroundTruth Label Count
            Index 1 - Prediction Label Count
    score_thresholds : NDArray[np.float64]
        A 1-D array contains score thresholds to compute metrics over.
    hardmax : bool
        Option to only allow a single positive prediction.
    n_datums : int
        The number of datums being operated over.

    Returns
    -------
    NDArray[np.int32]
        TP, FP, FN, TN counts.
    NDArray[np.float64]
        Precision.
    NDArray[np.float64]
        Recall.
    NDArray[np.float64]
        Accuracy
    NDArray[np.float64]
        F1 Score
    NDArray[np.float64]
        ROCAUC.
    float
        mROCAUC.
    """

    n_labels = label_metadata.shape[0]
    n_scores = score_thresholds.shape[0]

    pd_labels = detailed_pairs[:, 2].astype(int)

    mask_matching_labels = np.isclose(
        detailed_pairs[:, 1], detailed_pairs[:, 2]
    )
    mask_score_nonzero = ~np.isclose(detailed_pairs[:, 3], 0.0)
    mask_hardmax = detailed_pairs[:, 4] > 0.5

    # calculate ROCAUC
    rocauc, mean_rocauc = _compute_rocauc(
        data=detailed_pairs,
        label_metadata=label_metadata,
        n_datums=n_datums,
        n_labels=n_labels,
        mask_matching_labels=mask_matching_labels,
        pd_labels=pd_labels,
    )

    # calculate metrics at various score thresholds
    counts = np.zeros((n_scores, n_labels, 4), dtype=np.int32)
    for score_idx in range(n_scores):
        mask_score_threshold = (
            detailed_pairs[:, 3] >= score_thresholds[score_idx]
        )
        mask_score = mask_score_nonzero & mask_score_threshold

        if hardmax:
            mask_score &= mask_hardmax

        mask_tp = mask_matching_labels & mask_score
        mask_fp = ~mask_matching_labels & mask_score
        mask_fn = (mask_matching_labels & ~mask_score) | mask_fp
        mask_tn = ~mask_matching_labels & ~mask_score

        fn = np.unique(detailed_pairs[mask_fn][:, [0, 1]].astype(int), axis=0)
        tn = np.unique(detailed_pairs[mask_tn][:, [0, 2]].astype(int), axis=0)

        counts[score_idx, :, 0] = np.bincount(
            pd_labels[mask_tp], minlength=n_labels
        )
        counts[score_idx, :, 1] = np.bincount(
            pd_labels[mask_fp], minlength=n_labels
        )
        counts[score_idx, :, 2] = np.bincount(fn[:, 1], minlength=n_labels)
        counts[score_idx, :, 3] = np.bincount(tn[:, 1], minlength=n_labels)

    recall = np.zeros((n_scores, n_labels), dtype=np.float64)
    np.divide(
        counts[:, :, 0],
        (counts[:, :, 0] + counts[:, :, 2]),
        where=(counts[:, :, 0] + counts[:, :, 2]) > 1e-9,
        out=recall,
    )

    precision = np.zeros_like(recall)
    np.divide(
        counts[:, :, 0],
        (counts[:, :, 0] + counts[:, :, 1]),
        where=(counts[:, :, 0] + counts[:, :, 1]) > 1e-9,
        out=precision,
    )

    accuracy = np.zeros(n_scores, dtype=np.float64)
    np.divide(
        counts[:, :, 0].sum(axis=1),
        float(n_datums),
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


class PairClassification(IntFlag):
    TP = auto()
    FP_FN_MISCLF = auto()
    FN_UNMATCHED = auto()


def compute_confusion_matrix(
    detailed_pairs: NDArray[np.float64],
    score_thresholds: NDArray[np.float64],
    hardmax: bool,
) -> NDArray[np.uint8]:
    """
    Compute detailed confusion matrix.

    Parameters
    ----------
    detailed_pairs : NDArray[np.float64]
        A 2-D sorted array summarizing the IOU calculations of one or more pairs with shape (n_pairs, 5).
            Index 0 - Datum Index
            Index 1 - GroundTruth Label Index
            Index 2 - Prediction Label Index
            Index 3 - Score
            Index 4 - Hard Max Score
    iou_thresholds : NDArray[np.float64]
        A 1-D array containing IOU thresholds.
    score_thresholds : NDArray[np.float64]
        A 1-D array containing score thresholds.

    Returns
    -------
    NDArray[uint8]
        Row-wise classification of pairs.
    """
    n_pairs = detailed_pairs.shape[0]
    n_scores = score_thresholds.shape[0]

    pair_classifications = np.zeros(
        (n_scores, n_pairs),
        dtype=np.uint8,
    )

    mask_label_match = np.isclose(detailed_pairs[:, 1], detailed_pairs[:, 2])
    mask_score = detailed_pairs[:, 3] > 1e-9

    groundtruths = detailed_pairs[:, [0, 1]].astype(int)

    for score_idx in range(n_scores):
        mask_score &= detailed_pairs[:, 3] >= score_thresholds[score_idx]
        if hardmax:
            mask_score &= detailed_pairs[:, 4] > 0.5

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

    return pair_classifications
