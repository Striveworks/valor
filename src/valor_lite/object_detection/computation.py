from enum import IntFlag, auto

import numpy as np
import pyarrow as pa
import shapely
from numpy.typing import NDArray

EPSILON = 1e-9


def compute_bbox_iou(data: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Computes intersection-over-union (IOU) for axis-aligned bounding boxes.

    Takes data with shape (N, 8):

    Index 0 - xmin for Box 1
    Index 1 - xmax for Box 1
    Index 2 - ymin for Box 1
    Index 3 - ymax for Box 1
    Index 4 - xmin for Box 2
    Index 5 - xmax for Box 2
    Index 6 - ymin for Box 2
    Index 7 - ymax for Box 2

    Returns data with shape (N, 1):

    Index 0 - IOU

    Parameters
    ----------
    data : NDArray[np.float64]
        A sorted array of bounding box pairs.

    Returns
    -------
    NDArray[np.float64]
        Computed IOU's.
    """
    if data.size == 0:
        return np.array([], dtype=np.float64)

    n_pairs = data.shape[0]

    xmin1, xmax1, ymin1, ymax1 = (
        data[:, 0, 0],
        data[:, 0, 1],
        data[:, 0, 2],
        data[:, 0, 3],
    )
    xmin2, xmax2, ymin2, ymax2 = (
        data[:, 1, 0],
        data[:, 1, 1],
        data[:, 1, 2],
        data[:, 1, 3],
    )

    xmin = np.maximum(xmin1, xmin2)
    ymin = np.maximum(ymin1, ymin2)
    xmax = np.minimum(xmax1, xmax2)
    ymax = np.minimum(ymax1, ymax2)

    intersection_width = np.maximum(0, xmax - xmin)
    intersection_height = np.maximum(0, ymax - ymin)
    intersection_area = intersection_width * intersection_height

    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    union_area = area1 + area2 - intersection_area

    ious = np.zeros(n_pairs, dtype=np.float64)
    np.divide(
        intersection_area,
        union_area,
        where=union_area >= EPSILON,
        out=ious,
    )
    return ious


def compute_bitmask_iou(data: NDArray[np.bool_]) -> NDArray[np.float64]:
    """
    Computes intersection-over-union (IOU) for bitmasks.

    Takes data with shape (N, 2):

    Index 0 - first bitmask
    Index 1 - second bitmask

    Returns data with shape (N, 1):

    Index 0 - IOU

    Parameters
    ----------
    data : NDArray[np.float64]
        A sorted array of bitmask pairs.

    Returns
    -------
    NDArray[np.float64]
        Computed IOU's.
    """

    if data.size == 0:
        return np.array([], dtype=np.float64)

    n_pairs = data.shape[0]
    lhs = data[:, 0, :, :].reshape(n_pairs, -1)
    rhs = data[:, 1, :, :].reshape(n_pairs, -1)

    lhs_sum = lhs.sum(axis=1)
    rhs_sum = rhs.sum(axis=1)

    intersection_ = np.logical_and(lhs, rhs).sum(axis=1)
    union_ = lhs_sum + rhs_sum - intersection_

    ious = np.zeros(n_pairs, dtype=np.float64)
    np.divide(
        intersection_,
        union_,
        where=union_ >= EPSILON,
        out=ious,
    )
    return ious


def compute_polygon_iou(
    data: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Computes intersection-over-union (IOU) for shapely polygons.

    Takes data with shape (N, 2):

    Index 0 - first polygon
    Index 1 - second polygon

    Returns data with shape (N, 1):

    Index 0 - IOU

    Parameters
    ----------
    data : NDArray[np.float64]
        A sorted array of polygon pairs.

    Returns
    -------
    NDArray[np.float64]
        Computed IOU's.
    """

    if data.size == 0:
        return np.array([], dtype=np.float64)

    n_pairs = data.shape[0]

    lhs = data[:, 0]
    rhs = data[:, 1]

    intersections = shapely.intersection(lhs, rhs)
    intersection_areas = shapely.area(intersections)

    unions = shapely.union(lhs, rhs)
    union_areas = shapely.area(unions)

    ious = np.zeros(n_pairs, dtype=np.float64)
    np.divide(
        intersection_areas,
        union_areas,
        where=union_areas >= EPSILON,
        out=ious,
    )
    return ious


def rank_pairs(
    sorted_pairs: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    """
    Prunes and ranks prediction pairs.

    Should result in a single pair per prediction annotation.

    Parameters
    ----------
    sorted_pairs : NDArray[np.float64]
        Ranked annotation pairs.
        Index 0 - Datum Index
        Index 1 - GroundTruth Index
        Index 2 - Prediction Index
        Index 3 - GroundTruth Label Index
        Index 4 - Prediction Label Index
        Index 5 - IOU
        Index 6 - Score

    Returns
    -------
    NDArray[float64]
        Ranked prediction pairs.
    NDArray[intp]
        Indices of ranked prediction pairs.
    """

    # remove unmatched ground truths
    mask_predictions = sorted_pairs[:, 2] >= 0.0
    pairs = sorted_pairs[mask_predictions]
    indices = np.where(mask_predictions)[0]

    # find best fits for prediction
    mask_label_match = np.isclose(pairs[:, 3], pairs[:, 4])
    matched_predictions = np.unique(pairs[mask_label_match, 2])

    mask_unmatched_predictions = ~np.isin(pairs[:, 2], matched_predictions)

    pairs = pairs[mask_label_match | mask_unmatched_predictions]
    indices = indices[mask_label_match | mask_unmatched_predictions]

    # only keep the highest ranked prediction (datum_id, prediction_id, predicted_label_id)
    _, unique_indices = np.unique(
        pairs[:, [0, 2, 4]], axis=0, return_index=True
    )
    pairs = pairs[unique_indices]
    indices = indices[unique_indices]

    # np.unique orders its results by value, we need to sort the indices to maintain the results of the lexsort
    sorted_indices = np.lexsort(
        (
            -pairs[:, 5],  # iou
            -pairs[:, 6],  # score
        )
    )
    pairs = pairs[sorted_indices]
    indices = indices[sorted_indices]

    return pairs, indices


def calculate_ranking_boundaries(
    ranked_pairs: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Determine IOU boundaries for computing AP across chunks.

    Parameters
    ----------
    ranked_pairs : NDArray[np.float64]
        Ranked annotation pairs.
        Index 0 - Datum Index
        Index 1 - GroundTruth Index
        Index 2 - Prediction Index
        Index 3 - GroundTruth Label Index
        Index 4 - Prediction Label Index
        Index 5 - IOU
        Index 6 - Score

    Returns
    -------
    NDArray[np.float64]
        A 1-D array containing the lower IOU boundary for classifying pairs as true-positive across chunks.
    """
    ids = ranked_pairs[:, (0, 1, 2, 3, 4)].astype(np.int64)
    gts = ids[:, (0, 1, 3)]
    gt_labels = ids[:, 3]
    pd_labels = ids[:, 4]
    ious = ranked_pairs[:, 5]

    # set default boundary to 2.0 as it will be used to check lower boundary in range [0-1].
    iou_boundary = np.ones_like(ious) * 2

    mask_matching_labels = gt_labels == pd_labels
    mask_valid_gts = gts[:, 1] >= 0
    unique_gts = np.unique(gts[mask_valid_gts], axis=0)
    for gt in unique_gts:
        mask_gt = (gts == gt).all(axis=1)
        mask_gt &= mask_matching_labels
        if mask_gt.sum() <= 1:
            iou_boundary[mask_gt] = 0.0
            continue

        running_max = np.maximum.accumulate(ious[mask_gt])
        mask_rmax = np.isclose(running_max, ious[mask_gt])
        mask_rmax[1:] &= running_max[1:] > running_max[:-1]
        mask_gt[mask_gt] &= mask_rmax

        indices = np.where(mask_gt)[0]

        iou_boundary[indices[0]] = 0.0
        iou_boundary[indices[1:]] = ious[indices[:-1]]

    return iou_boundary


def rank_table(tbl: pa.Table) -> pa.Table:
    """Rank table for AP computation."""
    numeric_columns = [
        "datum_id",
        "gt_id",
        "pd_id",
        "gt_label_id",
        "pd_label_id",
        "iou",
        "pd_score",
    ]
    sorting_args = [
        ("pd_score", "descending"),
        ("iou", "descending"),
    ]

    # initial sort
    sorted_tbl = tbl.sort_by(sorting_args)
    pairs = np.column_stack(
        [sorted_tbl[col].to_numpy() for col in numeric_columns]
    )

    # rank pairs
    ranked_pairs, indices = rank_pairs(pairs)
    ranked_tbl = sorted_tbl.take(indices)

    # find boundaries
    lower_iou_bound = calculate_ranking_boundaries(ranked_pairs)
    ranked_tbl = ranked_tbl.append_column(
        pa.field("iou_prev", pa.float64()),
        pa.array(lower_iou_bound, type=pa.float64()),
    )

    return ranked_tbl


def compute_counts(
    ranked_pairs: NDArray[np.float64],
    iou_thresholds: NDArray[np.float64],
    score_thresholds: NDArray[np.float64],
    number_of_groundtruths_per_label: NDArray[np.uint64],
    number_of_labels: int,
    running_counts: NDArray[np.uint64],
    pr_curve: NDArray[np.float64],
) -> NDArray[np.uint64]:
    """
    Computes Object Detection metrics.

    Precision-recall curve and running counts are updated in-place.

    Parameters
    ----------
    ranked_pairs : NDArray[np.float64]
        A ranked array summarizing the IOU calculations of one or more pairs.
        Index 0 - Datum Index
        Index 1 - GroundTruth Index
        Index 2 - Prediction Index
        Index 3 - GroundTruth Label Index
        Index 4 - Prediction Label Index
        Index 5 - IOU
        Index 6 - Score
        Index 7 - IOU Lower Boundary
    iou_thresholds : NDArray[np.float64]
        A 1-D array containing IOU thresholds.
    score_thresholds : NDArray[np.float64]
        A 1-D array containing score thresholds.
    number_of_groundtruths_per_label : NDArray[np.uint64]
        A 1-D array containing total number of ground truths per label.
    number_of_labels : int
        Total number of unique labels.
    running_counts : NDArray[np.uint64]
        A 2-D array containing running counts of total predictions and true-positive. This array is mutated.
    pr_curve : NDArray[np.float64]
        A 2-D array containing 101-point binning of precision and score over a fixed recall interval. This array is mutated.

    Returns
    -------
    NDArray[uint64]
        Batched counts of TP, FP, FN.
    """
    n_rows = ranked_pairs.shape[0]
    n_labels = number_of_labels
    n_ious = iou_thresholds.shape[0]
    n_scores = score_thresholds.shape[0]

    # initialize result arrays
    counts = np.zeros((n_ious, n_scores, 3, n_labels), dtype=np.uint64)

    # start computation
    ids = ranked_pairs[:, :5].astype(np.int64)
    gt_ids = ids[:, 1]
    gt_labels = ids[:, 3]
    pd_labels = ids[:, 4]
    ious = ranked_pairs[:, 5]
    scores = ranked_pairs[:, 6]
    prev_ious = ranked_pairs[:, 7]

    unique_pd_labels, _ = np.unique(pd_labels, return_index=True)

    running_total_count = np.zeros(
        (n_ious, n_rows),
        dtype=np.uint64,
    )
    running_tp_count = np.zeros_like(running_total_count)
    running_gt_count = number_of_groundtruths_per_label[pd_labels]

    mask_score_nonzero = scores > EPSILON
    mask_gt_exists = gt_ids >= 0.0
    mask_labels_match = np.isclose(gt_labels, pd_labels)

    mask_gt_exists_labels_match = mask_gt_exists & mask_labels_match

    mask_tp = mask_score_nonzero & mask_gt_exists_labels_match
    mask_fp = mask_score_nonzero

    for iou_idx in range(n_ious):
        mask_iou_curr = ious >= iou_thresholds[iou_idx]
        mask_iou_prev = prev_ious < iou_thresholds[iou_idx]
        mask_iou = mask_iou_curr & mask_iou_prev

        mask_tp_outer = mask_tp & mask_iou
        mask_fp_outer = mask_fp & (
            (~mask_gt_exists_labels_match & mask_iou) | ~mask_iou
        )

        for score_idx in range(n_scores):
            mask_score_thresh = scores >= score_thresholds[score_idx]

            mask_tp_inner = mask_tp_outer & mask_score_thresh
            mask_fp_inner = mask_fp_outer & mask_score_thresh

            # create true-positive mask score threshold
            tp_candidates = ids[mask_tp_inner]
            _, indices_gt_unique = np.unique(
                tp_candidates[:, [0, 1, 3]], axis=0, return_index=True
            )
            mask_gt_unique = np.zeros(tp_candidates.shape[0], dtype=np.bool_)
            mask_gt_unique[indices_gt_unique] = True

            true_positives_mask = np.zeros(n_rows, dtype=np.bool_)
            true_positives_mask[mask_tp_inner] = mask_gt_unique

            mask_fp_inner |= mask_tp_inner & ~true_positives_mask

            # calculate intermediates
            counts[iou_idx, score_idx, 0, :] = np.bincount(
                pd_labels,
                weights=true_positives_mask,
                minlength=n_labels,
            )
            # fp count
            counts[iou_idx, score_idx, 1, :] = np.bincount(
                pd_labels[mask_fp_inner],
                minlength=n_labels,
            )

        # count running tp and total for AP
        for pd_label in unique_pd_labels:
            mask_pd_label = pd_labels == pd_label
            total_count = mask_pd_label.sum()
            if total_count == 0:
                continue

            # running total prediction count
            running_total_count[iou_idx, mask_pd_label] = np.arange(
                running_counts[iou_idx, pd_label, 0] + 1,
                running_counts[iou_idx, pd_label, 0] + total_count + 1,
            )
            running_counts[iou_idx, pd_label, 0] += total_count

            # running true-positive count
            mask_tp_for_counting = mask_pd_label & mask_tp_outer
            tp_count = mask_tp_for_counting.sum()
            running_tp_count[iou_idx, mask_tp_for_counting] = np.arange(
                running_counts[iou_idx, pd_label, 1] + 1,
                running_counts[iou_idx, pd_label, 1] + tp_count + 1,
            )
            running_counts[iou_idx, pd_label, 1] += tp_count

    # calculate running precision-recall points for AP
    precision = np.zeros_like(running_total_count, dtype=np.float64)
    np.divide(
        running_tp_count,
        running_total_count,
        where=running_total_count > 0,
        out=precision,
    )
    recall = np.zeros_like(running_total_count, dtype=np.float64)
    np.divide(
        running_tp_count,
        running_gt_count,
        where=running_gt_count > 0,
        out=recall,
    )
    recall_index = np.floor(recall * 100.0).astype(np.int32)

    # sort precision in descending order
    precision_indices = np.argsort(-precision, axis=1)

    # populate precision-recall curve
    for iou_idx in range(n_ious):
        labeled_recall = np.hstack(
            [
                pd_labels.reshape(-1, 1),
                recall_index[iou_idx, :].reshape(-1, 1),
            ]
        )

        # extract maximum score per (label, recall) bin
        # arrays are already ordered by descending score
        lr_pairs, recall_indices = np.unique(
            labeled_recall, return_index=True, axis=0
        )
        li = lr_pairs[:, 0]
        ri = lr_pairs[:, 1]
        pr_curve[iou_idx, li, ri, 1] = np.maximum(
            pr_curve[iou_idx, li, ri, 1],
            scores[recall_indices],
        )

        # extract maximum precision per (label, recall) bin
        # reorder arrays into descending precision order
        indices = precision_indices[iou_idx]
        sorted_precision = precision[iou_idx, indices]
        sorted_labeled_recall = labeled_recall[indices]
        lr_pairs, recall_indices = np.unique(
            sorted_labeled_recall, return_index=True, axis=0
        )
        li = lr_pairs[:, 0]
        ri = lr_pairs[:, 1]
        pr_curve[iou_idx, li, ri, 0] = np.maximum(
            pr_curve[iou_idx, li, ri, 0],
            sorted_precision[recall_indices],
        )

    return counts


def compute_precision_recall_f1(
    counts: NDArray[np.uint64],
    number_of_groundtruths_per_label: NDArray[np.uint64],
) -> NDArray[np.float64]:

    prec_rec_f1 = np.zeros_like(counts, dtype=np.float64)

    # alias
    tp_count = counts[:, :, 0, :]
    fp_count = counts[:, :, 1, :]
    tp_fp_count = tp_count + fp_count

    # calculate component metrics
    np.divide(
        tp_count,
        tp_fp_count,
        where=tp_fp_count > 0,
        out=prec_rec_f1[:, :, 0, :],
    )
    np.divide(
        tp_count,
        number_of_groundtruths_per_label,
        where=number_of_groundtruths_per_label > 0,
        out=prec_rec_f1[:, :, 1, :],
    )
    p = prec_rec_f1[:, :, 0, :]
    r = prec_rec_f1[:, :, 1, :]
    np.divide(
        2 * np.multiply(p, r),
        (p + r),
        where=(p + r) > EPSILON,
        out=prec_rec_f1[:, :, 2, :],
    )
    return prec_rec_f1


def compute_average_recall(prec_rec_f1: NDArray[np.float64]):
    recall = prec_rec_f1[:, :, 1, :]
    average_recall = recall.mean(axis=0)
    mAR = average_recall.mean(axis=-1)
    return average_recall, mAR


def compute_average_precision(pr_curve: NDArray[np.float64]):
    n_ious = pr_curve.shape[0]
    n_labels = pr_curve.shape[1]

    # initialize result arrays
    average_precision = np.zeros((n_ious, n_labels), dtype=np.float64)
    mAP = np.zeros(n_ious, dtype=np.float64)

    # calculate average precision
    running_max_precision = np.zeros((n_ious, n_labels), dtype=np.float64)
    running_max_score = np.zeros((n_labels), dtype=np.float64)
    for recall in range(100, -1, -1):

        # running max precision
        running_max_precision = np.maximum(
            pr_curve[:, :, recall, 0],
            running_max_precision,
        )
        pr_curve[:, :, recall, 0] = running_max_precision

        # running max score
        running_max_score = np.maximum(
            pr_curve[:, :, recall, 1],
            running_max_score,
        )
        pr_curve[:, :, recall, 1] = running_max_score

        average_precision += running_max_precision

    average_precision = average_precision / 101.0

    # calculate mAP and mAR
    if average_precision.size > 0:
        mAP = average_precision.mean(axis=1)

    return average_precision, mAP, pr_curve


def _isin(
    data: NDArray,
    subset: NDArray,
) -> NDArray[np.bool_]:
    """
    Creates a mask of rows that exist within the subset.

    Parameters
    ----------
    data : NDArray[np.int32]
        An array with shape (N, 2).
    subset : NDArray[np.int32]
        An array with shape (M, 2) where N >= M.

    Returns
    -------
    NDArray[np.bool_]
        Returns a bool mask with shape (N,).
    """
    combined_data = (data[:, 0].astype(np.int64) << 32) | data[:, 1].astype(
        np.int32
    )
    combined_subset = (subset[:, 0].astype(np.int64) << 32) | subset[
        :, 1
    ].astype(np.int32)
    mask = np.isin(combined_data, combined_subset, assume_unique=False)
    return mask


class PairClassification(IntFlag):
    NULL = auto()
    TP = auto()
    FP_FN_MISCLF = auto()
    FP_UNMATCHED = auto()
    FN_UNMATCHED = auto()


def mask_pairs_greedily(
    pairs: NDArray[np.float64],
):
    groundtruths = pairs[:, 1].astype(np.int32)
    predictions = pairs[:, 2].astype(np.int32)

    # Pre‑allocate "seen" flags for every possible x and y
    max_gt = groundtruths.max()
    max_pd = predictions.max()
    used_gt = np.zeros(max_gt + 1, dtype=np.bool_)
    used_pd = np.zeros(max_pd + 1, dtype=np.bool_)

    # This mask will mark which pairs to keep
    keep = np.zeros(pairs.shape[0], dtype=bool)

    for idx in range(groundtruths.shape[0]):
        gidx = groundtruths[idx]
        pidx = predictions[idx]

        if not (gidx < 0 or pidx < 0 or used_gt[gidx] or used_pd[pidx]):
            keep[idx] = True
            used_gt[gidx] = True
            used_pd[pidx] = True

    mask_matches = _isin(
        data=pairs[:, (1, 2)],
        subset=np.unique(pairs[np.ix_(keep, (1, 2))], axis=0),  # type: ignore - np.ix_ typing
    )

    return mask_matches


def compute_pair_classifications(
    detailed_pairs: NDArray[np.float64],
    iou_thresholds: NDArray[np.float64],
    score_thresholds: NDArray[np.float64],
) -> tuple[
    NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_]
]:
    """
    Compute detailed counts.

    Takes data with shape (N, 7):

    Index 0 - Datum Index
    Index 1 - GroundTruth Index
    Index 2 - Prediction Index
    Index 3 - GroundTruth Label Index
    Index 4 - Prediction Label Index
    Index 5 - IOU
    Index 6 - Score

    Parameters
    ----------
    detailed_pairs : NDArray[np.float64]
        An unsorted array summarizing the IOU calculations of one or more pairs.
    label_metadata : NDArray[np.int32]
        An array containing metadata related to labels.
    iou_thresholds : NDArray[np.float64]
        A 1-D array containing IOU thresholds.
    score_thresholds : NDArray[np.float64]
        A 1-D array containing score thresholds.

    Returns
    -------
    NDArray[np.uint8]
        Confusion matrix.
    """
    n_pairs = detailed_pairs.shape[0]
    n_ious = iou_thresholds.shape[0]
    n_scores = score_thresholds.shape[0]

    pair_classifications = np.zeros(
        (n_ious, n_scores, n_pairs),
        dtype=np.uint8,
    )

    ids = detailed_pairs[:, :5].astype(np.int32)
    groundtruths = ids[:, (0, 1)]
    predictions = ids[:, (0, 2)]
    gt_ids = ids[:, 1]
    pd_ids = ids[:, 2]
    gt_labels = ids[:, 3]
    pd_labels = ids[:, 4]
    ious = detailed_pairs[:, 5]
    scores = detailed_pairs[:, 6]

    mask_gt_exists = gt_ids > -0.5
    mask_pd_exists = pd_ids > -0.5
    mask_label_match = np.isclose(gt_labels, pd_labels)
    mask_score_nonzero = scores > EPSILON
    mask_iou_nonzero = ious > EPSILON

    mask_gt_pd_exists = mask_gt_exists & mask_pd_exists
    mask_gt_pd_match = mask_gt_pd_exists & mask_label_match

    mask_matched_pairs = mask_pairs_greedily(pairs=detailed_pairs)

    for iou_idx in range(n_ious):
        mask_iou_threshold = ious >= iou_thresholds[iou_idx]
        mask_iou = mask_iou_nonzero & mask_iou_threshold
        for score_idx in range(n_scores):
            mask_score_threshold = scores >= score_thresholds[score_idx]
            mask_score = mask_score_nonzero & mask_score_threshold

            mask_thresholded_matched_pairs = (
                mask_matched_pairs & mask_iou & mask_score
            )

            mask_true_positives = (
                mask_thresholded_matched_pairs & mask_gt_pd_match
            )
            mask_misclf = mask_thresholded_matched_pairs & ~mask_gt_pd_match

            mask_groundtruths_in_thresholded_matched_pairs = _isin(
                data=groundtruths,
                subset=np.unique(
                    groundtruths[mask_thresholded_matched_pairs], axis=0
                ),
            )
            mask_predictions_in_thresholded_matched_pairs = _isin(
                data=predictions,
                subset=np.unique(
                    predictions[mask_thresholded_matched_pairs], axis=0
                ),
            )

            mask_unmatched_predictions = (
                ~mask_predictions_in_thresholded_matched_pairs
                & mask_pd_exists
                & mask_score
            )
            mask_unmatched_groundtruths = (
                ~mask_groundtruths_in_thresholded_matched_pairs
                & mask_gt_exists
            )

            # classify pairings
            pair_classifications[
                iou_idx, score_idx, mask_true_positives
            ] |= np.uint8(PairClassification.TP)
            pair_classifications[iou_idx, score_idx, mask_misclf] |= np.uint8(
                PairClassification.FP_FN_MISCLF
            )
            pair_classifications[
                iou_idx, score_idx, mask_unmatched_predictions
            ] |= np.uint8(PairClassification.FP_UNMATCHED)
            pair_classifications[
                iou_idx, score_idx, mask_unmatched_groundtruths
            ] |= np.uint8(PairClassification.FN_UNMATCHED)

    mask_tp = np.bitwise_and(pair_classifications, PairClassification.TP) > 0
    mask_fp_fn_misclf = (
        np.bitwise_and(pair_classifications, PairClassification.FP_FN_MISCLF)
        > 0
    )
    mask_fp_unmatched = (
        np.bitwise_and(pair_classifications, PairClassification.FP_UNMATCHED)
        > 0
    )
    mask_fn_unmatched = (
        np.bitwise_and(pair_classifications, PairClassification.FN_UNMATCHED)
        > 0
    )

    return (
        mask_tp,
        mask_fp_fn_misclf,
        mask_fp_unmatched,
        mask_fn_unmatched,
    )


def compute_confusion_matrix(
    detailed_pairs: NDArray[np.float64],
    mask_tp: NDArray[np.bool_],
    mask_fp_fn_misclf: NDArray[np.bool_],
    mask_fp_unmatched: NDArray[np.bool_],
    mask_fn_unmatched: NDArray[np.bool_],
    number_of_labels: int,
    iou_thresholds: NDArray[np.float64],
    score_thresholds: NDArray[np.float64],
):
    n_ious = iou_thresholds.size
    n_scores = score_thresholds.size
    ids = detailed_pairs[:, :5].astype(np.int64)

    # initialize arrays
    confusion_matrices = np.zeros(
        (n_ious, n_scores, number_of_labels, number_of_labels), dtype=np.uint64
    )
    unmatched_groundtruths = np.zeros(
        (n_ious, n_scores, number_of_labels), dtype=np.uint64
    )
    unmatched_predictions = np.zeros_like(unmatched_groundtruths)

    mask_matched = mask_tp | mask_fp_fn_misclf
    for iou_idx in range(n_ious):
        for score_idx in range(n_scores):
            # matched annotations
            unique_pairs = np.unique(
                ids[np.ix_(mask_matched[iou_idx, score_idx], (0, 1, 2, 3, 4))],  # type: ignore - numpy ix_ typing
                axis=0,
            )
            unique_labels, unique_label_counts = np.unique(
                unique_pairs[:, (3, 4)], axis=0, return_counts=True
            )
            confusion_matrices[
                iou_idx, score_idx, unique_labels[:, 0], unique_labels[:, 1]
            ] = unique_label_counts

            # unmatched groundtruths
            unique_pairs = np.unique(
                ids[np.ix_(mask_fn_unmatched[iou_idx, score_idx], (0, 1, 3))],  # type: ignore - numpy ix_ typing
                axis=0,
            )
            unique_labels, unique_label_counts = np.unique(
                unique_pairs[:, 2], return_counts=True
            )
            unmatched_groundtruths[
                iou_idx, score_idx, unique_labels
            ] = unique_label_counts

            # unmatched predictions
            unique_pairs = np.unique(
                ids[np.ix_(mask_fp_unmatched[iou_idx, score_idx], (0, 2, 4))],  # type: ignore - numpy ix_ typing
                axis=0,
            )
            unique_labels, unique_label_counts = np.unique(
                unique_pairs[:, 2], return_counts=True
            )
            unmatched_predictions[
                iou_idx, score_idx, unique_labels
            ] = unique_label_counts

    return confusion_matrices, unmatched_groundtruths, unmatched_predictions
