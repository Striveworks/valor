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


def _encode_keys(*columns: NDArray[np.int64]) -> NDArray:
    """Encode integer tuples without collisions or integer overflow.

    Small key ranges use mixed-radix integers, avoiding NumPy's slower
    structured row sorting. Wide ranges use fixed-width records instead.
    Only key equality is used by callers; encoded ordering is private.
    """
    if not len(columns[0]):
        return np.empty(0, dtype=np.int64)

    bounds = [(int(col.min()), int(col.max())) for col in columns]
    cardinality = 1
    for lower, upper in bounds:
        cardinality *= upper - lower + 1
        if cardinality > np.iinfo(np.int64).max:
            rows = np.column_stack(columns)
            return rows.view(
                np.dtype((np.void, rows.dtype.itemsize * len(columns)))
            ).ravel()

    keys = columns[0] - bounds[0][0]
    for column, (lower, upper) in zip(columns[1:], bounds[1:]):
        keys *= upper - lower + 1
        keys += column - lower
    return keys


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
    n_ious = len(iou_thresholds)
    n_scores = len(score_thresholds)
    counts = np.zeros((n_ious, n_scores, 3, n_labels), dtype=np.uint64)
    if n_rows == 0:
        return counts

    ids = ranked_pairs[:, :5].astype(np.int64)
    pd_labels = ids[:, 4]
    scores = ranked_pairs[:, 6]
    mask_score_nonzero = scores > EPSILON
    mask_gt_labels_match = (ids[:, 1] >= 0) & np.isclose(ids[:, 3], pd_labels)

    # Group each label once, keeping the incoming score order within groups.
    label_order = np.argsort(pd_labels, kind="stable")
    sorted_labels = pd_labels[label_order]
    group_starts = np.r_[0, np.flatnonzero(np.diff(sorted_labels)) + 1]
    group_sizes = np.diff(np.r_[group_starts, n_rows])
    prediction_ranks = np.empty(n_rows, dtype=np.uint64)
    prediction_ranks[label_order] = (
        np.arange(n_rows) - np.repeat(group_starts, group_sizes) + 1
    )
    label_totals = np.bincount(pd_labels, minlength=n_labels).astype(np.uint64)

    # A row contributes to every score threshold <= its score. Histogram
    # suffix sums compute all thresholds together, including ties and repeats.
    score_order = np.argsort(score_thresholds, kind="stable")
    score_bins = np.searchsorted(
        score_thresholds[score_order], scores, side="right"
    )
    histogram_keys = score_bins * n_labels + pd_labels
    gt_keys = _encode_keys(ids[:, 0], ids[:, 1], ids[:, 3])
    groundtruth_counts = number_of_groundtruths_per_label[pd_labels]

    for iou_idx, threshold in enumerate(iou_thresholds):
        mask_tp_outer = (
            mask_score_nonzero
            & mask_gt_labels_match
            & (ranked_pairs[:, 5] >= threshold)
            & (ranked_pairs[:, 7] < threshold)
        )

        # Ranked rows are in descending score order. The first candidate for
        # a ground truth also remains first at every eligible score threshold.
        candidates = np.flatnonzero(mask_tp_outer)
        _, first = np.unique(gt_keys[candidates], return_index=True)
        mask_tp = np.zeros(n_rows, dtype=np.bool_)
        mask_tp[candidates[first]] = True
        mask_fp = mask_score_nonzero & ~mask_tp

        for count_idx, mask in enumerate((mask_tp, mask_fp)):
            histogram = np.bincount(
                histogram_keys[mask], minlength=(n_scores + 1) * n_labels
            ).reshape(n_scores + 1, n_labels)
            cumulative = np.cumsum(histogram[::-1], axis=0)[::-1][1:]
            counts[iou_idx, score_order, count_idx, :] = cumulative

        running_total = (
            prediction_ranks + running_counts[iou_idx, pd_labels, 0]
        )
        sorted_tp = mask_tp_outer[label_order]
        cumulative_tp = np.cumsum(sorted_tp, dtype=np.uint64)
        group_offsets = cumulative_tp[group_starts] - sorted_tp[
            group_starts
        ].astype(np.uint64)
        group_tp = cumulative_tp - np.repeat(group_offsets, group_sizes)
        running_tp = np.zeros(n_rows, dtype=np.uint64)
        running_tp[label_order] = group_tp
        running_tp = np.where(
            mask_tp_outer,
            running_tp + running_counts[iou_idx, pd_labels, 1],
            0,
        )
        running_counts[iou_idx, :, 0] += label_totals
        running_counts[iou_idx, :, 1] += np.bincount(
            pd_labels[mask_tp_outer], minlength=n_labels
        ).astype(np.uint64)

        precision = np.zeros(n_rows, dtype=np.float64)
        recall = np.zeros_like(precision)
        np.divide(
            running_tp, running_total, out=precision, where=running_total > 0
        )
        np.divide(
            running_tp,
            groundtruth_counts,
            out=recall,
            where=groundtruth_counts > 0,
        )
        recall_index = np.floor(recall * 100).astype(np.int32)
        recall_bins = pd_labels * 101 + recall_index

        # Reduce directly into the 101 recall bins instead of sorting all
        # rows twice to find their maximum precision and score.
        precision_bins = np.zeros(n_labels * 101, dtype=np.float64)
        score_values = np.zeros_like(precision_bins)
        np.maximum.at(precision_bins, recall_bins, precision)
        if np.isnan(scores).any():
            # Preserve the first score in each bin when NaNs are present;
            # a maximum would propagate NaNs from later rows in the bin.
            first_scores = np.full(n_labels * 101, n_rows, dtype=np.int64)
            np.minimum.at(first_scores, recall_bins, np.arange(n_rows))
            populated = first_scores < n_rows
            score_values[populated] = scores[first_scores[populated]]
        else:
            np.maximum.at(score_values, recall_bins, scores)
        pr_curve[iou_idx, :, :, 0] = np.maximum(
            pr_curve[iou_idx, :, :, 0], precision_bins.reshape(n_labels, 101)
        )
        pr_curve[iou_idx, :, :, 1] = np.maximum(
            pr_curve[iou_idx, :, :, 1], score_values.reshape(n_labels, 101)
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
            unmatched_groundtruths[iou_idx, score_idx, unique_labels] = (
                unique_label_counts
            )

            # unmatched predictions
            unique_pairs = np.unique(
                ids[np.ix_(mask_fp_unmatched[iou_idx, score_idx], (0, 2, 4))],  # type: ignore - numpy ix_ typing
                axis=0,
            )
            unique_labels, unique_label_counts = np.unique(
                unique_pairs[:, 2], return_counts=True
            )
            unmatched_predictions[iou_idx, score_idx, unique_labels] = (
                unique_label_counts
            )

    return confusion_matrices, unmatched_groundtruths, unmatched_predictions


def compute_confusion_matrix_from_table(
    table: pa.Table,
    number_of_labels: int,
    iou_thresholds: NDArray[np.float64],
    score_thresholds: NDArray[np.float64],
) -> tuple[NDArray[np.uint64], NDArray[np.uint64], NDArray[np.uint64]]:
    """Compute counts from compact pairs instead of masks over every row.

    Annotation IDs are globally unique and ground truths have one label, as
    assigned by Loader. Each prediction label has the same score wherever
    that annotation appears. Matching preserves the table's incoming order
    and happens before applying thresholds, as in compute_pair_classifications.
    """
    n_labels = number_of_labels
    shape = (len(iou_thresholds), len(score_thresholds))
    matrices = np.zeros((*shape, n_labels, n_labels), dtype=np.uint64)
    unmatched_gt = np.zeros((*shape, n_labels), dtype=np.uint64)
    unmatched_pd = np.zeros_like(unmatched_gt)
    if not table.num_rows:
        return matrices, unmatched_gt, unmatched_pd

    gt_ids, pd_ids, gt_labels, pd_labels, ious, scores = [
        table[name].to_numpy()
        for name in (
            "gt_id",
            "pd_id",
            "gt_label_id",
            "pd_label_id",
            "iou",
            "pd_score",
        )
    ]
    # Loader uses -1 for missing annotations, so the common case can pack
    # pairs directly. Keep the generic fallback for IDs too large to pack.
    pd_stride = int(pd_ids.max()) + 2
    if (int(gt_ids.max()) + 2) * pd_stride <= np.iinfo(np.int64).max:
        pair_keys = (gt_ids + 1) * pd_stride + (pd_ids + 1)
    else:
        pair_keys = _encode_keys(gt_ids, pd_ids)
    _, first_pairs, pair_codes = np.unique(
        pair_keys, return_index=True, return_inverse=True
    )
    pair_count = len(first_pairs)

    # Visit each physical pair once, ordered by its first labeled row.
    seen_gt: set[int] = set()
    seen_pd: set[int] = set()
    chosen = np.zeros(pair_count, dtype=np.bool_)
    for pair_idx in np.argsort(first_pairs):
        row = first_pairs[pair_idx]
        gt_id, pd_id = int(gt_ids[row]), int(pd_ids[row])
        if gt_id < 0 or pd_id < 0 or gt_id in seen_gt or pd_id in seen_pd:
            continue
        seen_gt.add(gt_id)
        seen_pd.add(pd_id)
        chosen[pair_idx] = True

    matched_rows = np.flatnonzero(chosen[pair_codes])
    labeled_keys = (
        pair_codes[matched_rows] * n_labels + gt_labels[matched_rows]
    ) * n_labels + pd_labels[matched_rows]
    _, first = np.unique(labeled_keys, return_index=True)
    matched_rows = matched_rows[first]

    # Compact annotation codes bound the flags by fragment size, even when
    # the persistent IDs are sparse or large.
    unique_gt, first_gt, gt_codes = np.unique(
        gt_ids[first_pairs], return_index=True, return_inverse=True
    )
    valid_gt_codes = np.flatnonzero(unique_gt >= 0)
    gt_rows = first_pairs[first_gt[valid_gt_codes]]
    unique_pd, first_pd, pd_codes = np.unique(
        pd_ids[first_pairs], return_index=True, return_inverse=True
    )
    prediction_pairs = np.zeros(pair_count, dtype=np.bool_)
    prediction_pairs[first_pd] = True
    pd_rows = np.flatnonzero(prediction_pairs[pair_codes] & (pd_ids >= 0))
    prediction_codes = pd_codes[pair_codes[pd_rows]]
    _, first = np.unique(
        prediction_codes * n_labels + pd_labels[pd_rows], return_index=True
    )
    pd_rows = pd_rows[first]
    prediction_codes = prediction_codes[first]
    matched_gt_codes = gt_codes[pair_codes[matched_rows]]
    matched_pd_codes = pd_codes[pair_codes[matched_rows]]
    matched_labels = (
        gt_labels[matched_rows] * n_labels + pd_labels[matched_rows]
    )
    matched_ious = ious[matched_rows]
    matched_scores = scores[matched_rows]

    for iou_idx, iou_threshold in enumerate(iou_thresholds):
        mask_iou = (matched_ious > EPSILON) & (matched_ious >= iou_threshold)
        for score_idx, score_threshold in enumerate(score_thresholds):
            eligible = (
                mask_iou
                & (matched_scores > EPSILON)
                & (matched_scores >= score_threshold)
            )
            matrices[iou_idx, score_idx] = np.bincount(
                matched_labels[eligible], minlength=n_labels * n_labels
            ).reshape(n_labels, n_labels)

            used_gt = np.zeros(len(unique_gt), dtype=np.bool_)
            used_gt[matched_gt_codes[eligible]] = True
            unmatched_gt[iou_idx, score_idx] = np.bincount(
                gt_labels[gt_rows][~used_gt[valid_gt_codes]],
                minlength=n_labels,
            )

            used_pd = np.zeros(len(unique_pd), dtype=np.bool_)
            used_pd[matched_pd_codes[eligible]] = True
            mask_unmatched_pd = (
                (scores[pd_rows] > EPSILON)
                & (scores[pd_rows] >= score_threshold)
                & ~used_pd[prediction_codes]
            )
            unmatched_pd[iou_idx, score_idx] = np.bincount(
                pd_labels[pd_rows][mask_unmatched_pd], minlength=n_labels
            )

    return matrices, unmatched_gt, unmatched_pd
