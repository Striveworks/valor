from enum import IntFlag, auto

import numpy as np
import shapely
from numpy.typing import NDArray


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
        where=union_area >= 1e-9,
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
        where=union_ >= 1e-9,
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
        where=union_areas >= 1e-9,
        out=ious,
    )
    return ious


def compute_label_metadata(
    ids: NDArray[np.int32],
    n_labels: int,
) -> NDArray[np.int32]:
    """
    Computes label metadata returning a count of annotations per label.

    Parameters
    ----------
    detailed_pairs : NDArray[np.int32]
        Detailed annotation pairings with shape (N, 7).
            Index 0 - Datum Index
            Index 1 - GroundTruth Index
            Index 2 - Prediction Index
            Index 3 - GroundTruth Label Index
            Index 4 - Prediction Label Index
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

    ground_truth_pairs = ids[:, (0, 1, 3)]
    ground_truth_pairs = ground_truth_pairs[ground_truth_pairs[:, 1] >= 0]
    unique_pairs = np.unique(ground_truth_pairs, axis=0)
    label_indices, unique_counts = np.unique(
        unique_pairs[:, 2], return_counts=True
    )
    label_metadata[label_indices.astype(np.int32), 0] = unique_counts

    prediction_pairs = ids[:, (0, 2, 4)]
    prediction_pairs = prediction_pairs[prediction_pairs[:, 1] >= 0]
    unique_pairs = np.unique(prediction_pairs, axis=0)
    label_indices, unique_counts = np.unique(
        unique_pairs[:, 2], return_counts=True
    )
    label_metadata[label_indices.astype(np.int32), 1] = unique_counts

    return label_metadata


def filter_cache(
    detailed_pairs: NDArray[np.float64],
    mask_datums: NDArray[np.bool_],
    mask_predictions: NDArray[np.bool_],
    mask_ground_truths: NDArray[np.bool_],
    n_labels: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int32],]:
    """
    Performs filtering on a detailed cache.

    Parameters
    ----------
    detailed_pairs : NDArray[float64]
        A list of sorted detailed pairs with size (N, 7).
    mask_datums : NDArray[bool]
        A boolean mask with size (N,).
    mask_ground_truths : NDArray[bool]
        A boolean mask with size (N,).
    mask_predictions : NDArray[bool]
        A boolean mask with size (N,).
    n_labels : int
        The total number of unique labels.

    Returns
    -------
    NDArray[float64]
        Filtered detailed pairs.
    NDArray[float64]
        Filtered ranked pairs.
    NDArray[int32]
        Label metadata.
    """
    # filter datums
    detailed_pairs = detailed_pairs[mask_datums].copy()

    # filter ground truths
    if mask_ground_truths.any():
        invalid_groundtruth_indices = np.where(mask_ground_truths)[0]
        detailed_pairs[
            invalid_groundtruth_indices[:, None], (1, 3, 5)
        ] = np.array([[-1, -1, 0]])

    # filter predictions
    if mask_predictions.any():
        invalid_prediction_indices = np.where(mask_predictions)[0]
        detailed_pairs[
            invalid_prediction_indices[:, None], (2, 4, 5, 6)
        ] = np.array([[-1, -1, 0, -1]])

    # filter null pairs
    mask_null_pairs = np.all(
        np.isclose(
            detailed_pairs[:, 1:5],
            np.array([-1.0, -1.0, -1.0, -1.0]),
        ),
        axis=1,
    )
    detailed_pairs = detailed_pairs[~mask_null_pairs]

    # sorts by score, iou with ground truth id as a tie-breaker
    indices = np.lexsort(
        (
            detailed_pairs[:, 1],  # ground truth id
            -detailed_pairs[:, 5],  # iou
            -detailed_pairs[:, 6],  # score
        )
    )
    detailed_pairs = detailed_pairs[indices]
    label_metadata = compute_label_metadata(
        ids=detailed_pairs[:, :5].astype(np.int32),
        n_labels=n_labels,
    )
    ranked_pairs = rank_pairs(
        detailed_pairs=detailed_pairs,
        label_metadata=label_metadata,
    )
    return (
        detailed_pairs,
        ranked_pairs,
        label_metadata,
    )


def rank_pairs(
    detailed_pairs: NDArray[np.float64],
    label_metadata: NDArray[np.int32],
) -> NDArray[np.float64]:
    """
    Highly optimized pair ranking for computing precision and recall based metrics.

    Only ground truths and predictions that provide unique information are kept. The unkept
    pairs are represented via the label metadata array.

    Parameters
    ----------
    detailed_pairs : NDArray[np.float64]
        Detailed annotation pairs with shape (n_pairs, 7).
            Index 0 - Datum Index
            Index 1 - GroundTruth Index
            Index 2 - Prediction Index
            Index 3 - GroundTruth Label Index
            Index 4 - Prediction Label Index
            Index 5 - IOU
            Index 6 - Score
    label_metadata : NDArray[np.int32]
        Array containing label counts with shape (n_labels, 2)
            Index 0 - Ground truth label count
            Index 1 - Prediction label count

    Returns
    -------
    NDArray[np.float64]
        Array of ranked pairs for precision-recall metric computation.
    """
    pairs = detailed_pairs

    # remove null predictions
    pairs = pairs[pairs[:, 2] >= 0.0]

    # find best fits for prediction
    mask_label_match = np.isclose(pairs[:, 3], pairs[:, 4])
    matched_predictions = np.unique(pairs[mask_label_match, 2])
    mask_unmatched_predictions = ~np.isin(pairs[:, 2], matched_predictions)
    pairs = pairs[mask_label_match | mask_unmatched_predictions]

    # remove predictions for labels that have no ground truths
    for label_idx, count in enumerate(label_metadata[:, 0]):
        if count > 0:
            continue
        pairs = pairs[pairs[:, 4] != label_idx]

    # only keep the highest ranked pair
    _, indices = np.unique(pairs[:, [0, 2, 4]], axis=0, return_index=True)
    pairs = pairs[indices]

    # np.unique orders its results by value, we need to sort the indices to maintain the results of the lexsort
    indices = np.lexsort(
        (
            -pairs[:, 5],  # iou
            -pairs[:, 6],  # score
        )
    )
    pairs = pairs[indices]

    return pairs


def compute_precion_recall(
    ranked_pairs: NDArray[np.float64],
    label_metadata: NDArray[np.int32],
    iou_thresholds: NDArray[np.float64],
    score_thresholds: NDArray[np.float64],
) -> tuple[
    tuple[
        NDArray[np.float64],
        NDArray[np.float64],
    ],
    tuple[
        NDArray[np.float64],
        NDArray[np.float64],
    ],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """
    Computes Object Detection metrics.

    Takes data with shape (N, 7):

    Index 0 - Datum Index
    Index 1 - GroundTruth Index
    Index 2 - Prediction Index
    Index 3 - IOU
    Index 4 - GroundTruth Label Index
    Index 5 - Prediction Label Index
    Index 6 - Score

    Parameters
    ----------
    ranked_pairs : NDArray[np.float64]
        A ranked array summarizing the IOU calculations of one or more pairs.
    label_metadata : NDArray[np.int32]
        An array containing metadata related to labels.
    iou_thresholds : NDArray[np.float64]
        A 1-D array containing IOU thresholds.
    score_thresholds : NDArray[np.float64]
        A 1-D array containing score thresholds.

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64]]
        Average Precision results (AP, mAP).
    tuple[NDArray[np.float64], NDArray[np.float64]]
        Average Recall results (AR, mAR).
    NDArray[np.float64]
        Precision, Recall, TP, FP, FN, F1 Score.
    NDArray[np.float64]
        Interpolated Precision-Recall Curves.
    """
    n_rows = ranked_pairs.shape[0]
    n_labels = label_metadata.shape[0]
    n_ious = iou_thresholds.shape[0]
    n_scores = score_thresholds.shape[0]

    # initialize result arrays
    average_precision = np.zeros((n_ious, n_labels), dtype=np.float64)
    mAP = np.zeros(n_ious, dtype=np.float64)
    average_recall = np.zeros((n_scores, n_labels), dtype=np.float64)
    mAR = np.zeros(n_scores, dtype=np.float64)
    counts = np.zeros((n_ious, n_scores, n_labels, 6), dtype=np.float64)
    pr_curve = np.zeros((n_ious, n_labels, 101, 2))

    # start computation
    ids = ranked_pairs[:, :5].astype(np.int32)
    gt_ids = ids[:, 1]
    gt_labels = ids[:, 3]
    pd_labels = ids[:, 4]
    ious = ranked_pairs[:, 5]
    scores = ranked_pairs[:, 6]

    unique_pd_labels, unique_pd_indices = np.unique(
        pd_labels, return_index=True
    )
    gt_count = label_metadata[:, 0]
    running_total_count = np.zeros(
        (n_ious, n_rows),
        dtype=np.float64,
    )
    running_tp_count = np.zeros_like(running_total_count)
    running_gt_count = np.zeros_like(running_total_count)

    mask_score_nonzero = scores > 1e-9
    mask_gt_exists = gt_ids >= 0.0
    mask_labels_match = np.isclose(gt_labels, pd_labels)

    mask_gt_exists_labels_match = mask_gt_exists & mask_labels_match

    mask_tp = mask_score_nonzero & mask_gt_exists_labels_match
    mask_fp = mask_score_nonzero
    mask_fn = mask_gt_exists_labels_match

    for iou_idx in range(n_ious):
        mask_iou = ious >= iou_thresholds[iou_idx]

        mask_tp_outer = mask_tp & mask_iou
        mask_fp_outer = mask_fp & (
            (~mask_gt_exists_labels_match & mask_iou) | ~mask_iou
        )
        mask_fn_outer = mask_fn & mask_iou

        for score_idx in range(n_scores):
            mask_score_thresh = scores >= score_thresholds[score_idx]

            mask_tp_inner = mask_tp_outer & mask_score_thresh
            mask_fp_inner = mask_fp_outer & mask_score_thresh
            mask_fn_inner = mask_fn_outer & ~mask_score_thresh

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
            tp_count = np.bincount(
                pd_labels,
                weights=true_positives_mask,
                minlength=n_labels,
            ).astype(np.float64)
            fp_count = np.bincount(
                pd_labels[mask_fp_inner],
                minlength=n_labels,
            ).astype(np.float64)
            fn_count = np.bincount(
                pd_labels[mask_fn_inner],
                minlength=n_labels,
            )

            fn_count = gt_count - tp_count
            tp_fp_count = tp_count + fp_count

            # calculate component metrics
            recall = np.zeros_like(tp_count)
            np.divide(tp_count, gt_count, where=gt_count > 1e-9, out=recall)

            precision = np.zeros_like(tp_count)
            np.divide(
                tp_count, tp_fp_count, where=tp_fp_count > 1e-9, out=precision
            )

            f1_score = np.zeros_like(precision)
            np.divide(
                2 * np.multiply(precision, recall),
                (precision + recall),
                where=(precision + recall) > 1e-9,
                out=f1_score,
                dtype=np.float64,
            )

            counts[iou_idx][score_idx] = np.concatenate(
                (
                    tp_count[:, np.newaxis],
                    fp_count[:, np.newaxis],
                    fn_count[:, np.newaxis],
                    precision[:, np.newaxis],
                    recall[:, np.newaxis],
                    f1_score[:, np.newaxis],
                ),
                axis=1,
            )

            # calculate recall for AR
            average_recall[score_idx] += recall

        # create true-positive mask score threshold
        tp_candidates = ids[mask_tp_outer]
        _, indices_gt_unique = np.unique(
            tp_candidates[:, [0, 1, 3]], axis=0, return_index=True
        )
        mask_gt_unique = np.zeros(tp_candidates.shape[0], dtype=np.bool_)
        mask_gt_unique[indices_gt_unique] = True
        true_positives_mask = np.zeros(n_rows, dtype=np.bool_)
        true_positives_mask[mask_tp_outer] = mask_gt_unique

        # count running tp and total for AP
        for pd_label in unique_pd_labels:
            mask_pd_label = pd_labels == pd_label
            running_gt_count[iou_idx][mask_pd_label] = gt_count[pd_label]
            running_total_count[iou_idx][mask_pd_label] = np.arange(
                1, mask_pd_label.sum() + 1
            )
            mask_tp_for_counting = mask_pd_label & true_positives_mask
            running_tp_count[iou_idx][mask_tp_for_counting] = np.arange(
                1, mask_tp_for_counting.sum() + 1
            )

    # calculate running precision-recall points for AP
    precision = np.zeros_like(running_total_count)
    np.divide(
        running_tp_count,
        running_total_count,
        where=running_total_count > 1e-9,
        out=precision,
    )
    recall = np.zeros_like(running_total_count)
    np.divide(
        running_tp_count,
        running_gt_count,
        where=running_gt_count > 1e-9,
        out=recall,
    )
    recall_index = np.floor(recall * 100.0).astype(np.int32)

    # bin precision-recall curve
    for iou_idx in range(n_ious):
        p = precision[iou_idx]
        r = recall_index[iou_idx]
        pr_curve[iou_idx, pd_labels, r, 0] = np.maximum(
            pr_curve[iou_idx, pd_labels, r, 0],
            p,
        )
        pr_curve[iou_idx, pd_labels, r, 1] = np.maximum(
            pr_curve[iou_idx, pd_labels, r, 1],
            scores,
        )

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

    # calculate average recall
    average_recall = average_recall / n_ious

    # calculate mAP and mAR
    if unique_pd_labels.size > 0:
        mAP: NDArray[np.float64] = average_precision[:, unique_pd_labels].mean(
            axis=1
        )
        mAR: NDArray[np.float64] = average_recall[:, unique_pd_labels].mean(
            axis=1
        )

    return (
        (average_precision.astype(np.float64), mAP),
        (average_recall, mAR),
        counts,
        pr_curve,
    )


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


def compute_confusion_matrix(
    detailed_pairs: NDArray[np.float64],
    iou_thresholds: NDArray[np.float64],
    score_thresholds: NDArray[np.float64],
) -> NDArray[np.uint8]:
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
    mask_score_nonzero = scores > 1e-9
    mask_iou_nonzero = ious > 1e-9

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

    return pair_classifications
