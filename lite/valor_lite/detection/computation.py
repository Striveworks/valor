import numpy as np
import shapely
from numpy.typing import NDArray


def compute_bbox_iou(data: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Computes intersection-over-union (IoU) for axis-aligned bounding boxes.

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

    Index 0 - IoU

    Parameters
    ----------
    data : NDArray[np.float64]
        A sorted array of bounding box pairs.

    Returns
    -------
    NDArray[np.float64]
        Computed IoU's.
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
    Computes intersection-over-union (IoU) for bitmasks.

    Takes data with shape (N, 2):

    Index 0 - first bitmask
    Index 1 - second bitmask

    Returns data with shape (N, 1):

    Index 0 - IoU

    Parameters
    ----------
    data : NDArray[np.float64]
        A sorted array of bitmask pairs.

    Returns
    -------
    NDArray[np.float64]
        Computed IoU's.
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
    Computes intersection-over-union (IoU) for shapely polygons.

    Takes data with shape (N, 2):

    Index 0 - first polygon
    Index 1 - second polygon

    Returns data with shape (N, 1):

    Index 0 - IoU

    Parameters
    ----------
    data : NDArray[np.float64]
        A sorted array of polygon pairs.

    Returns
    -------
    NDArray[np.float64]
        Computed IoU's.
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


def _compute_ranked_pairs_for_datum(
    data: NDArray[np.float64],
    label_metadata: NDArray[np.int32],
) -> NDArray[np.float64]:
    """
    Computes ranked pairs for a datum.
    """

    # remove null predictions
    data = data[data[:, 2] >= 0.0]

    # find best fits for prediction
    mask_label_match = data[:, 4] == data[:, 5]
    matched_predicitons = np.unique(data[mask_label_match, 2].astype(int))
    mask_unmatched_predictions = ~np.isin(data[:, 2], matched_predicitons)
    data = data[mask_label_match | mask_unmatched_predictions]

    # sort by gt_id, iou, score
    indices = np.lexsort(
        (
            data[:, 1],
            -data[:, 3],
            -data[:, 6],
        )
    )
    data = data[indices]

    # remove ignored predictions
    for label_idx, count in enumerate(label_metadata[:, 0]):
        if count > 0:
            continue
        data = data[data[:, 5] != label_idx]

    # only keep the highest ranked pair
    _, indices = np.unique(data[:, [0, 2, 5]], axis=0, return_index=True)

    # np.unique orders its results by value, we need to sort the indices to maintain the results of the lexsort
    data = data[indices, :]

    return data


def compute_ranked_pairs(
    data: list[NDArray[np.float64]],
    label_metadata: NDArray[np.int32],
) -> NDArray[np.float64]:
    """
    Performs pair ranking on input data.

    Takes data with shape (N, 7):

    Index 0 - Datum Index
    Index 1 - GroundTruth Index
    Index 2 - Prediction Index
    Index 3 - IoU
    Index 4 - GroundTruth Label Index
    Index 5 - Prediction Label Index
    Index 6 - Score

    Returns data with shape (N - M, 7)

    Parameters
    ----------
    data : NDArray[np.float64]
        A sorted array summarizing the IOU calculations of one or more pairs.
    label_metadata : NDArray[np.int32]
        An array containing metadata related to labels.

    Returns
    -------
    NDArray[np.float64]
        A filtered array containing only ranked pairs.
    """

    ranked_pairs_by_datum = [
        _compute_ranked_pairs_for_datum(
            data=datum,
            label_metadata=label_metadata,
        )
        for datum in data
    ]
    ranked_pairs = np.concatenate(ranked_pairs_by_datum, axis=0)
    indices = np.lexsort(
        (
            -ranked_pairs[:, 3],  # iou
            -ranked_pairs[:, 6],  # score
        )
    )
    return ranked_pairs[indices]


def compute_metrics(
    data: NDArray[np.float64],
    label_metadata: NDArray[np.int32],
    iou_thresholds: NDArray[np.float64],
    score_thresholds: NDArray[np.float64],
) -> tuple[
    tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        float,
    ],
    tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        float,
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
    Index 3 - IoU
    Index 4 - GroundTruth Label Index
    Index 5 - Prediction Label Index
    Index 6 - Score

    Parameters
    ----------
    data : NDArray[np.float64]
        A sorted array summarizing the IOU calculations of one or more pairs.
    label_metadata : NDArray[np.int32]
        An array containing metadata related to labels.
    iou_thresholds : NDArray[np.float64]
        A 1-D array containing IoU thresholds.
    score_thresholds : NDArray[np.float64]
        A 1-D array containing score thresholds.

    Returns
    -------
    tuple[NDArray, NDArray, NDArray, float]
        Average Precision results.
    tuple[NDArray, NDArray, NDArray, float]
        Average Recall results.
    np.ndarray
        Precision, Recall, TP, FP, FN, F1 Score, Accuracy.
    np.ndarray
        Interpolated Precision-Recall Curves.
    """

    n_rows = data.shape[0]
    n_labels = label_metadata.shape[0]
    n_ious = iou_thresholds.shape[0]
    n_scores = score_thresholds.shape[0]

    if n_ious == 0:
        raise ValueError("At least one IoU threshold must be passed.")
    elif n_scores == 0:
        raise ValueError("At least one score threshold must be passed.")

    average_precision = np.zeros((n_ious, n_labels))
    average_recall = np.zeros((n_scores, n_labels))
    counts = np.zeros((n_ious, n_scores, n_labels, 7))

    pd_labels = data[:, 5].astype(int)
    unique_pd_labels = np.unique(pd_labels)
    gt_count = label_metadata[:, 0]
    running_total_count = np.zeros(
        (n_ious, n_rows),
        dtype=np.float64,
    )
    running_tp_count = np.zeros_like(running_total_count)
    running_gt_count = np.zeros_like(running_total_count)
    pr_curve = np.zeros((n_ious, n_labels, 101))

    mask_score_nonzero = data[:, 6] > 1e-9
    mask_gt_exists = data[:, 1] >= 0.0
    mask_labels_match = np.isclose(data[:, 4], data[:, 5])

    mask_gt_exists_labels_match = mask_gt_exists & mask_labels_match

    mask_tp = mask_score_nonzero & mask_gt_exists_labels_match
    mask_fp = mask_score_nonzero
    mask_fn = mask_gt_exists_labels_match

    for iou_idx in range(n_ious):
        mask_iou = data[:, 3] >= iou_thresholds[iou_idx]

        mask_tp_outer = mask_tp & mask_iou
        mask_fp_outer = mask_fp & (
            (~mask_gt_exists_labels_match & mask_iou) | ~mask_iou
        )
        mask_fn_outer = mask_fn & mask_iou

        for score_idx in range(n_scores):
            mask_score_thresh = data[:, 6] >= score_thresholds[score_idx]

            mask_tp_inner = mask_tp_outer & mask_score_thresh
            mask_fp_inner = mask_fp_outer & mask_score_thresh
            mask_fn_inner = mask_fn_outer & ~mask_score_thresh

            # create true-positive mask score threshold
            tp_candidates = data[mask_tp_inner]
            _, indices_gt_unique = np.unique(
                tp_candidates[:, [0, 1, 4]], axis=0, return_index=True
            )
            mask_gt_unique = np.zeros(tp_candidates.shape[0], dtype=bool)
            mask_gt_unique[indices_gt_unique] = True
            true_positives_mask = np.zeros(n_rows, dtype=bool)
            true_positives_mask[mask_tp_inner] = mask_gt_unique

            # calculate intermediates
            pd_count = np.bincount(pd_labels, minlength=n_labels).astype(float)
            tp_count = np.bincount(
                pd_labels,
                weights=true_positives_mask,
                minlength=n_labels,
            ).astype(float)

            fp_count = np.bincount(
                pd_labels[mask_fp_inner],
                minlength=n_labels,
            ).astype(float)

            fn_count = np.bincount(
                pd_labels[mask_fn_inner],
                minlength=n_labels,
            )

            # calculate component metrics
            recall = np.zeros_like(tp_count)
            precision = np.zeros_like(tp_count)
            np.divide(tp_count, gt_count, where=gt_count > 1e-9, out=recall)
            np.divide(tp_count, pd_count, where=pd_count > 1e-9, out=precision)
            fn_count = gt_count - tp_count

            f1_score = np.zeros_like(precision)
            np.divide(
                np.multiply(precision, recall),
                (precision + recall),
                where=(precision + recall) > 1e-9,
                out=f1_score,
            )

            accuracy = np.zeros_like(tp_count)
            np.divide(
                tp_count,
                (gt_count + pd_count),
                where=(gt_count + pd_count) > 1e-9,
                out=accuracy,
            )

            counts[iou_idx][score_idx] = np.concatenate(
                (
                    tp_count[:, np.newaxis],
                    fp_count[:, np.newaxis],
                    fn_count[:, np.newaxis],
                    precision[:, np.newaxis],
                    recall[:, np.newaxis],
                    f1_score[:, np.newaxis],
                    accuracy[:, np.newaxis],
                ),
                axis=1,
            )

            # calculate recall for AR
            average_recall[score_idx] += recall

        # create true-positive mask score threshold
        tp_candidates = data[mask_tp_outer]
        _, indices_gt_unique = np.unique(
            tp_candidates[:, [0, 1, 4]], axis=0, return_index=True
        )
        mask_gt_unique = np.zeros(tp_candidates.shape[0], dtype=bool)
        mask_gt_unique[indices_gt_unique] = True
        true_positives_mask = np.zeros(n_rows, dtype=bool)
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
    recall_index = np.floor(recall * 100.0).astype(int)
    for iou_idx in range(n_ious):
        p = precision[iou_idx]
        r = recall_index[iou_idx]
        pr_curve[iou_idx, pd_labels, r] = np.maximum(
            pr_curve[iou_idx, pd_labels, r], p
        )

    # calculate average precision
    running_max = np.zeros((n_ious, n_labels))
    for recall in range(100, -1, -1):
        precision = pr_curve[:, :, recall]
        running_max = np.maximum(precision, running_max)
        average_precision += running_max
        pr_curve[:, :, recall] = running_max
    average_precision = average_precision / 101.0

    # calculate average recall
    average_recall = average_recall / n_ious

    # calculate mAP and mAR
    if unique_pd_labels.size > 0:
        mAP = average_precision[:, unique_pd_labels].mean(axis=1)
        mAR = average_recall[:, unique_pd_labels].mean(axis=1)
    else:
        mAP = np.zeros(n_ious, dtype=np.float64)
        mAR = np.zeros(n_scores, dtype=np.float64)

    # calculate AR and AR averaged over thresholds
    APAveragedOverIoUs = average_precision.mean(axis=0)
    ARAveragedOverScores = average_recall.mean(axis=0)

    # calculate mAP and mAR averaged over thresholds
    mAPAveragedOverIoUs = mAP.mean(axis=0)
    mARAveragedOverScores = mAR.mean(axis=0)

    ap_results = (
        average_precision,
        mAP,
        APAveragedOverIoUs,
        mAPAveragedOverIoUs,
    )
    ar_results = (
        average_recall,
        mAR,
        ARAveragedOverScores,
        mARAveragedOverScores,
    )

    return (
        ap_results,
        ar_results,
        counts,
        pr_curve,
    )


def _count_with_examples(
    data: NDArray[np.float64],
    unique_idx: int | list[int],
    label_idx: int | list[int],
) -> tuple[NDArray[np.float64], NDArray[np.int32], NDArray[np.int32]]:
    """
    Helper function for counting occurences of unique detailed pairs.

    Parameters
    ----------
    data : NDArray[np.float64]
        A masked portion of a detailed pairs array.
    unique_idx : int | list[int]
        The index or indices upon which uniqueness is constrained.
    label_idx : int | list[int]
        The index or indices within the unique index or indices that encode labels.

    Returns
    -------
    NDArray[np.float64]
        Examples drawn from the data input.
    NDArray[np.int32]
        Unique label indices.
    NDArray[np.int32]
        Counts for each unique label index.
    """
    unique_rows, indices = np.unique(
        data.astype(int)[:, unique_idx],
        return_index=True,
        axis=0,
    )
    examples = data[indices]
    labels, counts = np.unique(
        unique_rows[:, label_idx], return_counts=True, axis=0
    )
    return examples, labels, counts


def compute_confusion_matrix(
    data: NDArray[np.float64],
    label_metadata: NDArray[np.int32],
    iou_thresholds: NDArray[np.float64],
    score_thresholds: NDArray[np.float64],
    n_examples: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int32]]:
    """
    Compute detailed counts.

    Takes data with shape (N, 7):

    Index 0 - Datum Index
    Index 1 - GroundTruth Index
    Index 2 - Prediction Index
    Index 3 - IoU
    Index 4 - GroundTruth Label Index
    Index 5 - Prediction Label Index
    Index 6 - Score

    Parameters
    ----------
    data : NDArray[np.float64]
        A sorted array summarizing the IOU calculations of one or more pairs.
    label_metadata : NDArray[np.int32]
        An array containing metadata related to labels.
    iou_thresholds : NDArray[np.float64]
        A 1-D array containing IoU thresholds.
    score_thresholds : NDArray[np.float64]
        A 1-D array containing score thresholds.
    n_examples : int
        The maximum number of examples to return per count.

    Returns
    -------
    NDArray[np.float64]
        Confusion matrix.
    NDArray[np.float64]
        Hallucinations.
    NDArray[np.int32]
        Missing Predictions.
    """

    n_labels = label_metadata.shape[0]
    n_ious = iou_thresholds.shape[0]
    n_scores = score_thresholds.shape[0]

    confusion_matrix = -1 * np.ones(
        # (datum idx, gt idx, pd idx, pd score) * n_examples + count
        (n_ious, n_scores, n_labels, n_labels, 4 * n_examples + 1),
        dtype=np.float32,
    )
    hallucinations = -1 * np.ones(
        # (datum idx, pd idx, pd score) * n_examples + count
        (n_ious, n_scores, n_labels, 3 * n_examples + 1),
        dtype=np.float32,
    )
    missing_predictions = -1 * np.ones(
        # (datum idx, gt idx) * n_examples + count
        (n_ious, n_scores, n_labels, 2 * n_examples + 1),
        dtype=np.int32,
    )

    mask_gt_exists = data[:, 1] > -0.5
    mask_pd_exists = data[:, 2] > -0.5
    mask_label_match = np.isclose(data[:, 4], data[:, 5])
    mask_score_nonzero = data[:, 6] > 1e-9
    mask_iou_nonzero = data[:, 3] > 1e-9

    mask_gt_pd_exists = mask_gt_exists & mask_pd_exists
    mask_gt_pd_match = mask_gt_pd_exists & mask_label_match
    mask_gt_pd_mismatch = mask_gt_pd_exists & ~mask_label_match

    groundtruths = data[:, [0, 1]].astype(int)
    predictions = data[:, [0, 2]].astype(int)
    for iou_idx in range(n_ious):
        mask_iou_threshold = data[:, 3] >= iou_thresholds[iou_idx]
        mask_iou = mask_iou_nonzero & mask_iou_threshold

        groundtruths_passing_ious = np.unique(groundtruths[mask_iou], axis=0)
        mask_groundtruths_with_passing_ious = (
            (
                groundtruths.reshape(-1, 1, 2)
                == groundtruths_passing_ious.reshape(1, -1, 2)
            )
            .all(axis=2)
            .any(axis=1)
        )
        mask_groundtruths_without_passing_ious = (
            ~mask_groundtruths_with_passing_ious & mask_gt_exists
        )

        predictions_with_passing_ious = np.unique(
            predictions[mask_iou], axis=0
        )
        mask_predictions_with_passing_ious = (
            (
                predictions.reshape(-1, 1, 2)
                == predictions_with_passing_ious.reshape(1, -1, 2)
            )
            .all(axis=2)
            .any(axis=1)
        )
        mask_predictions_without_passing_ious = (
            ~mask_predictions_with_passing_ious & mask_pd_exists
        )

        for score_idx in range(n_scores):
            mask_score_threshold = data[:, 6] >= score_thresholds[score_idx]
            mask_score = mask_score_nonzero & mask_score_threshold

            groundtruths_with_passing_score = np.unique(
                groundtruths[mask_iou & mask_score], axis=0
            )
            mask_groundtruths_with_passing_score = (
                (
                    groundtruths.reshape(-1, 1, 2)
                    == groundtruths_with_passing_score.reshape(1, -1, 2)
                )
                .all(axis=2)
                .any(axis=1)
            )
            mask_groundtruths_without_passing_score = (
                ~mask_groundtruths_with_passing_score & mask_gt_exists
            )

            # create category masks
            mask_tp = mask_score & mask_iou & mask_gt_pd_match
            mask_misclf = mask_iou & (
                (
                    ~mask_score
                    & mask_gt_pd_match
                    & mask_groundtruths_with_passing_score
                )
                | (mask_score & mask_gt_pd_mismatch)
            )
            mask_halluc = mask_score & mask_predictions_without_passing_ious
            mask_misprd = (
                mask_groundtruths_without_passing_ious
                | mask_groundtruths_without_passing_score
            )

            # filter out true-positives from misclf and misprd
            mask_gts_with_tp_override = (
                (
                    data[mask_misclf][:, [0, 1]].reshape(-1, 1, 2)
                    == data[mask_tp][:, [0, 1]].reshape(1, -1, 2)
                )
                .all(axis=2)
                .any(axis=1)
            )
            mask_pds_with_tp_override = (
                (
                    data[mask_misclf][:, [0, 2]].reshape(-1, 1, 2)
                    == data[mask_tp][:, [0, 2]].reshape(1, -1, 2)
                )
                .all(axis=2)
                .any(axis=1)
            )
            mask_misprd[mask_misclf] |= (
                ~mask_gts_with_tp_override & mask_pds_with_tp_override
            )
            mask_misclf[mask_misclf] &= (
                ~mask_gts_with_tp_override & ~mask_pds_with_tp_override
            )

            # count true positives
            tp_examples, tp_labels, tp_counts = _count_with_examples(
                data[mask_tp],
                unique_idx=[0, 2, 5],
                label_idx=2,
            )

            # count misclassifications
            (
                misclf_examples,
                misclf_labels,
                misclf_counts,
            ) = _count_with_examples(
                data[mask_misclf], unique_idx=[0, 1, 2, 4, 5], label_idx=[3, 4]
            )

            # count hallucinations
            (
                halluc_examples,
                halluc_labels,
                halluc_counts,
            ) = _count_with_examples(
                data[mask_halluc], unique_idx=[0, 2, 5], label_idx=2
            )

            # count missing predictions
            (
                misprd_examples,
                misprd_labels,
                misprd_counts,
            ) = _count_with_examples(
                data[mask_misprd], unique_idx=[0, 1, 4], label_idx=2
            )

            # store the counts
            confusion_matrix[
                iou_idx, score_idx, tp_labels, tp_labels, 0
            ] = tp_counts
            confusion_matrix[
                iou_idx,
                score_idx,
                misclf_labels[:, 0],
                misclf_labels[:, 1],
                0,
            ] = misclf_counts
            hallucinations[
                iou_idx,
                score_idx,
                halluc_labels,
                0,
            ] = halluc_counts
            missing_predictions[
                iou_idx,
                score_idx,
                misprd_labels,
                0,
            ] = misprd_counts

            # store examples
            if n_examples > 0:
                for label_idx in range(n_labels):

                    # true-positive examples
                    mask_tp_label = tp_examples[:, 5] == label_idx
                    if mask_tp_label.sum() > 0:
                        tp_label_examples = tp_examples[mask_tp_label][
                            :n_examples
                        ]
                        confusion_matrix[
                            iou_idx,
                            score_idx,
                            label_idx,
                            label_idx,
                            1 : 4 * tp_label_examples.shape[0] + 1,
                        ] = tp_label_examples[:, [0, 1, 2, 6]].flatten()

                    # misclassification examples
                    mask_misclf_gt_label = misclf_examples[:, 4] == label_idx
                    if mask_misclf_gt_label.sum() > 0:
                        for pd_label_idx in range(n_labels):
                            mask_misclf_pd_label = (
                                misclf_examples[:, 5] == pd_label_idx
                            )
                            mask_misclf_label_combo = (
                                mask_misclf_gt_label & mask_misclf_pd_label
                            )
                            if mask_misclf_label_combo.sum() > 0:
                                misclf_label_examples = misclf_examples[
                                    mask_misclf_label_combo
                                ][:n_examples]
                                confusion_matrix[
                                    iou_idx,
                                    score_idx,
                                    label_idx,
                                    pd_label_idx,
                                    1 : 4 * misclf_label_examples.shape[0] + 1,
                                ] = misclf_label_examples[
                                    :, [0, 1, 2, 6]
                                ].flatten()

                    # hallucination examples
                    mask_halluc_label = halluc_examples[:, 5] == label_idx
                    if mask_halluc_label.sum() > 0:
                        halluc_label_examples = halluc_examples[
                            mask_halluc_label
                        ][:n_examples]
                        hallucinations[
                            iou_idx,
                            score_idx,
                            label_idx,
                            1 : 3 * halluc_label_examples.shape[0] + 1,
                        ] = halluc_label_examples[:, [0, 2, 6]].flatten()

                    # missing prediction examples
                    mask_misprd_label = misprd_examples[:, 4] == label_idx
                    if misprd_examples.size > 0:
                        misprd_label_examples = misprd_examples[
                            mask_misprd_label
                        ][:n_examples]
                        missing_predictions[
                            iou_idx,
                            score_idx,
                            label_idx,
                            1 : 2 * misprd_label_examples.shape[0] + 1,
                        ] = misprd_label_examples[:, [0, 1]].flatten()

    return (
        confusion_matrix,
        hallucinations,
        missing_predictions,
    )
