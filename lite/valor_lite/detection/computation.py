import numpy as np
from numpy.typing import NDArray


def compute_bbox_iou(data: NDArray[np.floating]) -> NDArray[np.floating]:
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
    data : NDArray[np.floating]
        A sorted array of bounding box pairs.

    Returns
    -------
    NDArray[np.floating]
        Computed IoU's.
    """

    xmin1, xmax1, ymin1, ymax1 = (
        data[:, 0],
        data[:, 1],
        data[:, 2],
        data[:, 3],
    )
    xmin2, xmax2, ymin2, ymax2 = (
        data[:, 4],
        data[:, 5],
        data[:, 6],
        data[:, 7],
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

    iou = np.zeros(data.shape[0])
    valid_union_mask = union_area >= 1e-9
    iou[valid_union_mask] = (
        intersection_area[valid_union_mask] / union_area[valid_union_mask]
    )
    return iou


def compute_bitmask_iou(data: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Computes intersection-over-union (IoU) for bitmasks.

    Takes data with shape (N, 2):

    Index 0 - first bitmask
    Index 1 - second bitmask

    Returns data with shape (N, 1):

    Index 0 - IoU

    Parameters
    ----------
    data : NDArray[np.floating]
        A sorted array of bitmask pairs.

    Returns
    -------
    NDArray[np.floating]
        Computed IoU's.
    """
    intersection_ = np.array([np.logical_and(x, y).sum() for x, y in data])
    union_ = np.array([np.logical_or(x, y).sum() for x, y in data])

    return intersection_ / union_


def compute_polygon_iou(
    data: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    Computes intersection-over-union (IoU) for shapely polygons.

    Takes data with shape (N, 2):

    Index 0 - first polygon
    Index 1 - second polygon

    Returns data with shape (N, 1):

    Index 0 - IoU

    Parameters
    ----------
    data : NDArray[np.floating]
        A sorted array of polygon pairs.

    Returns
    -------
    NDArray[np.floating]
        Computed IoU's.
    """
    intersection_ = np.array(
        [poly1.intersection(poly2).area for poly1, poly2 in data]
    )
    union_ = np.array(
        [
            poly1.area + poly2.area - intersection_[i]
            for i, (poly1, poly2) in enumerate(data)
        ]
    )

    return intersection_ / union_


def _compute_ranked_pairs_for_datum(
    data: NDArray[np.floating],
    label_metadata: NDArray[np.int32],
) -> NDArray[np.floating]:
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
    data: list[NDArray[np.floating]],
    label_metadata: NDArray[np.int32],
) -> NDArray[np.floating]:
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
    data : NDArray[np.floating]
        A sorted array summarizing the IOU calculations of one or more pairs.
    label_metadata : NDArray[np.int32]
        An array containing metadata related to labels.

    Returns
    -------
    NDArray[np.floating]
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
    data: NDArray[np.floating],
    label_metadata: NDArray[np.int32],
    iou_thresholds: NDArray[np.floating],
    score_thresholds: NDArray[np.floating],
) -> tuple[
    tuple[
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.floating],
    ],
    tuple[
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.floating],
    ],
    NDArray[np.floating],
    NDArray[np.floating],
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
    data : NDArray[np.floating]
        A sorted array summarizing the IOU calculations of one or more pairs.
    label_metadata : NDArray[np.int32]
        An array containing metadata related to labels.
    iou_thresholds : NDArray[np.floating]
        A 1-D array containing IoU thresholds.
    score_thresholds : NDArray[np.floating]
        A 1-D array containing score thresholds.

    Returns
    -------
    tuple[NDArray, NDArray, NDArray NDArray]
        Average Precision results.
    tuple[NDArray, NDArray, NDArray NDArray]
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
    average_recall /= n_ious

    # calculate mAP and mAR
    label_key_mapping = label_metadata[unique_pd_labels, 2]
    label_keys = np.unique(label_metadata[:, 2])
    mAP = np.ones((n_ious, label_keys.shape[0])) * -1.0
    mAR = np.ones((n_scores, label_keys.shape[0])) * -1.0
    for key in np.unique(label_key_mapping):
        labels = unique_pd_labels[label_key_mapping == key]
        key_idx = int(key)
        mAP[:, key_idx] = average_precision[:, labels].mean(axis=1)
        mAR[:, key_idx] = average_recall[:, labels].mean(axis=1)

    # calculate AP and mAP averaged over iou thresholds
    APAveragedOverIoUs = average_precision.mean(axis=0)
    mAPAveragedOverIoUs = mAP.mean(axis=0)

    # calculate AR and mAR averaged over score thresholds
    ARAveragedOverIoUs = average_recall.mean(axis=0)
    mARAveragedOverIoUs = mAR.mean(axis=0)

    ap_results = (
        average_precision,
        mAP,
        APAveragedOverIoUs,
        mAPAveragedOverIoUs,
    )
    ar_results = (
        average_recall,
        mAR,
        ARAveragedOverIoUs,
        mARAveragedOverIoUs,
    )

    return (
        ap_results,
        ar_results,
        counts,
        pr_curve,
    )


def compute_detailed_metrics(
    data: NDArray[np.floating],
    label_metadata: NDArray[np.int32],
    iou_thresholds: NDArray[np.floating],
    score_thresholds: NDArray[np.floating],
    n_examples: int,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.int32]]:

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
    data : NDArray[np.floating]
        A sorted array summarizing the IOU calculations of one or more pairs.
    label_metadata : NDArray[np.int32]
        An array containing metadata related to labels.
    iou_thresholds : NDArray[np.floating]
        A 1-D array containing IoU thresholds.
    score_thresholds : NDArray[np.floating]
        A 1-D array containing score thresholds.
    n_examples : int
        The maximum number of examples to return per count.

    Returns
    -------
    NDArray[np.floating]
        Confusion matrix.
    NDArray[np.floating]
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

            mask_tp = mask_score & mask_iou & mask_gt_pd_match
            mask_fp_misclf = mask_score & mask_iou & mask_gt_pd_mismatch
            mask_fn_misclf = mask_iou & (
                (
                    ~mask_score
                    & mask_gt_pd_match
                    & mask_groundtruths_with_passing_score
                )
                | (mask_score & mask_gt_pd_mismatch)
            )
            mask_fp_halluc = mask_score & mask_predictions_without_passing_ious
            mask_fn_misprd = (
                mask_groundtruths_without_passing_ious
                | mask_groundtruths_without_passing_score
            )

            # find unique pairs
            tp_labels, tp_indices, tp_inverse, tp_counts = np.unique(
                data[mask_tp][:, 4].astype(int),
                return_index=True,
                return_counts=True,
                return_inverse=True,
                axis=0,
            )
            (
                fp_misclf_labels,
                fp_misclf_indices,
                fp_misclf_inverse,
                fp_misclf_counts,
            ) = np.unique(
                data[mask_fp_misclf][:, [4, 5]].astype(int),
                return_index=True,
                return_inverse=True,
                return_counts=True,
                axis=0,
            )
            (
                fp_halluc_labels,
                fp_halluc_indices,
                fp_halluc_inverse,
                fp_halluc_counts,
            ) = np.unique(
                data[mask_fp_halluc][:, 5].astype(int),
                axis=0,
                return_index=True,
                return_inverse=True,
                return_counts=True,
            )
            (
                fn_misprd_labels,
                fn_misprd_indices,
                fn_misprd_inverse,
                fn_misprd_counts,
            ) = np.unique(
                data[mask_fn_misprd][:, 4].astype(int),
                axis=0,
                return_index=True,
                return_inverse=True,
                return_counts=True,
            )

            # filter out predictions and groundtruths involved in true-positives
            if tp_indices.size > 0:
                tp_predictions = np.unique(
                    data[mask_tp][tp_indices, [0, 2, 4]]
                    .astype(int)
                    .reshape(-1, 3),
                    axis=0,
                )
                # tp_groundtruths = np.unique(
                #     data[mask_tp][tp_indices, [0, 1, 3]]
                #     .astype(int)
                #     .reshape(-1, 3),
                #     axis=0,
                # )
                data[mask_fp_misclf][fp_misclf_indices, [0, 2, 4]].reshape(
                    -1, 1, 3
                )
                mask_fp_misclf_is_tp = (
                    (
                        data[mask_fp_misclf][
                            fp_misclf_indices, [0, 2, 4]
                        ].reshape(-1, 1, 3)
                        == tp_predictions.reshape(1, -1, 3)
                    )
                    .all(axis=2)
                    .any(axis=1)
                )
                fp_misclf_labels = fp_misclf_labels[~mask_fp_misclf_is_tp]
                fp_misclf_indices = fp_misclf_indices[~mask_fp_misclf_is_tp]

            # store the results
            confusion_matrix[
                iou_idx, score_idx, tp_labels, tp_labels, 0
            ] = tp_counts
            confusion_matrix[
                iou_idx,
                score_idx,
                fp_misclf_labels[:, 0],
                fp_misclf_labels[:, 1],
                0,
            ] = fp_misclf_counts
            hallucinations[
                iou_idx,
                score_idx,
                fp_halluc_labels,
                0,
            ] = fp_halluc_counts
            missing_predictions[
                iou_idx,
                score_idx,
                fn_misprd_labels,
                0,
            ] = fn_misprd_counts

            # store examples
            if n_examples > 0:
                for label_idx in range(n_labels):

                    tp_label_idx = tp_indices[tp_labels == label_idx]
                    if tp_label_idx.size > 0:
                        tp_example_indices = np.where(
                            tp_inverse == tp_label_idx
                        )[0][:n_examples]
                        tp_examples = data[mask_tp][tp_example_indices, :][
                            :, [0, 1, 2, 6]
                        ]
                        confusion_matrix[
                            iou_idx,
                            score_idx,
                            label_idx,
                            label_idx,
                            1 : 4 * tp_example_indices.size + 1,
                        ] = tp_examples.flatten()

                    mask_misclf_gt_indices = (
                        fp_misclf_labels[:, 0] == label_idx
                    )
                    if mask_misclf_gt_indices.size > 0:
                        for inner_label_idx in range(n_labels):
                            mask_misclf_pd_indices = (
                                fp_misclf_labels[:, 1] == inner_label_idx
                            )
                            misclf_label_idx = fp_misclf_indices[
                                mask_misclf_gt_indices & mask_misclf_pd_indices
                            ]
                            if misclf_label_idx.size > 0:
                                misclf_example_indices = np.where(
                                    fp_misclf_inverse == misclf_label_idx
                                )[0][:n_examples]
                                misclf_examples = data[mask_fp_misclf][
                                    misclf_example_indices, :
                                ][:, [0, 1, 2, 6]]
                                confusion_matrix[
                                    iou_idx,
                                    score_idx,
                                    label_idx,
                                    inner_label_idx,
                                    1 : 4 * misclf_example_indices.size + 1,
                                ] = misclf_examples.flatten()

                    fp_halluc_label_idx = fp_halluc_indices[
                        fp_halluc_labels == label_idx
                    ]
                    if fp_halluc_label_idx.size > 0:
                        fp_halluc_example_indices = np.where(
                            fp_halluc_inverse == fp_halluc_label_idx
                        )[0][:n_examples]
                        fp_halluc_examples = data[mask_fp_halluc][
                            fp_halluc_example_indices, :
                        ][:, [0, 2, 6]]
                        hallucinations[
                            iou_idx,
                            score_idx,
                            label_idx,
                            1 : 3 * fp_halluc_example_indices.size + 1,
                        ] = fp_halluc_examples.flatten()

                    fn_misprd_label_idx = fn_misprd_indices[
                        fn_misprd_labels == label_idx
                    ]
                    if fn_misprd_label_idx.size > 0:
                        fn_misprd_example_indices = np.where(
                            fn_misprd_inverse == fn_misprd_label_idx
                        )[0][:n_examples]
                        fn_misprd_examples = data[mask_fn_misprd][
                            fn_misprd_example_indices, :
                        ][:, [0, 1]]
                        missing_predictions[
                            iou_idx,
                            score_idx,
                            label_idx,
                            1 : 2 * fn_misprd_example_indices.size + 1,
                        ] = fn_misprd_examples.flatten()

    return (
        confusion_matrix,
        hallucinations,
        missing_predictions,
    )
