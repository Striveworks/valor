import numpy as np
from numpy.typing import NDArray


def compute_running_counts(
    ranked_pairs: NDArray[np.float64],
    label_metadata: NDArray[np.int32],
    iou_thresholds: NDArray[np.float64],
    score_thresholds: NDArray[np.float64],
    groundtruth_count: int,    
) -> tuple[
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
    gt_count = groundtruth_count
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
        running_tp_counts
        counts,
        pr_curve,
    )