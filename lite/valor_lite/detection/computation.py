import numpy as np
from numpy.typing import NDArray

# datum id  0
# gt        1
# pd        2
# iou       3
# gt label  4
# pd label  5
# score     6


def compute_iou(data: NDArray[np.floating]) -> NDArray[np.floating]:

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


def _compute_ranked_pairs_for_datum(
    data: np.ndarray,
    label_counts: np.ndarray,
) -> np.ndarray:
    """
    Computes ranked pairs for a datum.
    """

    # remove null predictions
    data = data[data[:, 2] >= 0.0]

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
    for label_idx, count in enumerate(label_counts[:, 0]):
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
    label_counts: NDArray[np.integer],
) -> NDArray[np.floating]:
    pairs = np.concatenate(
        [
            _compute_ranked_pairs_for_datum(
                datum,
                label_counts=label_counts,
            )
            for datum in data
        ],
        axis=0,
    )
    indices = np.lexsort(
        (
            -pairs[:, 3],  # iou
            -pairs[:, 6],  # score
        )
    )
    return pairs[indices]


def compute_metrics(
    data: np.ndarray,
    label_counts: np.ndarray,
    iou_thresholds: np.ndarray,
    score_thresholds: np.ndarray,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Computes Object Detection metrics.

    Returns
    -------
    np.ndarray
        Average Precision.
    np.ndarray
        Average Recall.
    np.ndarray
        mAP.
    np.ndarray
        mAR.
    np.ndarray
        Precision, Recall, TP, FP, FN, F1 Score, Accuracy.
    np.ndarray
        Interpolated Precision-Recall Curves.
    """

    n_rows = data.shape[0]
    n_labels = label_counts.shape[0]
    n_ious = iou_thresholds.shape[0]
    n_scores = score_thresholds.shape[0]

    average_precision = np.zeros((n_ious, n_labels))
    average_recall = np.zeros((n_scores, n_labels))
    precision_recall = np.zeros((n_ious, n_scores, n_labels, 7))

    pd_labels = data[:, 5].astype(int)
    unique_pd_labels = np.unique(pd_labels)
    gt_count = label_counts[unique_pd_labels, 0]
    running_total_count = np.zeros(
        (n_ious, n_rows),
        dtype=np.int32,
    )
    running_tp_count = np.zeros_like(running_total_count)
    running_gt_count = np.zeros_like(running_total_count)
    pr_curve = np.zeros((n_ious, n_labels, 101))

    mask_score_nonzero = data[:, 6] > 1e-9
    mask_labels_match = np.isclose(data[:, 4], data[:, 5])
    mask_gt_exists = data[:, 1] >= 0.0

    for iou_idx in range(n_ious):

        mask_iou = data[:, 3] >= iou_thresholds[iou_idx]
        mask = (
            mask_score_nonzero & mask_iou & mask_labels_match & mask_gt_exists
        )

        for score_idx in range(n_scores):

            mask_score_thresh = data[:, 6] >= score_thresholds[score_idx]

            # create true-positive mask score threshold
            tp_candidates = data[mask & mask_score_thresh]
            _, indices_gt_unique = np.unique(
                tp_candidates[:, [0, 1, 4]], axis=0, return_index=True
            )
            mask_gt_unique = np.zeros(tp_candidates.shape[0], dtype=bool)
            mask_gt_unique[indices_gt_unique] = True
            true_positives_mask = np.zeros(n_rows, dtype=bool)
            true_positives_mask[mask & mask_score_thresh] = mask_gt_unique

            # calculate intermediates
            pd_count = np.bincount(pd_labels)
            pd_count = pd_count[unique_pd_labels]
            tp_count = np.bincount(
                pd_labels,
                weights=true_positives_mask,
            )
            tp_count = tp_count[unique_pd_labels]

            # calculate component metrics
            precision = tp_count / pd_count
            recall = tp_count / gt_count
            fp_count = pd_count - tp_count
            fn_count = gt_count - tp_count
            f1_score = np.divide(
                np.multiply(precision, recall),
                (precision + recall),
                where=(precision + recall) > 1e-9,
            )
            accuracy = tp_count / (gt_count + pd_count)
            precision_recall[iou_idx][score_idx][
                unique_pd_labels
            ] = np.concatenate(
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
            average_recall[score_idx][unique_pd_labels] += recall

        # create true-positive mask score threshold
        tp_candidates = data[mask]
        _, indices_gt_unique = np.unique(
            tp_candidates[:, [0, 1, 4]], axis=0, return_index=True
        )
        mask_gt_unique = np.zeros(tp_candidates.shape[0], dtype=bool)
        mask_gt_unique[indices_gt_unique] = True
        true_positives_mask = np.zeros(n_rows, dtype=bool)
        true_positives_mask[mask] = mask_gt_unique

        # count running tp and total for AP
        for pd_label in unique_pd_labels:
            mask_pd_label = pd_labels == pd_label
            running_gt_count[iou_idx][mask_pd_label] = label_counts[pd_label][
                0
            ]
            running_total_count[iou_idx][mask_pd_label] = np.arange(
                1, mask_pd_label.sum() + 1
            )
            mask_tp = mask_pd_label & true_positives_mask
            running_tp_count[iou_idx][mask_tp] = np.arange(
                1, mask_tp.sum() + 1
            )

    # calculate running precision-recall points for AP
    precision = np.divide(
        running_tp_count, running_total_count, where=running_total_count > 0
    )
    recall = np.divide(
        running_tp_count, running_gt_count, where=running_gt_count > 0
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
    label_key_mapping = label_counts[unique_pd_labels, 2]
    label_keys = np.unique(label_counts[:, 2])
    mAP = np.ones((n_ious, label_keys.shape[0])) * -1.0
    mAR = np.ones((n_scores, label_keys.shape[0])) * -1.0
    for key in np.unique(label_key_mapping):
        labels = unique_pd_labels[label_key_mapping == key]
        key_idx = int(key)
        mAP[:, key_idx] = average_precision[:, labels].mean(axis=1)
        mAR[:, key_idx] = average_recall[:, labels].mean(axis=1)

    return (
        average_precision,
        average_recall,
        mAP,
        mAR,
        precision_recall,
        pr_curve,
    )


def compute_detailed_pr_curve(
    data: np.ndarray,
    label_counts: np.ndarray,
    iou_thresholds: np.ndarray,
    score_thresholds: np.ndarray,
    n_samples: int,
) -> np.ndarray:

    """
    0  label
    1  tp
    ...
    2  fp - 1
    3  fp - 2
    4  fn - misclassification
    5  fn - hallucination
    """

    n_labels = label_counts.shape[0]
    n_ious = iou_thresholds.shape[0]
    n_scores = score_thresholds.shape[0]
    n_metrics = 5 * (n_samples + 1)

    tp_idx = 0
    fp_misclf_idx = tp_idx + n_samples + 1
    fp_halluc_idx = fp_misclf_idx + n_samples + 1
    fn_misclf_idx = fp_halluc_idx + n_samples + 1
    fn_misprd_idx = fn_misclf_idx + n_samples + 1

    detailed_pr_curve = np.ones((n_ious, n_scores, n_labels, n_metrics)) * -1.0

    mask_gt_exists = data[:, 1] > -0.5
    mask_pd_exists = data[:, 2] > -0.5
    mask_label_match = np.isclose(data[:, 4], data[:, 5])

    for iou_idx in range(n_ious):
        mask_iou = data[:, 3] >= iou_thresholds[iou_idx]

        for score_idx in range(n_scores):
            mask_score = data[:, 6] >= score_thresholds[score_idx]

            mask_tp = (
                mask_gt_exists
                & mask_pd_exists
                & mask_iou
                & mask_score
                & mask_label_match
            )
            mask_fp_misclf = (
                mask_gt_exists
                & mask_pd_exists
                & mask_iou
                & mask_score
                & ~mask_label_match
            )
            mask_fn_misclf = (
                mask_gt_exists
                & mask_pd_exists
                & mask_iou
                & ~mask_score
                & mask_label_match
            )
            mask_fp_halluc = (
                ~(mask_tp | mask_fp_misclf | mask_fn_misclf) & mask_pd_exists
            )
            mask_fn_misprd = (
                ~(mask_tp | mask_fp_misclf | mask_fn_misclf) & mask_gt_exists
            )

            tp_slice = data[mask_tp]
            fp_misclf_slice = data[mask_fp_misclf]
            fp_halluc_slice = data[mask_fp_halluc]
            fn_misclf_slice = data[mask_fn_misclf]
            fn_misprd_slice = data[mask_fn_misprd]

            tp_count = np.bincount(
                tp_slice[:, 5].astype(int), minlength=n_labels
            )
            fp_misclf_count = np.bincount(
                fp_misclf_slice[:, 5].astype(int), minlength=n_labels
            )
            fp_halluc_count = np.bincount(
                fp_halluc_slice[:, 5].astype(int), minlength=n_labels
            )
            fn_misclf_count = np.bincount(
                fn_misclf_slice[:, 4].astype(int), minlength=n_labels
            )
            fn_misprd_count = np.bincount(
                fn_misprd_slice[:, 4].astype(int), minlength=n_labels
            )

            detailed_pr_curve[iou_idx, score_idx, :, tp_idx] = tp_count
            detailed_pr_curve[
                iou_idx, score_idx, :, fp_misclf_idx
            ] = fp_misclf_count
            detailed_pr_curve[
                iou_idx, score_idx, :, fp_halluc_idx
            ] = fp_halluc_count
            detailed_pr_curve[
                iou_idx, score_idx, :, fn_misclf_idx
            ] = fn_misclf_count
            detailed_pr_curve[
                iou_idx, score_idx, :, fn_misprd_idx
            ] = fn_misprd_count

            if n_samples > 0:
                for label_idx in range(n_labels):
                    tp_examples = tp_slice[
                        tp_slice[:, 5].astype(int) == label_idx
                    ][:n_samples, 0]
                    fp_misclf_examples = fp_misclf_slice[
                        fp_misclf_slice[:, 5].astype(int) == label_idx
                    ][:n_samples, 0]
                    fp_halluc_examples = fp_halluc_slice[
                        fp_halluc_slice[:, 5].astype(int) == label_idx
                    ][:n_samples, 0]
                    fn_misclf_examples = fn_misclf_slice[
                        fn_misclf_slice[:, 4].astype(int) == label_idx
                    ][:n_samples, 0]
                    fn_misprd_examples = fn_misprd_slice[
                        fn_misprd_slice[:, 4].astype(int) == label_idx
                    ][:n_samples, 0]

                    detailed_pr_curve[
                        iou_idx,
                        score_idx,
                        label_idx,
                        tp_idx + 1 : tp_idx + 1 + tp_examples.shape[0],
                    ] = tp_examples
                    detailed_pr_curve[
                        iou_idx,
                        score_idx,
                        label_idx,
                        fp_misclf_idx
                        + 1 : fp_misclf_idx
                        + 1
                        + fp_misclf_examples.shape[0],
                    ] = fp_misclf_examples
                    detailed_pr_curve[
                        iou_idx,
                        score_idx,
                        label_idx,
                        fp_halluc_idx
                        + 1 : fp_halluc_idx
                        + 1
                        + fp_halluc_examples.shape[0],
                    ] = fp_halluc_examples
                    detailed_pr_curve[
                        iou_idx,
                        score_idx,
                        label_idx,
                        fn_misclf_idx
                        + 1 : fn_misclf_idx
                        + 1
                        + fn_misclf_examples.shape[0],
                    ] = fn_misclf_examples
                    detailed_pr_curve[
                        iou_idx,
                        score_idx,
                        label_idx,
                        fn_misprd_idx
                        + 1 : fn_misprd_idx
                        + 1
                        + fn_misprd_examples.shape[0],
                    ] = fn_misprd_examples

    return detailed_pr_curve