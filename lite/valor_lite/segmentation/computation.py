import numpy as np
from numpy.typing import NDArray


def compute_metrics(
    data: NDArray[np.float64],
    score_thresholds: NDArray[np.float64],
    label_metadata: NDArray[np.int32],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """
    Computes semantic segmenation metrics.

    Takes data with shape (3, N).

    Parameters
    ----------
    data : NDArray[np.float64]
        An containing segmentations.
    label_metadata : NDArray[np.int32]
        A 2-D array containing label metadata.
    score_threhsolds : NDArray[np.int32]
        A 1-D array containing score thresholds.

    Returns
    -------
    NDArray[np.float64]
        Precision.
    NDArray[np.float64]
        Recall.
    NDArray[np.float64]
        F1 Score.
    float
        Accuracy
    NDArray[np.float64]
        Confusion matrix containing IoU values.
    NDArray[np.float64]
        Missing prediction ratios.
    """
    n_labels = label_metadata.shape[0]
    n_scores = score_thresholds.shape[0]

    groundtruth_labels = data[:, 0].astype(int)
    prediction_labels = data[:, 1].astype(int)
    prediction_scores = data[:, 2]

    ious = np.zeros((n_scores, n_labels, n_labels), dtype=np.float64)
    missing_predictions = np.zeros((n_scores, n_labels), dtype=np.float64)

    precision = np.zeros((n_scores, n_labels), dtype=np.float64)
    recall = np.zeros_like(precision)
    f1_score = np.zeros_like(precision)
    accuracy = np.zeros((n_scores), dtype=np.float64)

    true_prediction_count = np.zeros_like(accuracy)
    total_prediciton_count = np.zeros_like(accuracy)

    for gt_label_idx in range(n_labels):

        gt_count = label_metadata[gt_label_idx, 0]
        mask_gt_label = groundtruth_labels == gt_label_idx

        for pd_label_idx in range(n_labels):

            mask_pd_label = prediction_labels == pd_label_idx
            print(mask_pd_label)

            for score_idx in range(n_scores):

                if score_thresholds[score_idx] > 1e-9:
                    mask_scores = (
                        prediction_scores >= score_thresholds[score_idx]
                    )
                else:
                    mask_scores = prediction_scores > 1e-9

                mask_scored_pd_label = mask_scores & mask_pd_label
                mask_intersection = mask_gt_label & mask_scored_pd_label

                pd_count = mask_scored_pd_label.sum()
                intersection_ = mask_intersection.sum()
                union_ = gt_count + pd_count - intersection_

                missing_predictions[score_idx, gt_label_idx] = (
                    (gt_count - intersection_) / gt_count
                    if gt_count > 1e-9
                    else 0.0
                )
                ious[score_idx, gt_label_idx, pd_label_idx] = (
                    intersection_ / union_ if union_ > 1e-9 else 0.0
                )

                if gt_label_idx == pd_label_idx:
                    true_prediction_count[score_idx] += intersection_
                    total_prediciton_count[score_idx] += pd_count
                    precision[score_idx, pd_label_idx] = (
                        (intersection_ / pd_count) if pd_count > 1e-9 else 0.0
                    )
                    recall[score_idx, gt_label_idx] = (
                        (intersection_ / gt_count) if gt_count > 1e-9 else 0.0
                    )

    np.divide(
        2 * (precision * recall),
        (precision + recall),
        where=(precision + recall) > 0,
        out=f1_score,
    )

    np.divide(
        true_prediction_count,
        total_prediciton_count,
        where=total_prediciton_count > 1e-9,
        out=accuracy,
    )

    return (
        precision,
        recall,
        f1_score,
        accuracy,
        ious,
        missing_predictions,
    )
