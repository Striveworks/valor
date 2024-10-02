import numpy as np
from numpy.typing import NDArray


def compute_metrics(
    data: NDArray[np.floating],
    score_thresholds: NDArray[np.floating],
    label_metadata: NDArray[np.int32],
) -> tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
]:
    """
    Computes semantic segmenation metrics.

    Takes data with shape (3, N).

    Parameters
    ----------
    data : NDArray[np.floating]
        An containing segmentations.
    label_metadata : NDArray[np.int32]
        A 2-D array containing label metadata.
    score_threhsolds : NDArray[np.int32]
        A 1-D array containing score thresholds.

    Returns
    -------
    NDArray[np.floating]
        Precision.
    NDArray[np.floating]
        Recall.
    NDArray[np.floating]
        F1 Score.
    float
        Accuracy
    NDArray[np.floating]
        Confusion matrix containing IoU values.
    NDArray[np.floating]
        Missing prediction ratios.
    """
    n_labels = label_metadata.shape[0]
    n_scores = score_thresholds.shape[0]

    groundtruth_labels = data[:, 0].astype(int)
    prediction_labels = data[:, 1].astype(int)
    prediction_scores = data[:, 2]

    ious = np.zeros((n_scores, n_labels, n_labels), dtype=np.floating)
    missing_predictions = np.zeros((n_scores, n_labels), dtype=np.floating)

    precision = np.zeros((n_scores, n_labels), dtype=np.floating)
    recall = np.zeros((n_scores, n_labels), dtype=np.floating)
    f1_score = np.zeros((n_scores, n_labels), dtype=np.floating)
    accuracy = np.zeros((n_scores), dtype=np.floating)

    mask_scores = prediction_scores > 0.0
    for gt_label_idx in range(n_labels):
        gt_count = label_metadata[gt_label_idx, 0]
        mask_gt_label = groundtruth_labels == gt_label_idx
        for pd_label_idx in range(n_labels):
            mask_pd_label = prediction_labels == pd_label_idx
            mask_intersection = mask_gt_label & mask_pd_label
            for score_idx in range(n_scores):
                mask_scores[mask_scores] = (
                    prediction_scores[mask_scores]
                    > score_thresholds[score_idx]
                )
                mask_scored_pd_label = mask_scores & mask_pd_label
                pd_count = mask_scored_pd_label.sum()

                intersection_ = mask_intersection[mask_scores].sum()
                union_ = gt_count + pd_count - intersection_

                missing_predictions[score_idx, gt_label_idx] = (
                    union_ - pd_count
                ) / union_
                ious[score_idx, gt_label_idx, pd_label_idx] = (
                    intersection_ / union_ if union_ > 1e-9 else 0.0
                )

                if gt_label_idx == pd_label_idx:
                    accuracy[score_idx] += intersection_
                    precision[score_idx, gt_label_idx] = (
                        intersection_ / pd_count
                    )
                    recall[score_idx, gt_label_idx] = intersection_ / gt_count

    f1_score = np.zeros_like(precision)
    np.divide(
        2 * (precision * recall),
        (precision + recall),
        where=(precision + recall) > 0,
        out=f1_score,
    )

    total_prediction_count = label_metadata[:, 1].sum()
    accuracy = accuracy / total_prediction_count

    return (
        precision,
        recall,
        f1_score,
        accuracy,
        ious,
        missing_predictions,
    )
