import numpy as np
from numpy.typing import NDArray


def compute_confusion_matrix(
    groundtruths: NDArray[np.int32],
    predictions: NDArray[np.int32],
    n_labels: int,
) -> NDArray[np.int32]:
    """
    Computes confusion matrix containing label counts.

    Parameters
    ----------
    groundtruths : NDArray[np.int32]
        A 2-D array containing labeled pixels.
    predictions : NDArray[np.int32]
        A 2-D array containing labeled pixels.
    n_labels : int
        The number of unique labels.

    Returns
    -------
    NDArray[np.int32]
        A 2-D confusion matrix with shape (n_labels + 1, n_labels + 1).
    """

    confusion_matrix = np.zeros((n_labels + 1, n_labels + 1), dtype=np.int32)

    mask_no_groundtruth = groundtruths == -1
    mask_no_predictions = predictions == -1

    for gt_label_idx in range(n_labels):
        mask_groundtruths = groundtruths == gt_label_idx
        for pd_label_idx in range(n_labels):
            mask_predictions = predictions == pd_label_idx
            confusion_matrix[gt_label_idx + 1, pd_label_idx + 1] = (
                mask_groundtruths & mask_predictions
            ).sum()
            if gt_label_idx == 0:
                confusion_matrix[0, pd_label_idx + 1] = (
                    mask_no_groundtruth & mask_predictions
                ).sum()
        confusion_matrix[gt_label_idx + 1, 0] = (
            mask_no_predictions & mask_groundtruths
        ).sum()

    return confusion_matrix


def compute_metrics(
    data: NDArray[np.float64],
    label_metadata: NDArray[np.int32],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    float,
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

    counts = data.sum(axis=0)

    ious = np.zeros((n_labels, n_labels), dtype=np.float64)
    missing_predictions = np.zeros((n_labels), dtype=np.float64)

    true_prediction_count = 0
    total_prediciton_count = 0

    gt_counts = counts.sum(axis=0)[1:]
    pd_counts = counts.sum(axis=1)[1:]
    tp_counts = counts.diagonal()[1:]

    for gt_label_idx in range(n_labels):
        for pd_label_idx in range(n_labels):
            intersection_ = counts[gt_label_idx + 1, pd_label_idx + 1]
            union_ = (
                gt_counts[gt_label_idx]
                + pd_counts[pd_label_idx]
                - intersection_
            )
            ious[gt_label_idx, pd_label_idx] = (
                intersection_ / union_ if union_ > 1e-9 else 0.0
            )

    precision = np.zeros(n_labels, dtype=np.float64)
    np.divide(tp_counts, pd_counts, where=pd_counts > 1e-9, out=precision)

    recall = np.zeros_like(precision)
    np.divide(tp_counts, gt_counts, where=gt_counts > 1e-9, out=recall)

    f1_score = np.zeros_like(precision)
    np.divide(
        2 * (precision * recall),
        (precision + recall),
        where=(precision + recall) > 0,
        out=f1_score,
    )

    accuracy = (
        (true_prediction_count / total_prediciton_count)
        if total_prediciton_count > 0
        else 0.0
    )

    return (
        precision,
        recall,
        f1_score,
        accuracy,
        ious,
        missing_predictions,
    )
