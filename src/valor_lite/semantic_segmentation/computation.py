from typing import Any

import numpy as np
from numpy.typing import NDArray


def compute_intermediates(
    groundtruths: NDArray[np.integer[Any]],
    predictions: NDArray[np.integer[Any]],
    n_labels: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.uint64]]:
    """Count observed pairs in validated, equally shaped label maps.

    Returns local ground truth indices, prediction indices and pixel counts.
    Arithmetic is promoted before encoding pairs, including label ID 65535.
    Only observed pairs are stored; no class-by-class matrix is allocated.
    """
    pairs = groundtruths.ravel().astype(np.int64) * n_labels
    pairs += predictions.ravel().astype(np.int64)
    pairs, counts = np.unique(pairs, return_counts=True)
    return pairs // n_labels, pairs % n_labels, counts.astype(np.uint64)


def compute_metrics(
    confusion_matrix: NDArray[np.uint64],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    float,
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """
    Computes semantic segmentation metrics.

    Parameters
    ----------
    counts : NDArray[np.uint64]
        A 2-D confusion matrix with shape (n_labels + 1, n_labels + 1).

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
        Confusion matrix containing IOU values.
    NDArray[np.float64]
        Unmatched prediction ratios.
    NDArray[np.float64]
        Unmatched ground truth ratios.
    """
    n_labels = confusion_matrix.shape[0] - 1
    n_pixels = confusion_matrix.sum()
    gt_counts = confusion_matrix[1:, :].sum(axis=1)
    pd_counts = confusion_matrix[:, 1:].sum(axis=0)

    # compute iou, unmatched_ground_truth and unmatched predictions
    intersection_ = confusion_matrix[1:, 1:]
    union_ = (
        gt_counts[:, np.newaxis] + pd_counts[np.newaxis, :] - intersection_
    )

    ious = np.zeros((n_labels, n_labels), dtype=np.float64)
    np.divide(
        intersection_,
        union_,
        where=union_ > 1e-9,
        out=ious,
    )

    unmatched_prediction_ratio = np.zeros((n_labels), dtype=np.float64)
    np.divide(
        confusion_matrix[0, 1:],
        pd_counts,
        where=pd_counts > 1e-9,
        out=unmatched_prediction_ratio,
    )

    unmatched_ground_truth_ratio = np.zeros((n_labels), dtype=np.float64)
    np.divide(
        confusion_matrix[1:, 0],
        gt_counts,
        where=gt_counts > 1e-9,
        out=unmatched_ground_truth_ratio,
    )

    # compute precision, recall, f1
    tp_counts = confusion_matrix.diagonal()[1:]

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

    # compute accuracy
    tp_count = confusion_matrix[1:, 1:].diagonal().sum()
    # Preserve implicit-background accuracy for legacy caches. New input
    # class zero is an ordinary named class, stored in the main matrix.
    legacy_background_count = confusion_matrix[0, 0]
    accuracy = (
        (tp_count + legacy_background_count) / n_pixels
        if n_pixels > 0
        else 0.0
    )

    return (
        precision,
        recall,
        f1_score,
        accuracy,
        ious,
        unmatched_prediction_ratio,
        unmatched_ground_truth_ratio,
    )
