import numpy as np
from numpy.typing import NDArray


def compute_label_metadata(
    confusion_matrices: NDArray[np.int64],
    n_labels: int,
) -> NDArray[np.int64]:
    """
    Computes label metadata returning a count of annotations per label.

    Parameters
    ----------
    confusion_matrices : NDArray[np.int64]
        Confusion matrices per datum with shape (n_datums, n_labels + 1, n_labels + 1).
    n_labels : int
        The total number of unique labels.

    Returns
    -------
    NDArray[np.int64]
        The label metadata array with shape (n_labels, 2).
            Index 0 - Ground truth label count
            Index 1 - Prediction label count
    """
    label_metadata = np.zeros((n_labels, 2), dtype=np.int64)
    label_metadata[:, 0] = confusion_matrices[:, 1:, :].sum(axis=(0, 2))
    label_metadata[:, 1] = confusion_matrices[:, :, 1:].sum(axis=(0, 1))
    return label_metadata


def filter_cache(
    confusion_matrices: NDArray[np.int64],
    datum_mask: NDArray[np.bool_],
    label_mask: NDArray[np.bool_],
    number_of_labels: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """
    Performs the filter operation over the internal cache.

    Parameters
    ----------
    confusion_matrices : NDArray[int64]
        The internal evaluator cache.
    datum_mask : NDArray[bool]
        A mask that filters out datums.
    datum_mask : NDArray[bool]
        A mask that filters out labels.

    Returns
    -------
    NDArray[int64]
        Filtered confusion matrices.
    NDArray[int64]
        Filtered label metadata.
    """
    if label_mask.any():
        # add filtered labels to background
        null_predictions = confusion_matrices[:, label_mask, :].sum(
            axis=(1, 2)
        )
        null_groundtruths = confusion_matrices[:, :, label_mask].sum(
            axis=(1, 2)
        )
        null_intersection = (
            confusion_matrices[:, label_mask, label_mask]
            .reshape(confusion_matrices.shape[0], -1)
            .sum(axis=1)
        )
        confusion_matrices[:, 0, 0] += (
            null_groundtruths + null_predictions - null_intersection
        )
        confusion_matrices[:, label_mask, :] = 0
        confusion_matrices[:, :, label_mask] = 0

    confusion_matrices = confusion_matrices[datum_mask]

    label_metadata = compute_label_metadata(
        confusion_matrices=confusion_matrices,
        n_labels=number_of_labels,
    )
    return confusion_matrices, label_metadata


def compute_intermediate_confusion_matrices(
    groundtruths: NDArray[np.bool_],
    predictions: NDArray[np.bool_],
    groundtruth_labels: NDArray[np.int64],
    prediction_labels: NDArray[np.int64],
    n_labels: int,
) -> NDArray[np.int64]:
    """
    Computes an intermediate confusion matrix containing label counts.

    Parameters
    ----------
    groundtruths : NDArray[np.bool_]
        A 2-D array containing flattened bitmasks for each label.
    predictions : NDArray[np.bool_]
        A 2-D array containing flattened bitmasks for each label.
    groundtruth_labels : NDArray[np.int64]
        A 1-D array containing label indices.
    groundtruth_labels : NDArray[np.int64]
        A 1-D array containing label indices.
    n_labels : int
        The number of unique labels.

    Returns
    -------
    NDArray[np.int64]
        A 2-D confusion matrix with shape (n_labels + 1, n_labels + 1).
    """

    groundtruth_counts = groundtruths.sum(axis=1)
    prediction_counts = predictions.sum(axis=1)

    background_counts = np.logical_not(
        groundtruths.any(axis=0) | predictions.any(axis=0)
    ).sum()

    intersection_counts = np.logical_and(
        groundtruths[:, None, :],
        predictions[None, :, :],
    ).sum(axis=2)
    intersected_groundtruth_counts = intersection_counts.sum(axis=1)
    intersected_prediction_counts = intersection_counts.sum(axis=0)

    confusion_matrix = np.zeros((n_labels + 1, n_labels + 1), dtype=np.int64)
    confusion_matrix[0, 0] = background_counts
    confusion_matrix[
        np.ix_(groundtruth_labels + 1, prediction_labels + 1)
    ] = intersection_counts
    confusion_matrix[0, prediction_labels + 1] = (
        prediction_counts - intersected_prediction_counts
    )
    confusion_matrix[groundtruth_labels + 1, 0] = (
        groundtruth_counts - intersected_groundtruth_counts
    )

    return confusion_matrix


def compute_metrics(
    confusion_matrices: NDArray[np.int64],
    label_metadata: NDArray[np.int64],
    n_pixels: int,
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

    Takes data with shape (3, N).

    Parameters
    ----------
    confusion_matrices : NDArray[np.int64]
        A 3-D array containing confusion matrices for each datum with shape (n_datums, n_labels + 1, n_labels + 1).
    label_metadata : NDArray[np.int64]
        A 2-D array containing label metadata with shape (n_labels, 2).
            Index 0: Ground Truth Label Count
            Index 1: Prediction Label Count

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
    n_labels = label_metadata.shape[0]
    gt_counts = label_metadata[:, 0]
    pd_counts = label_metadata[:, 1]

    counts = confusion_matrices.sum(axis=0)

    # compute iou, unmatched_ground_truth and unmatched predictions
    intersection_ = counts[1:, 1:]
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
        counts[0, 1:],
        pd_counts,
        where=pd_counts > 1e-9,
        out=unmatched_prediction_ratio,
    )

    unmatched_ground_truth_ratio = np.zeros((n_labels), dtype=np.float64)
    np.divide(
        counts[1:, 0],
        gt_counts,
        where=gt_counts > 1e-9,
        out=unmatched_ground_truth_ratio,
    )

    # compute precision, recall, f1
    tp_counts = counts.diagonal()[1:]

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
    tp_count = counts[1:, 1:].diagonal().sum()
    background_count = counts[0, 0]
    accuracy = (
        (tp_count + background_count) / n_pixels if n_pixels > 0 else 0.0
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
