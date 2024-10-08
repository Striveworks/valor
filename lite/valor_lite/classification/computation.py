import numpy as np
from numpy.typing import NDArray


def _compute_rocauc(
    data: NDArray[np.float64],
    label_metadata: NDArray[np.int32],
    n_datums: int,
    n_labels: int,
    mask_matching_labels: NDArray[np.bool_],
    pd_labels: NDArray[np.int32],
):
    """
    Compute ROCAUC and mean ROCAUC.
    """
    positive_count = label_metadata[:, 0]
    negative_count = label_metadata[:, 1] - label_metadata[:, 0]

    true_positives = np.zeros((n_labels, n_datums), dtype=np.int32)
    false_positives = np.zeros_like(true_positives)
    scores = np.zeros_like(true_positives, dtype=np.float64)

    for label_idx in range(n_labels):
        if label_metadata[label_idx, 1] == 0:
            continue

        mask_pds = pd_labels == label_idx
        true_positives[label_idx] = mask_matching_labels[mask_pds]
        false_positives[label_idx] = ~mask_matching_labels[mask_pds]
        scores[label_idx] = data[mask_pds, 3]

    cumulative_fp = np.cumsum(false_positives, axis=1)
    cumulative_tp = np.cumsum(true_positives, axis=1)

    fpr = np.zeros_like(true_positives, dtype=np.float64)
    np.divide(
        cumulative_fp,
        negative_count[:, np.newaxis],
        where=negative_count[:, np.newaxis] > 1e-9,
        out=fpr,
    )
    tpr = np.zeros_like(true_positives, dtype=np.float64)
    np.divide(
        cumulative_tp,
        positive_count[:, np.newaxis],
        where=positive_count[:, np.newaxis] > 1e-9,
        out=tpr,
    )

    # sort by -tpr, -score
    indices = np.lexsort((-tpr, -scores), axis=1)
    fpr = np.take_along_axis(fpr, indices, axis=1)
    tpr = np.take_along_axis(tpr, indices, axis=1)

    # running max of tpr
    np.maximum.accumulate(tpr, axis=1, out=tpr)

    # compute rocauc
    rocauc = np.trapz(x=fpr, y=tpr, axis=1)  # type: ignore - numpy will be switching to `trapezoid` in the future.

    # compute mean rocauc
    mean_rocauc = rocauc.mean()

    return rocauc, mean_rocauc


def compute_metrics(
    data: NDArray[np.float64],
    label_metadata: NDArray[np.int32],
    score_thresholds: NDArray[np.float64],
    hardmax: bool,
    n_datums: int,
) -> tuple[
    NDArray[np.int32],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    float,
]:
    """
    Computes classification metrics.

    Takes data with shape (N, 5):

    Index 0 - Datum Index
    Index 1 - GroundTruth Label Index
    Index 2 - Prediction Label Index
    Index 3 - Score
    Index 4 - Hard-Max Score

    Parameters
    ----------
    data : NDArray[np.float64]
        A sorted array of classification pairs.
    label_metadata : NDArray[np.int32]
        An array containing metadata related to labels.
    score_thresholds : NDArray[np.float64]
        A 1-D array contains score thresholds to compute metrics over.
    hardmax : bool
        Option to only allow a single positive prediction.
    n_datums : int
        The number of datums being operated over.

    Returns
    -------
    NDArray[np.int32]
        TP, FP, FN, TN counts.
    NDArray[np.float64]
        Precision.
    NDArray[np.float64]
        Recall.
    NDArray[np.float64]
        Accuracy
    NDArray[np.float64]
        F1 Score
    NDArray[np.float64]
        ROCAUC.
    float
        mROCAUC.
    """

    n_labels = label_metadata.shape[0]
    n_scores = score_thresholds.shape[0]

    pd_labels = data[:, 2].astype(int)

    mask_matching_labels = np.isclose(data[:, 1], data[:, 2])
    mask_score_nonzero = ~np.isclose(data[:, 3], 0.0)
    mask_hardmax = data[:, 4] > 0.5

    # calculate ROCAUC
    rocauc, mean_rocauc = _compute_rocauc(
        data=data,
        label_metadata=label_metadata,
        n_datums=n_datums,
        n_labels=n_labels,
        mask_matching_labels=mask_matching_labels,
        pd_labels=pd_labels,
    )

    # calculate metrics at various score thresholds
    counts = np.zeros((n_scores, n_labels, 4), dtype=np.int32)
    for score_idx in range(n_scores):
        mask_score_threshold = data[:, 3] >= score_thresholds[score_idx]
        mask_score = mask_score_nonzero & mask_score_threshold

        if hardmax:
            mask_score &= mask_hardmax

        mask_tp = mask_matching_labels & mask_score
        mask_fp = ~mask_matching_labels & mask_score
        mask_fn = (mask_matching_labels & ~mask_score) | mask_fp
        mask_tn = ~mask_matching_labels & ~mask_score

        fn = np.unique(data[mask_fn][:, [0, 1]].astype(int), axis=0)
        tn = np.unique(data[mask_tn][:, [0, 2]].astype(int), axis=0)

        counts[score_idx, :, 0] = np.bincount(
            pd_labels[mask_tp], minlength=n_labels
        )
        counts[score_idx, :, 1] = np.bincount(
            pd_labels[mask_fp], minlength=n_labels
        )
        counts[score_idx, :, 2] = np.bincount(fn[:, 1], minlength=n_labels)
        counts[score_idx, :, 3] = np.bincount(tn[:, 1], minlength=n_labels)

    recall = np.zeros((n_scores, n_labels), dtype=np.float64)
    np.divide(
        counts[:, :, 0],
        (counts[:, :, 0] + counts[:, :, 2]),
        where=(counts[:, :, 0] + counts[:, :, 2]) > 1e-9,
        out=recall,
    )

    precision = np.zeros_like(recall)
    np.divide(
        counts[:, :, 0],
        (counts[:, :, 0] + counts[:, :, 1]),
        where=(counts[:, :, 0] + counts[:, :, 1]) > 1e-9,
        out=precision,
    )

    accuracy = np.zeros_like(recall)
    np.divide(
        (counts[:, :, 0] + counts[:, :, 3]),
        float(n_datums),
        out=accuracy,
    )

    f1_score = np.zeros_like(recall)
    np.divide(
        (2 * precision * recall),
        (precision + recall),
        where=(precision + recall) > 1e-9,
        out=f1_score,
    )

    return (
        counts,
        precision,
        recall,
        accuracy,
        f1_score,
        rocauc,
        mean_rocauc,
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
    score_thresholds: NDArray[np.float64],
    hardmax: bool,
    n_examples: int,
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """
    Compute detailed confusion matrix.

    Takes data with shape (N, 5):

    Index 0 - Datum Index
    Index 1 - GroundTruth Label Index
    Index 2 - Prediction Label Index
    Index 3 - Score
    Index 4 - Hard Max Score

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
    NDArray[np.int32]
        Ground truths with missing predictions.
    """

    n_labels = label_metadata.shape[0]
    n_scores = score_thresholds.shape[0]

    confusion_matrix = -1 * np.ones(
        (n_scores, n_labels, n_labels, 2 * n_examples + 1),
        dtype=np.float32,
    )
    missing_predictions = -1 * np.ones(
        (n_scores, n_labels, n_examples + 1),
        dtype=np.int32,
    )

    mask_label_match = np.isclose(data[:, 1], data[:, 2])
    mask_score = data[:, 3] > 1e-9

    groundtruths = data[:, [0, 1]].astype(int)

    for score_idx in range(n_scores):
        mask_score &= data[:, 3] >= score_thresholds[score_idx]
        if hardmax:
            mask_score &= data[:, 4] > 0.5

        mask_tp = mask_label_match & mask_score
        mask_misclf = ~mask_label_match & mask_score
        mask_misprd = ~(
            (
                groundtruths.reshape(-1, 1, 2)
                == groundtruths[mask_score].reshape(1, -1, 2)
            )
            .all(axis=2)
            .any(axis=1)
        )

        tp_examples, tp_labels, tp_counts = _count_with_examples(
            data=data[mask_tp],
            unique_idx=[0, 2],
            label_idx=1,
        )
        misclf_examples, misclf_labels, misclf_counts = _count_with_examples(
            data=data[mask_misclf],
            unique_idx=[0, 1, 2],
            label_idx=[1, 2],
        )
        misprd_examples, misprd_labels, misprd_counts = _count_with_examples(
            data=data[mask_misprd],
            unique_idx=[0, 1],
            label_idx=1,
        )

        confusion_matrix[score_idx, tp_labels, tp_labels, 0] = tp_counts
        confusion_matrix[
            score_idx, misclf_labels[:, 0], misclf_labels[:, 1], 0
        ] = misclf_counts

        missing_predictions[score_idx, misprd_labels, 0] = misprd_counts

        if n_examples > 0:
            for label_idx in range(n_labels):
                # true-positive examples
                mask_tp_label = tp_examples[:, 2] == label_idx
                if mask_tp_label.sum() > 0:
                    tp_label_examples = tp_examples[mask_tp_label][:n_examples]
                    confusion_matrix[
                        score_idx,
                        label_idx,
                        label_idx,
                        1 : 2 * tp_label_examples.shape[0] + 1,
                    ] = tp_label_examples[:, [0, 3]].flatten()

                # misclassification examples
                mask_misclf_gt_label = misclf_examples[:, 1] == label_idx
                if mask_misclf_gt_label.sum() > 0:
                    for pd_label_idx in range(n_labels):
                        mask_misclf_pd_label = (
                            misclf_examples[:, 2] == pd_label_idx
                        )
                        mask_misclf_label_combo = (
                            mask_misclf_gt_label & mask_misclf_pd_label
                        )
                        if mask_misclf_label_combo.sum() > 0:
                            misclf_label_examples = misclf_examples[
                                mask_misclf_label_combo
                            ][:n_examples]
                            confusion_matrix[
                                score_idx,
                                label_idx,
                                pd_label_idx,
                                1 : 2 * misclf_label_examples.shape[0] + 1,
                            ] = misclf_label_examples[:, [0, 3]].flatten()

                # missing prediction examples
                mask_misprd_label = misprd_examples[:, 1] == label_idx
                if misprd_examples.size > 0:
                    misprd_label_examples = misprd_examples[mask_misprd_label][
                        :n_examples
                    ]
                    missing_predictions[
                        score_idx,
                        label_idx,
                        1 : misprd_label_examples.shape[0] + 1,
                    ] = misprd_label_examples[:, 0].flatten()

    return confusion_matrix, missing_predictions
