from collections import defaultdict

import numpy as np
from numpy.typing import NDArray

from valor_lite.classification.computation import PairClassification
from valor_lite.classification.metric import Metric, MetricType


def unpack_precision_recall_rocauc_into_metric_lists(
    counts: NDArray[np.uint64],
    precision: NDArray[np.float64],
    recall: NDArray[np.float64],
    accuracy: NDArray[np.float64],
    f1_score: NDArray[np.float64],
    rocauc: NDArray[np.float64],
    mean_rocauc: float,
    score_thresholds: list[float],
    hardmax: bool,
    index_to_label: dict[int, str],
) -> dict[MetricType, list[Metric]]:

    metrics = defaultdict(list)

    metrics[MetricType.ROCAUC] = [
        Metric.roc_auc(
            value=float(rocauc[label_idx]),
            label=label,
        )
        for label_idx, label in index_to_label.items()
    ]

    metrics[MetricType.mROCAUC] = [
        Metric.mean_roc_auc(
            value=float(mean_rocauc),
        )
    ]

    metrics[MetricType.Accuracy] = [
        Metric.accuracy(
            value=float(accuracy[score_idx]),
            score_threshold=score_threshold,
            hardmax=hardmax,
        )
        for score_idx, score_threshold in enumerate(score_thresholds)
    ]

    for label_idx, label in index_to_label.items():
        for score_idx, score_threshold in enumerate(score_thresholds):

            kwargs = {
                "label": label,
                "hardmax": hardmax,
                "score_threshold": score_threshold,
            }
            row = counts[:, label_idx]
            metrics[MetricType.Counts].append(
                Metric.counts(
                    tp=int(row[score_idx, 0]),
                    fp=int(row[score_idx, 1]),
                    fn=int(row[score_idx, 2]),
                    tn=int(row[score_idx, 3]),
                    **kwargs,
                )
            )

            metrics[MetricType.Precision].append(
                Metric.precision(
                    value=float(precision[score_idx, label_idx]),
                    **kwargs,
                )
            )
            metrics[MetricType.Recall].append(
                Metric.recall(
                    value=float(recall[score_idx, label_idx]),
                    **kwargs,
                )
            )
            metrics[MetricType.F1].append(
                Metric.f1_score(
                    value=float(f1_score[score_idx, label_idx]),
                    **kwargs,
                )
            )
    return metrics


def unpack_confusion_matrix(
    confusion_matrices: NDArray[np.uint64],
    unmatched_groundtruths: NDArray[np.uint64],
    index_to_label: dict[int, str],
    iou_thresholds: list[float],
    score_thresholds: list[float],
) -> list[Metric]:
    metrics = []
    for iou_idx, iou_thresh in enumerate(iou_thresholds):
        for score_idx, score_thresh in enumerate(score_thresholds):
            cm_dict = {}
            ugt_dict = {}
            upd_dict = {}
            for idx, label in index_to_label.items():
                ugt_dict[label] = int(
                    unmatched_groundtruths[iou_idx, score_idx, idx]
                )
                upd_dict[label] = int(
                    unmatched_predictions[iou_idx, score_idx, idx]
                )
                for pidx, plabel in index_to_label.items():
                    if label not in cm_dict:
                        cm_dict[label] = {}
                    cm_dict[label][plabel] = int(
                        confusion_matrices[iou_idx, score_idx, idx, pidx]
                    )
            metrics.append(
                Metric.confusion_matrix(
                    confusion_matrix=cm_dict,
                    unmatched_ground_truths=ugt_dict,
                    unmatched_predictions=upd_dict,
                    iou_threshold=iou_thresh,
                    score_threshold=score_thresh,
                )
            )
    return metrics


def _create_empty_confusion_matrix_with_examples(index_to_labels: list[str]):
    unmatched_ground_truths = dict()
    confusion_matrix = dict()
    for label in index_to_labels:
        unmatched_ground_truths[label] = {"count": 0, "examples": []}
        confusion_matrix[label] = {}
        for plabel in index_to_labels:
            confusion_matrix[label][plabel] = {"count": 0, "examples": []}
    return (
        confusion_matrix,
        unmatched_ground_truths,
    )


def _unpack_confusion_matrix_with_examples(
    ids: NDArray[np.int32],
    scores: NDArray[np.float64],
    mask_matched: NDArray[np.bool_],
    mask_fn_unmatched: NDArray[np.bool_],
    index_to_datum_id: list[str],
    index_to_label: list[str],
    score_threshold: float,
):
    (
        confusion_matrix,
        unmatched_ground_truths,
    ) = _create_empty_confusion_matrix_with_examples(index_to_label)

    unique_matches, unique_match_indices = np.unique(
        ids[np.ix_(mask_matched, (0, 1, 2))],  # type: ignore - numpy ix_ typing
        axis=0,
        return_index=True,
    )
    (
        unique_unmatched_groundtruths,
        unique_unmatched_groundtruth_indices,
    ) = np.unique(
        ids[np.ix_(mask_fn_unmatched, (0, 1))],  # type: ignore - numpy ix_ typing
        axis=0,
        return_index=True,
    )

    n_matched = unique_matches.shape[0]
    n_unmatched_groundtruths = unique_unmatched_groundtruths.shape[0]
    n_max = max(n_matched, n_unmatched_groundtruths)

    for idx in range(n_max):
        if idx < n_matched:
            glabel = index_to_label[unique_matches[idx, 1]]
            plabel = index_to_label[unique_matches[idx, 2]]
            confusion_matrix[glabel][plabel]["count"] += 1
            confusion_matrix[glabel][plabel]["examples"].append(
                {
                    "datum_id": index_to_datum_id[unique_matches[idx, 0]],
                    "score": float(scores[unique_match_indices[idx]]),
                }
            )
        if idx < n_unmatched_groundtruths:
            label = index_to_label[unique_unmatched_groundtruths[idx, 1]]
            unmatched_ground_truths[label]["count"] += 1
            unmatched_ground_truths[label]["examples"].append(
                {
                    "datum_id": index_to_datum_id[
                        unique_unmatched_groundtruths[idx, 0]
                    ],
                }
            )

    return Metric.confusion_matrix(
        confusion_matrix=confusion_matrix,
        unmatched_ground_truths=unmatched_ground_truths,
        score_threshold=score_threshold,
    )


def unpack_confusion_matrix_with_examples(
    result: NDArray[np.uint8],
    detailed_pairs: NDArray[np.float64],
    score_thresholds: list[float],
    index_to_datum_id: list[str],
    index_to_label: list[str],
) -> list[Metric]:

    ids = detailed_pairs[:, :3].astype(np.int32)

    mask_matched = (
        np.bitwise_and(
            result, PairClassification.TP | PairClassification.FP_FN_MISCLF
        )
        > 0
    )
    mask_fn_unmatched = (
        np.bitwise_and(result, PairClassification.FN_UNMATCHED) > 0
    )

    return [
        _unpack_confusion_matrix_with_examples(
            ids=ids,
            scores=detailed_pairs[:, 3],
            mask_matched=mask_matched[score_idx, :],
            mask_fn_unmatched=mask_fn_unmatched[score_idx, :],
            index_to_datum_id=index_to_datum_id,
            index_to_label=index_to_label,
            score_threshold=score_threshold,
        )
        for score_idx, score_threshold in enumerate(score_thresholds)
    ]
