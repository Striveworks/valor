from collections import defaultdict

import numpy as np
from numpy.typing import NDArray

from valor_lite.object_detection.computation import PairClassification
from valor_lite.object_detection.metric import Metric, MetricType


def unpack_precision_recall_into_metric_lists(
    counts: NDArray[np.uint64],
    precision_recall_f1: NDArray[np.float64],
    average_precision: NDArray[np.float64],
    mean_average_precision: NDArray[np.float64],
    average_recall: NDArray[np.float64],
    mean_average_recall: NDArray[np.float64],
    pr_curve: NDArray[np.float64],
    iou_thresholds: list[float],
    score_thresholds: list[float],
    index_to_label: dict[int, str],
):
    metrics = defaultdict(list)

    metrics[MetricType.AP] = [
        Metric.average_precision(
            value=float(average_precision[iou_idx][label_idx]),
            iou_threshold=iou_threshold,
            label=label,
        )
        for iou_idx, iou_threshold in enumerate(iou_thresholds)
        for label_idx, label in index_to_label.items()
    ]

    metrics[MetricType.mAP] = [
        Metric.mean_average_precision(
            value=float(mean_average_precision[iou_idx]),
            iou_threshold=iou_threshold,
        )
        for iou_idx, iou_threshold in enumerate(iou_thresholds)
    ]

    # TODO - (c.zaloom) will be removed in the future
    metrics[MetricType.APAveragedOverIOUs] = [
        Metric.average_precision_averaged_over_IOUs(
            value=float(average_precision.mean(axis=0)[label_idx]),
            iou_thresholds=iou_thresholds,
            label=label,
        )
        for label_idx, label in index_to_label.items()
    ]

    # TODO - (c.zaloom) will be removed in the future
    metrics[MetricType.mAPAveragedOverIOUs] = [
        Metric.mean_average_precision_averaged_over_IOUs(
            value=float(mean_average_precision.mean()),
            iou_thresholds=iou_thresholds,
        )
    ]

    metrics[MetricType.AR] = [
        Metric.average_recall(
            value=float(average_recall[score_idx, label_idx]),
            iou_thresholds=iou_thresholds,
            score_threshold=score_threshold,
            label=label,
        )
        for score_idx, score_threshold in enumerate(score_thresholds)
        for label_idx, label in index_to_label.items()
    ]

    metrics[MetricType.mAR] = [
        Metric.mean_average_recall(
            value=float(mean_average_recall[score_idx]),
            iou_thresholds=iou_thresholds,
            score_threshold=score_threshold,
        )
        for score_idx, score_threshold in enumerate(score_thresholds)
    ]

    # TODO - (c.zaloom) will be removed in the future
    metrics[MetricType.ARAveragedOverScores] = [
        Metric.average_recall_averaged_over_scores(
            value=float(average_recall.mean(axis=0)[label_idx]),
            score_thresholds=score_thresholds,
            iou_thresholds=iou_thresholds,
            label=label,
        )
        for label_idx, label in index_to_label.items()
    ]

    # TODO - (c.zaloom) will be removed in the future
    metrics[MetricType.mARAveragedOverScores] = [
        Metric.mean_average_recall_averaged_over_scores(
            value=float(mean_average_recall.mean()),
            score_thresholds=score_thresholds,
            iou_thresholds=iou_thresholds,
        )
    ]

    metrics[MetricType.PrecisionRecallCurve] = [
        Metric.precision_recall_curve(
            precisions=pr_curve[iou_idx, label_idx, :, 0].tolist(),
            scores=pr_curve[iou_idx, label_idx, :, 1].tolist(),
            iou_threshold=iou_threshold,
            label=label,
        )
        for iou_idx, iou_threshold in enumerate(iou_thresholds)
        for label_idx, label in index_to_label.items()
    ]

    for label_idx, label in index_to_label.items():
        for score_idx, score_threshold in enumerate(score_thresholds):
            for iou_idx, iou_threshold in enumerate(iou_thresholds):

                row = counts[iou_idx, score_idx, :, label_idx]
                kwargs = {
                    "label": label,
                    "iou_threshold": iou_threshold,
                    "score_threshold": score_threshold,
                }
                metrics[MetricType.Counts].append(
                    Metric.counts(
                        tp=int(row[0]),
                        fp=int(row[1]),
                        fn=int(row[2]),
                        **kwargs,
                    )
                )

                row = precision_recall_f1[iou_idx, score_idx, :, label_idx]
                metrics[MetricType.Precision].append(
                    Metric.precision(
                        value=float(row[0]),
                        **kwargs,
                    )
                )
                metrics[MetricType.Recall].append(
                    Metric.recall(
                        value=float(row[1]),
                        **kwargs,
                    )
                )
                metrics[MetricType.F1].append(
                    Metric.f1_score(
                        value=float(row[2]),
                        **kwargs,
                    )
                )

    return metrics


def _create_empty_confusion_matrix(index_to_labels: dict[int, str]):
    unmatched_ground_truths = dict()
    unmatched_predictions = dict()
    confusion_matrix = dict()
    for label in index_to_labels.values():
        unmatched_ground_truths[label] = {"count": 0, "examples": []}
        unmatched_predictions[label] = {"count": 0, "examples": []}
        confusion_matrix[label] = {}
        for plabel in index_to_labels.values():
            confusion_matrix[label][plabel] = {"count": 0, "examples": []}
    return (
        confusion_matrix,
        unmatched_predictions,
        unmatched_ground_truths,
    )


def _unpack_confusion_matrix_legacy(
    ids: NDArray[np.int32],
    mask_matched: NDArray[np.bool_],
    mask_fp_unmatched: NDArray[np.bool_],
    mask_fn_unmatched: NDArray[np.bool_],
    index_to_datum_id: dict[int, str],
    index_to_groundtruth_id: dict[int, str],
    index_to_prediction_id: dict[int, str],
    index_to_label: dict[int, str],
    iou_threhsold: float,
    score_threshold: float,
):
    (
        confusion_matrix,
        unmatched_predictions,
        unmatched_ground_truths,
    ) = _create_empty_confusion_matrix(index_to_label)

    unique_matches = np.unique(
        ids[np.ix_(mask_matched, (0, 1, 2, 3, 4))], axis=0  # type: ignore - numpy ix_ typing
    )
    unique_unmatched_predictions = np.unique(
        ids[np.ix_(mask_fp_unmatched, (0, 2, 4))], axis=0  # type: ignore - numpy ix_ typing
    )
    unique_unmatched_groundtruths = np.unique(
        ids[np.ix_(mask_fn_unmatched, (0, 1, 3))], axis=0  # type: ignore - numpy ix_ typing
    )

    n_matched = unique_matches.shape[0]
    n_unmatched_predictions = unique_unmatched_predictions.shape[0]
    n_unmatched_groundtruths = unique_unmatched_groundtruths.shape[0]
    n_max = max(n_matched, n_unmatched_groundtruths, n_unmatched_predictions)

    for idx in range(n_max):
        if idx < n_unmatched_groundtruths:
            label = index_to_label[unique_unmatched_groundtruths[idx, 2]]
            unmatched_ground_truths[label]["count"] += 1
            unmatched_ground_truths[label]["examples"].append(
                {
                    "datum_id": index_to_datum_id[
                        unique_unmatched_groundtruths[idx, 0]
                    ],
                    "ground_truth_id": index_to_groundtruth_id[
                        unique_unmatched_groundtruths[idx, 1]
                    ],
                }
            )
        if idx < n_unmatched_predictions:
            label_id = unique_unmatched_predictions[idx, 2]
            label = index_to_label[label_id]
            unmatched_predictions[label]["count"] += 1
            unmatched_predictions[label]["examples"].append(
                {
                    "datum_id": index_to_datum_id[
                        unique_unmatched_predictions[idx, 0]
                    ],
                    "prediction_id": index_to_prediction_id[
                        unique_unmatched_predictions[idx, 1]
                    ],
                }
            )
        if idx < n_matched:
            glabel = index_to_label[unique_matches[idx, 3]]
            plabel = index_to_label[unique_matches[idx, 4]]
            confusion_matrix[glabel][plabel]["count"] += 1
            confusion_matrix[glabel][plabel]["examples"].append(
                {
                    "datum_id": index_to_datum_id[unique_matches[idx, 0]],
                    "ground_truth_id": index_to_groundtruth_id[
                        unique_matches[idx, 1]
                    ],
                    "prediction_id": index_to_prediction_id[
                        unique_matches[idx, 2]
                    ],
                }
            )

    return Metric.confusion_matrix(
        confusion_matrix=confusion_matrix,
        unmatched_ground_truths=unmatched_ground_truths,
        unmatched_predictions=unmatched_predictions,
        iou_threshold=iou_threhsold,
        score_threshold=score_threshold,
    )


def unpack_confusion_matrix_into_metric_list_legacy(
    detailed_pairs: NDArray[np.float64],
    mask_tp: NDArray[np.bool_],
    mask_fp_fn_misclf: NDArray[np.bool_],
    mask_fp_unmatched: NDArray[np.bool_],
    mask_fn_unmatched: NDArray[np.bool_],
    iou_thresholds: list[float],
    score_thresholds: list[float],
    index_to_datum_id: dict[int, str],
    index_to_groundtruth_id: dict[int, str],
    index_to_prediction_id: dict[int, str],
    index_to_label: dict[int, str],
) -> list[Metric]:

    ids = detailed_pairs[:, :5].astype(np.int32)

    mask_matched = mask_tp | mask_fp_fn_misclf

    return [
        _unpack_confusion_matrix_legacy(
            ids=ids,
            mask_matched=mask_matched[iou_idx, score_idx],
            mask_fp_unmatched=mask_fp_unmatched[iou_idx, score_idx],
            mask_fn_unmatched=mask_fn_unmatched[iou_idx, score_idx],
            index_to_datum_id=index_to_datum_id,
            index_to_groundtruth_id=index_to_groundtruth_id,
            index_to_prediction_id=index_to_prediction_id,
            index_to_label=index_to_label,
            iou_threhsold=iou_threshold,
            score_threshold=score_threshold,
        )
        for iou_idx, iou_threshold in enumerate(iou_thresholds)
        for score_idx, score_threshold in enumerate(score_thresholds)
    ]


def unpack_confusion_matrix(
    confusion_matrices: NDArray[np.uint64],
    unmatched_groundtruths: NDArray[np.uint64],
    unmatched_predictions: NDArray[np.uint64],
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


def unpack_examples(
    detailed_pairs: NDArray[np.float64],
    mask_tp: NDArray[np.bool_],
    mask_fn: NDArray[np.bool_],
    mask_fp: NDArray[np.bool_],
    iou_thresholds: list[float],
    score_thresholds: list[float],
    index_to_datum_id: dict[int, str],
    index_to_groundtruth_id: dict[int, str],
    index_to_prediction_id: dict[int, str],
) -> list[Metric]:
    metrics = []
    ids = detailed_pairs[:, :5].astype(np.int64)
    unique_datums = np.unique(detailed_pairs[:, 0].astype(np.int64))
    for datum_index in unique_datums:
        mask_datum = detailed_pairs[:, 0] == datum_index
        mask_datum_tp = mask_tp & mask_datum
        mask_datum_fp = mask_fp & mask_datum
        mask_datum_fn = mask_fn & mask_datum

        datum_id = index_to_datum_id[datum_index]
        for iou_idx, iou_thresh in enumerate(iou_thresholds):
            for score_idx, score_thresh in enumerate(score_thresholds):

                unique_tp = np.unique(
                    ids[np.ix_(mask_datum_tp[iou_idx, score_idx], (0, 1, 2, 3, 4))], axis=0  # type: ignore - numpy ix_ typing
                )
                unique_fp = np.unique(
                    ids[np.ix_(mask_datum_fp[iou_idx, score_idx], (0, 2, 4))], axis=0  # type: ignore - numpy ix_ typing
                )
                unique_fn = np.unique(
                    ids[np.ix_(mask_datum_fn[iou_idx, score_idx], (0, 1, 3))], axis=0  # type: ignore - numpy ix_ typing
                )

                tp = [
                    (
                        index_to_groundtruth_id[row[1]],
                        index_to_prediction_id[row[2]],
                    )
                    for row in unique_tp
                ]
                fp = [index_to_prediction_id[row[1]] for row in unique_fp]
                fn = [index_to_groundtruth_id[row[1]] for row in unique_fn]
                metrics.append(
                    Metric.examples(
                        datum_id=datum_id,
                        true_positives=tp,
                        false_negatives=fn,
                        false_positives=fp,
                        iou_threshold=iou_thresh,
                        score_threshold=score_thresh,
                    )
                )
    return metrics
