from collections import defaultdict

import numpy as np
from numpy.typing import NDArray

from valor_lite.object_detection.computation import PairClassification
from valor_lite.object_detection.metric import Metric, MetricType


def unpack_precision_recall_into_metric_lists(
    results: tuple[
        tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        tuple[
            NDArray[np.float64],
            NDArray[np.float64],
        ],
        NDArray[np.float64],
        NDArray[np.float64],
    ],
    iou_thresholds: list[float],
    score_thresholds: list[float],
    index_to_label: list[str],
    label_metadata: NDArray[np.int32],
):
    (
        (
            average_precision,
            mean_average_precision,
        ),
        (
            average_recall,
            mean_average_recall,
        ),
        precision_recall,
        pr_curves,
    ) = results

    metrics = defaultdict(list)

    metrics[MetricType.AP] = [
        Metric.average_precision(
            value=float(average_precision[iou_idx][label_idx]),
            iou_threshold=iou_threshold,
            label=label,
        )
        for iou_idx, iou_threshold in enumerate(iou_thresholds)
        for label_idx, label in enumerate(index_to_label)
        if int(label_metadata[label_idx, 0]) > 0
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
        for label_idx, label in enumerate(index_to_label)
        if int(label_metadata[label_idx, 0]) > 0
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
        for label_idx, label in enumerate(index_to_label)
        if int(label_metadata[label_idx, 0]) > 0
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
        for label_idx, label in enumerate(index_to_label)
        if int(label_metadata[label_idx, 0]) > 0
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
            precisions=pr_curves[iou_idx, label_idx, :, 0].tolist(),  # type: ignore[reportArgumentType]
            scores=pr_curves[iou_idx, label_idx, :, 1].tolist(),  # type: ignore[reportArgumentType]
            iou_threshold=iou_threshold,
            label=label,
        )
        for iou_idx, iou_threshold in enumerate(iou_thresholds)
        for label_idx, label in enumerate(index_to_label)
        if label_metadata[label_idx, 0] > 0
    ]

    for label_idx, label in enumerate(index_to_label):
        if label_metadata[label_idx, 0] == 0:
            continue

        for score_idx, score_threshold in enumerate(score_thresholds):
            for iou_idx, iou_threshold in enumerate(iou_thresholds):

                row = precision_recall[iou_idx, score_idx, label_idx, :]
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

                metrics[MetricType.Precision].append(
                    Metric.precision(
                        value=float(row[3]),
                        **kwargs,
                    )
                )
                metrics[MetricType.Recall].append(
                    Metric.recall(
                        value=float(row[4]),
                        **kwargs,
                    )
                )
                metrics[MetricType.F1].append(
                    Metric.f1_score(
                        value=float(row[5]),
                        **kwargs,
                    )
                )

    return metrics


def defaultdict_dict():
    return defaultdict(dict)


def _unpack_confusion_matrix(
    results: NDArray[np.uint8],
    detailed_pairs: NDArray[np.float64],
    index_to_datum_id: list[str],
    index_to_groundtruth_id: list[str],
    index_to_prediction_id: list[str],
    index_to_label: list[str],
    iou_threhsold: float,
    score_threshold: float,
):
    ids = detailed_pairs[:, :5].astype(np.int32)

    mask_matched = (
        np.bitwise_and(
            results, PairClassification.TP | PairClassification.FP_FN_MISCLF
        )
        > 0
    )
    mask_fp_unmatched = (
        np.bitwise_and(results, PairClassification.FP_UNMATCHED) > 0
    )
    mask_fn_unmatched = (
        np.bitwise_and(results, PairClassification.FN_UNMATCHED) > 0
    )

    unique_matches = np.unique(
        ids[np.ix_(mask_matched, (0, 1, 2, 3, 4))], axis=0  # type: ignore - numpy ix_ typing
    )
    unique_unmatched_predictions = np.unique(
        ids[np.ix_(mask_fp_unmatched, (0, 2, 4))], axis=0  # type: ignore - numpy ix_ typing
    )
    unique_unmatched_groundtruths = np.unique(
        ids[np.ix_(mask_fn_unmatched, (0, 1, 3))], axis=0  # type: ignore - numpy ix_ typing
    )

    confusion_matrix = dict()
    unmatched_ground_truths = dict()
    unmatched_predictions = dict()
    for label_idx, label in enumerate(index_to_label):
        unmatched_fn = unique_unmatched_groundtruths[
            unique_unmatched_groundtruths[:, 2] == label_idx
        ]
        unmatched_ground_truths[label] = {
            "count": unmatched_fn.shape[0],
            "examples": [
                {
                    "datum_id": index_to_datum_id[unmatched_fn[idx, 0]],
                    "ground_truth_id": index_to_groundtruth_id[
                        unmatched_fn[idx, 1]
                    ],
                }
                for idx in range(unmatched_fn.shape[0])
            ],
        }

        unmatched_fp = unique_unmatched_predictions[
            unique_unmatched_predictions[:, 2] == label_idx
        ]
        unmatched_predictions[label] = {
            "count": unmatched_fp.shape[0],
            "examples": [
                {
                    "datum_id": index_to_datum_id[unmatched_fp[idx, 0]],
                    "prediction_id": index_to_prediction_id[
                        unmatched_fp[idx, 1]
                    ],
                }
                for idx in range(unmatched_fp.shape[0])
            ],
        }

        confusion_matrix[label] = {}
        for plabel_idx, plabel in enumerate(index_to_label):
            mask_labels = np.all(
                unique_matches[:, (3, 4)] == np.array([label_idx, plabel_idx]),
                axis=1,
            )
            matches = unique_matches[mask_labels]
            confusion_matrix[label][plabel] = {
                "count": matches.shape[0],
                "examples": [
                    {
                        "datum_id": index_to_datum_id[matches[idx, 0]],
                        "ground_truth_id": index_to_groundtruth_id[
                            matches[idx, 1]
                        ],
                        "prediction_id": index_to_prediction_id[
                            matches[idx, 2]
                        ],
                    }
                    for idx in range(matches.shape[0])
                ],
            }

    return Metric.confusion_matrix(
        confusion_matrix=confusion_matrix,
        unmatched_ground_truths=unmatched_ground_truths,
        unmatched_predictions=unmatched_predictions,
        iou_threshold=iou_threhsold,
        score_threshold=score_threshold,
    )


def unpack_confusion_matrix_into_metric_list(
    results: NDArray[np.uint8],
    detailed_pairs: NDArray[np.float64],
    iou_thresholds: list[float],
    score_thresholds: list[float],
    index_to_datum_id: list[str],
    index_to_groundtruth_id: list[str],
    index_to_prediction_id: list[str],
    index_to_label: list[str],
) -> list[Metric]:
    return [
        _unpack_confusion_matrix(
            results=results[iou_idx, score_idx, :],
            detailed_pairs=detailed_pairs,
            index_to_datum_id=index_to_datum_id,
            index_to_groundtruth_id=index_to_groundtruth_id,
            index_to_prediction_id=index_to_prediction_id,
            index_to_label=index_to_label,
            iou_threhsold=iou_threshold,
            score_threshold=score_threshold,
        )
        for iou_idx, iou_threshold in enumerate(iou_thresholds)
        for score_idx, score_threshold in enumerate(score_thresholds)
        if (results[iou_idx, score_idx] != -1).any()
    ]
