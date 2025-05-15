from collections import defaultdict
from typing import Iterator

import numpy as np
from numpy.typing import NDArray

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
    results: NDArray[np.int32],
    detailed_pairs: NDArray[np.float64],
    n_labels: int,
    index_to_datum_id: list[str],
    index_to_groundtruth_id: list[str],
    index_to_prediction_id: list[str],
    index_to_label: list[str],
    iou_idx: int,
    iou_threhsold: float,
    score_idx: int,
    score_threshold: float,
):
    def get_position_encoding(gt_label_idx: int, pd_label_idx: int) -> int:
        n_positions = n_labels + 1
        gt_label_idx += 1
        pd_label_idx += 1
        return (pd_label_idx * n_positions) + gt_label_idx

    def get_counts(gt_label_idx: int, pd_label_idx: int) -> int:
        key = get_position_encoding(
            gt_label_idx=gt_label_idx, pd_label_idx=pd_label_idx
        )
        return int((results[iou_idx, score_idx, :] == key).sum())

    def get_example_indices(
        gt_label_idx: int, pd_label_idx: int
    ) -> Iterator[tuple[int, int, int]]:
        key = get_position_encoding(
            gt_label_idx=gt_label_idx, pd_label_idx=pd_label_idx
        )
        mask = results[iou_idx, score_idx, :] == key
        if not mask.any():
            return
        indices = np.where(mask)[0]
        pairs = detailed_pairs[np.ix_(indices, (0, 1, 2))]  # type: ignore - numpy typing being annoying
        for idx in range(pairs.shape[0]):
            yield tuple(pairs[idx].astype(int).tolist())

    confusion_matrix = dict()
    unmatched_ground_truths = dict()
    unmatched_predictions = dict()

    for label_idx, label in enumerate(index_to_label):
        # unmatched ground truths
        unmatched_ground_truths[label] = {
            "counts": get_counts(
                gt_label_idx=label_idx,
                pd_label_idx=-1,
            ),
            "examples": [
                {
                    "datum_id": index_to_datum_id[datum_id],
                    "ground_truth_id": index_to_groundtruth_id[gt_id],
                }
                for datum_id, gt_id, _ in get_example_indices(
                    gt_label_idx=label_idx,
                    pd_label_idx=-1,
                )
            ],
        }

        # unmatched predictions
        unmatched_predictions[label] = {
            "counts": get_counts(
                gt_label_idx=-1,
                pd_label_idx=label_idx,
            ),
            "examples": [
                {
                    "datum_id": index_to_datum_id[datum_id],
                    "prediction_id": index_to_prediction_id[pd_id],
                }
                for datum_id, pd_id, _ in get_example_indices(
                    gt_label_idx=-1,
                    pd_label_idx=label_idx,
                )
            ],
        }

        # confusion matrix
        confusion_matrix[label] = {}
        for pd_label_idx, pd_label in enumerate(index_to_label):
            gt_label_idx = label_idx
            gt_label = label
            confusion_matrix[gt_label][pd_label] = {
                "counts": get_counts(
                    gt_label_idx=gt_label_idx,
                    pd_label_idx=pd_label_idx,
                ),
                "examples": [
                    {
                        "datum_id": index_to_datum_id[datum_id],
                        "ground_truth_id": index_to_groundtruth_id[gt_id],
                        "prediction_id": index_to_prediction_id[pd_id],
                    }
                    for datum_id, gt_id, pd_id in get_example_indices(
                        gt_label_idx=gt_label_idx,
                        pd_label_idx=pd_label_idx,
                    )
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
    results: NDArray[np.int32],
    detailed_pairs: NDArray[np.float64],
    iou_thresholds: list[float],
    score_thresholds: list[float],
    index_to_datum_id: list[str],
    index_to_groundtruth_id: list[str],
    index_to_prediction_id: list[str],
    index_to_label: list[str],
) -> list[Metric]:
    n_labels = len(index_to_label)
    return [
        _unpack_confusion_matrix(
            results=results,
            detailed_pairs=detailed_pairs,
            n_labels=n_labels,
            index_to_datum_id=index_to_datum_id,
            index_to_groundtruth_id=index_to_groundtruth_id,
            index_to_prediction_id=index_to_prediction_id,
            index_to_label=index_to_label,
            iou_idx=iou_idx,
            iou_threhsold=iou_threshold,
            score_idx=score_idx,
            score_threshold=score_threshold,
        )
        for iou_idx, iou_threshold in enumerate(iou_thresholds)
        for score_idx, score_threshold in enumerate(score_thresholds)
        if (results[iou_idx, score_idx] != -1).any()
    ]
