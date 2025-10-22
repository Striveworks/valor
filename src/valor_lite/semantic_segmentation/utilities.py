from collections import defaultdict

import numpy as np
from numpy.typing import NDArray

from valor_lite.semantic_segmentation.metric import Metric, MetricType


def unpack_precision_recall_iou_into_metric_lists(
    results: tuple,
    index_to_label: list[str],
) -> dict[MetricType, list[Metric]]:

    n_labels = len(index_to_label)
    (
        precision,
        recall,
        f1_score,
        accuracy,
        ious,
        unmatched_prediction_ratios,
        unmatched_ground_truth_ratios,
    ) = results

    metrics = defaultdict(list)

    metrics[MetricType.Accuracy] = [
        Metric.accuracy(
            value=float(accuracy),
        )
    ]

    metrics[MetricType.ConfusionMatrix] = [
        Metric.confusion_matrix(
            confusion_matrix={
                index_to_label[gt_label_idx]: {
                    index_to_label[pd_label_idx]: {
                        "iou": float(ious[gt_label_idx, pd_label_idx])
                    }
                    for pd_label_idx in range(n_labels)
                }
                for gt_label_idx in range(n_labels)
            },
            unmatched_predictions={
                index_to_label[pd_label_idx]: {
                    "ratio": float(unmatched_prediction_ratios[pd_label_idx])
                }
                for pd_label_idx in range(n_labels)
            },
            unmatched_ground_truths={
                index_to_label[gt_label_idx]: {
                    "ratio": float(unmatched_ground_truth_ratios[gt_label_idx])
                }
                for gt_label_idx in range(n_labels)
            },
        )
    ]

    metrics[MetricType.mIOU] = [
        Metric.mean_iou(
            value=float(ious.diagonal().mean()),
        )
    ]

    for label_idx, label in enumerate(index_to_label):

        kwargs = {
            "label": label,
        }

        metrics[MetricType.Precision].append(
            Metric.precision(
                value=float(precision[label_idx]),
                **kwargs,
            )
        )
        metrics[MetricType.Recall].append(
            Metric.recall(
                value=float(recall[label_idx]),
                **kwargs,
            )
        )
        metrics[MetricType.F1].append(
            Metric.f1_score(
                value=float(f1_score[label_idx]),
                **kwargs,
            )
        )
        metrics[MetricType.IOU].append(
            Metric.iou(
                value=float(ious[label_idx, label_idx]),
                **kwargs,
            )
        )

    return metrics
