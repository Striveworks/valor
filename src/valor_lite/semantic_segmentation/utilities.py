from collections import defaultdict

import numpy as np
from numpy.typing import NDArray

from valor_lite.semantic_segmentation.metric import Metric, MetricType


def unpack_precision_recall_iou_into_metric_lists(
    results: tuple,
    label_metadata: NDArray[np.int32],
    index_to_label: dict[int, str],
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
                    if label_metadata[pd_label_idx, 0] > 0
                }
                for gt_label_idx in range(n_labels)
                if label_metadata[gt_label_idx, 0] > 0
            },
            unmatched_predictions={
                index_to_label[pd_label_idx]: {
                    "ratio": float(unmatched_prediction_ratios[pd_label_idx])
                }
                for pd_label_idx in range(n_labels)
                if label_metadata[pd_label_idx, 0] > 0
            },
            unmatched_ground_truths={
                index_to_label[gt_label_idx]: {
                    "ratio": float(unmatched_ground_truth_ratios[gt_label_idx])
                }
                for gt_label_idx in range(n_labels)
                if label_metadata[gt_label_idx, 0] > 0
            },
        )
    ]

    metrics[MetricType.mIOU] = [
        Metric.mean_iou(
            value=float(ious.diagonal().mean()),
        )
    ]

    for label_idx, label in index_to_label.items():

        kwargs = {
            "label": label,
        }

        # if no groundtruths exists for a label, skip it.
        if label_metadata[label_idx, 0] == 0:
            continue

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
