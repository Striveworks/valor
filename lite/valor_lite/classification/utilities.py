from collections import defaultdict

import numpy as np
from numpy.typing import NDArray
from valor_lite.classification.metric import Metric, MetricType


def unpack_precision_recall_rocauc_into_metric_lists(
    results: tuple[
        NDArray[np.int32],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        float,
    ],
    score_thresholds: list[float],
    hardmax: bool,
    label_metadata: NDArray[np.int32],
    index_to_label: dict[int, str],
) -> dict[MetricType, list[Metric]]:
    (
        counts,
        precision,
        recall,
        accuracy,
        f1_score,
        rocauc,
        mean_rocauc,
    ) = results

    metrics = defaultdict(list)

    metrics[MetricType.ROCAUC] = [
        Metric.roc_auc(
            value=float(rocauc[label_idx]),
            label=label,
        )
        for label_idx, label in index_to_label.items()
        if label_metadata[label_idx, 0] > 0
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

            # if no groundtruths exists for a label, skip it.
            if label_metadata[label_idx, 0] == 0:
                continue

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


def _unpack_confusion_matrix_value(
    confusion_matrix: NDArray[np.float64],
    number_of_labels: int,
    number_of_examples: int,
    index_to_uid: dict[int, str],
    index_to_label: dict[int, str],
) -> dict[str, dict[str, dict[str, int | list[dict[str, str | float]]]]]:
    """
    Unpacks a numpy array of confusion matrix counts and examples.
    """

    datum_idx = lambda gt_label_idx, pd_label_idx, example_idx: int(  # noqa: E731 - lambda fn
        confusion_matrix[
            gt_label_idx,
            pd_label_idx,
            example_idx * 2 + 1,
        ]
    )

    score_idx = lambda gt_label_idx, pd_label_idx, example_idx: float(  # noqa: E731 - lambda fn
        confusion_matrix[
            gt_label_idx,
            pd_label_idx,
            example_idx * 2 + 2,
        ]
    )

    return {
        index_to_label[gt_label_idx]: {
            index_to_label[pd_label_idx]: {
                "count": max(
                    int(confusion_matrix[gt_label_idx, pd_label_idx, 0]),
                    0,
                ),
                "examples": [
                    {
                        "datum": index_to_uid[
                            datum_idx(gt_label_idx, pd_label_idx, example_idx)
                        ],
                        "score": score_idx(
                            gt_label_idx, pd_label_idx, example_idx
                        ),
                    }
                    for example_idx in range(number_of_examples)
                    if datum_idx(gt_label_idx, pd_label_idx, example_idx) >= 0
                ],
            }
            for pd_label_idx in range(number_of_labels)
        }
        for gt_label_idx in range(number_of_labels)
    }


def _unpack_unmatched_ground_truths_value(
    unmatched_ground_truths: NDArray[np.int32],
    number_of_labels: int,
    number_of_examples: int,
    index_to_uid: dict[int, str],
    index_to_label: dict[int, str],
) -> dict[str, dict[str, int | list[dict[str, str]]]]:
    """
    Unpacks a numpy array of unmatched ground truth counts and examples.
    """

    datum_idx = (
        lambda gt_label_idx, example_idx: int(  # noqa: E731 - lambda fn
            unmatched_ground_truths[
                gt_label_idx,
                example_idx + 1,
            ]
        )
    )

    return {
        index_to_label[gt_label_idx]: {
            "count": max(
                int(unmatched_ground_truths[gt_label_idx, 0]),
                0,
            ),
            "examples": [
                {"datum": index_to_uid[datum_idx(gt_label_idx, example_idx)]}
                for example_idx in range(number_of_examples)
                if datum_idx(gt_label_idx, example_idx) >= 0
            ],
        }
        for gt_label_idx in range(number_of_labels)
    }


def unpack_confusion_matrix_into_metric_list(
    results: tuple[NDArray[np.float64], NDArray[np.int32]],
    score_thresholds: list[float],
    number_of_examples: int,
    index_to_uid: dict[int, str],
    index_to_label: dict[int, str],
) -> list[Metric]:

    (confusion_matrix, unmatched_ground_truths) = results
    n_scores, n_labels, _, _ = confusion_matrix.shape
    return [
        Metric.confusion_matrix(
            score_threshold=score_threshold,
            maximum_number_of_examples=number_of_examples,
            confusion_matrix=_unpack_confusion_matrix_value(
                confusion_matrix=confusion_matrix[score_idx, :, :, :],
                number_of_labels=n_labels,
                number_of_examples=number_of_examples,
                index_to_label=index_to_label,
                index_to_uid=index_to_uid,
            ),
            unmatched_ground_truths=_unpack_unmatched_ground_truths_value(
                unmatched_ground_truths=unmatched_ground_truths[
                    score_idx, :, :
                ],
                number_of_labels=n_labels,
                number_of_examples=number_of_examples,
                index_to_label=index_to_label,
                index_to_uid=index_to_uid,
            ),
        )
        for score_idx, score_threshold in enumerate(score_thresholds)
    ]
