from collections import defaultdict

import numpy as np
from numpy.typing import NDArray
from valor_lite.object_detection.metric import Metric, MetricType


def unpack_precision_recall_into_metric_lists(
    results: tuple[
        tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            float,
        ],
        tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            float,
        ],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ],
    iou_thresholds: list[float],
    score_thresholds: list[float],
    index_to_label: dict[int, str],
    label_metadata: NDArray[np.int32],
):
    (
        (
            average_precision,
            mean_average_precision,
            average_precision_average_over_ious,
            mean_average_precision_average_over_ious,
        ),
        (
            average_recall,
            mean_average_recall,
            average_recall_averaged_over_scores,
            mean_average_recall_averaged_over_scores,
        ),
        accuracy,
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
        for label_idx, label in index_to_label.items()
        if int(label_metadata[label_idx, 0]) > 0
    ]

    metrics[MetricType.mAP] = [
        Metric.mean_average_precision(
            value=float(mean_average_precision[iou_idx]),
            iou_threshold=iou_threshold,
        )
        for iou_idx, iou_threshold in enumerate(iou_thresholds)
    ]

    metrics[MetricType.APAveragedOverIOUs] = [
        Metric.average_precision_averaged_over_IOUs(
            value=float(average_precision_average_over_ious[label_idx]),
            iou_thresholds=iou_thresholds,
            label=label,
        )
        for label_idx, label in index_to_label.items()
        if int(label_metadata[label_idx, 0]) > 0
    ]

    metrics[MetricType.mAPAveragedOverIOUs] = [
        Metric.mean_average_precision_averaged_over_IOUs(
            value=float(mean_average_precision_average_over_ious),
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

    metrics[MetricType.ARAveragedOverScores] = [
        Metric.average_recall_averaged_over_scores(
            value=float(average_recall_averaged_over_scores[label_idx]),
            score_thresholds=score_thresholds,
            iou_thresholds=iou_thresholds,
            label=label,
        )
        for label_idx, label in index_to_label.items()
        if int(label_metadata[label_idx, 0]) > 0
    ]

    metrics[MetricType.mARAveragedOverScores] = [
        Metric.mean_average_recall_averaged_over_scores(
            value=float(mean_average_recall_averaged_over_scores),
            score_thresholds=score_thresholds,
            iou_thresholds=iou_thresholds,
        )
    ]

    metrics[MetricType.Accuracy] = [
        Metric.accuracy(
            value=float(accuracy[iou_idx, score_idx]),
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
        )
        for iou_idx, iou_threshold in enumerate(iou_thresholds)
        for score_idx, score_threshold in enumerate(score_thresholds)
    ]

    metrics[MetricType.PrecisionRecallCurve] = [
        Metric.precision_recall_curve(
            precisions=pr_curves[iou_idx, label_idx, :, 0]
            .astype(float)
            .tolist(),
            scores=pr_curves[iou_idx, label_idx, :, 1].astype(float).tolist(),
            iou_threshold=iou_threshold,
            label=label,
        )
        for iou_idx, iou_threshold in enumerate(iou_thresholds)
        for label_idx, label in index_to_label.items()
        if label_metadata[label_idx, 0] > 0
    ]

    for label_idx, label in index_to_label.items():

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


def _convert_example_to_dict(box: NDArray[np.float16]) -> dict[str, float]:
    """
    Converts a cached bounding box example to dictionary format.
    """
    return {
        "xmin": float(box[0]),
        "xmax": float(box[1]),
        "ymin": float(box[2]),
        "ymax": float(box[3]),
    }


def _unpack_confusion_matrix_value(
    confusion_matrix: NDArray[np.float64],
    number_of_labels: int,
    number_of_examples: int,
    index_to_uid: dict[int, str],
    index_to_label: dict[int, str],
    groundtruth_examples: dict[int, NDArray[np.float16]],
    prediction_examples: dict[int, NDArray[np.float16]],
) -> dict[
    str,
    dict[
        str,
        dict[
            str,
            int
            | list[
                dict[
                    str,
                    str | dict[str, float] | float,
                ]
            ],
        ],
    ],
]:
    """
    Unpacks a numpy array of confusion matrix counts and examples.
    """

    datum_idx = lambda gt_label_idx, pd_label_idx, example_idx: int(  # noqa: E731 - lambda fn
        confusion_matrix[
            gt_label_idx,
            pd_label_idx,
            example_idx * 4 + 1,
        ]
    )

    groundtruth_idx = lambda gt_label_idx, pd_label_idx, example_idx: int(  # noqa: E731 - lambda fn
        confusion_matrix[
            gt_label_idx,
            pd_label_idx,
            example_idx * 4 + 2,
        ]
    )

    prediction_idx = lambda gt_label_idx, pd_label_idx, example_idx: int(  # noqa: E731 - lambda fn
        confusion_matrix[
            gt_label_idx,
            pd_label_idx,
            example_idx * 4 + 3,
        ]
    )

    score_idx = lambda gt_label_idx, pd_label_idx, example_idx: float(  # noqa: E731 - lambda fn
        confusion_matrix[
            gt_label_idx,
            pd_label_idx,
            example_idx * 4 + 4,
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
                        "groundtruth": _convert_example_to_dict(
                            groundtruth_examples[
                                datum_idx(
                                    gt_label_idx,
                                    pd_label_idx,
                                    example_idx,
                                )
                            ][
                                groundtruth_idx(
                                    gt_label_idx,
                                    pd_label_idx,
                                    example_idx,
                                )
                            ]
                        ),
                        "prediction": _convert_example_to_dict(
                            prediction_examples[
                                datum_idx(
                                    gt_label_idx,
                                    pd_label_idx,
                                    example_idx,
                                )
                            ][
                                prediction_idx(
                                    gt_label_idx,
                                    pd_label_idx,
                                    example_idx,
                                )
                            ]
                        ),
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


def _unpack_unmatched_predictions_value(
    unmatched_predictions: NDArray[np.float64],
    number_of_labels: int,
    number_of_examples: int,
    index_to_uid: dict[int, str],
    index_to_label: dict[int, str],
    prediction_examples: dict[int, NDArray[np.float16]],
) -> dict[
    str,
    dict[
        str,
        int | list[dict[str, str | float | dict[str, float]]],
    ],
]:
    """
    Unpacks a numpy array of unmatched_prediction counts and examples.
    """

    datum_idx = (
        lambda pd_label_idx, example_idx: int(  # noqa: E731 - lambda fn
            unmatched_predictions[
                pd_label_idx,
                example_idx * 3 + 1,
            ]
        )
    )

    prediction_idx = (
        lambda pd_label_idx, example_idx: int(  # noqa: E731 - lambda fn
            unmatched_predictions[
                pd_label_idx,
                example_idx * 3 + 2,
            ]
        )
    )

    score_idx = (
        lambda pd_label_idx, example_idx: float(  # noqa: E731 - lambda fn
            unmatched_predictions[
                pd_label_idx,
                example_idx * 3 + 3,
            ]
        )
    )

    return {
        index_to_label[pd_label_idx]: {
            "count": max(
                int(unmatched_predictions[pd_label_idx, 0]),
                0,
            ),
            "examples": [
                {
                    "datum": index_to_uid[
                        datum_idx(pd_label_idx, example_idx)
                    ],
                    "prediction": _convert_example_to_dict(
                        prediction_examples[
                            datum_idx(pd_label_idx, example_idx)
                        ][prediction_idx(pd_label_idx, example_idx)]
                    ),
                    "score": score_idx(pd_label_idx, example_idx),
                }
                for example_idx in range(number_of_examples)
                if datum_idx(pd_label_idx, example_idx) >= 0
            ],
        }
        for pd_label_idx in range(number_of_labels)
    }


def _unpack_unmatched_ground_truths_value(
    unmatched_ground_truths: NDArray[np.int32],
    number_of_labels: int,
    number_of_examples: int,
    index_to_uid: dict[int, str],
    index_to_label: dict[int, str],
    groundtruth_examples: dict[int, NDArray[np.float16]],
) -> dict[str, dict[str, int | list[dict[str, str | dict[str, float]]]]]:
    """
    Unpacks a numpy array of unmatched ground truth counts and examples.
    """

    datum_idx = (
        lambda gt_label_idx, example_idx: int(  # noqa: E731 - lambda fn
            unmatched_ground_truths[
                gt_label_idx,
                example_idx * 2 + 1,
            ]
        )
    )

    groundtruth_idx = (
        lambda gt_label_idx, example_idx: int(  # noqa: E731 - lambda fn
            unmatched_ground_truths[
                gt_label_idx,
                example_idx * 2 + 2,
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
                {
                    "datum": index_to_uid[
                        datum_idx(gt_label_idx, example_idx)
                    ],
                    "groundtruth": _convert_example_to_dict(
                        groundtruth_examples[
                            datum_idx(gt_label_idx, example_idx)
                        ][groundtruth_idx(gt_label_idx, example_idx)]
                    ),
                }
                for example_idx in range(number_of_examples)
                if datum_idx(gt_label_idx, example_idx) >= 0
            ],
        }
        for gt_label_idx in range(number_of_labels)
    }


def unpack_confusion_matrix_into_metric_list(
    results: tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.int32],
    ],
    iou_thresholds: list[float],
    score_thresholds: list[float],
    number_of_examples: int,
    index_to_label: dict[int, str],
    index_to_uid: dict[int, str],
    groundtruth_examples: dict[int, NDArray[np.float16]],
    prediction_examples: dict[int, NDArray[np.float16]],
) -> list[Metric]:
    (
        confusion_matrix,
        unmatched_predictions,
        unmatched_ground_truths,
    ) = results
    n_labels = len(index_to_label)
    return [
        Metric.confusion_matrix(
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
            maximum_number_of_examples=number_of_examples,
            confusion_matrix=_unpack_confusion_matrix_value(
                confusion_matrix=confusion_matrix[iou_idx, score_idx, :, :, :],
                number_of_labels=n_labels,
                number_of_examples=number_of_examples,
                index_to_label=index_to_label,
                index_to_uid=index_to_uid,
                groundtruth_examples=groundtruth_examples,
                prediction_examples=prediction_examples,
            ),
            unmatched_predictions=_unpack_unmatched_predictions_value(
                unmatched_predictions=unmatched_predictions[
                    iou_idx, score_idx, :, :
                ],
                number_of_labels=n_labels,
                number_of_examples=number_of_examples,
                index_to_label=index_to_label,
                index_to_uid=index_to_uid,
                prediction_examples=prediction_examples,
            ),
            unmatched_ground_truths=_unpack_unmatched_ground_truths_value(
                unmatched_ground_truths=unmatched_ground_truths[
                    iou_idx, score_idx, :, :
                ],
                number_of_labels=n_labels,
                number_of_examples=number_of_examples,
                index_to_label=index_to_label,
                index_to_uid=index_to_uid,
                groundtruth_examples=groundtruth_examples,
            ),
        )
        for iou_idx, iou_threshold in enumerate(iou_thresholds)
        for score_idx, score_threshold in enumerate(score_thresholds)
    ]
