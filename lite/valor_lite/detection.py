from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from tqdm import tqdm
from valor_lite import schemas

"""
Usage
-----

manager = Manager()
manager.add_data(
    groundtruths=groundtruths,
    predictions=predictions,
)
manager.finalize()

metrics = manager.evaluate(iou_thresholds=[0.5])

ap_metrics = metrics[MetricType.AP]
ar_metrics = metrics[MetricType.AR]
"""

# datum id  0
# gt        1
# pd        2
# iou       3
# gt label  4
# pd label  5
# score     6


@dataclass
class BoundingBox:
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    labels: list[tuple[str, str]]
    scores: list[float] | None = None

    def __post_init__(self):
        if self.scores is not None:
            if len(self.labels) != len(self.scores):
                raise ValueError

    @property
    def extrema(self) -> tuple[float, float, float, float]:
        return (self.xmin, self.xmax, self.ymin, self.ymax)


@dataclass
class Detection:
    uid: str
    groundtruths: list[BoundingBox]
    predictions: list[BoundingBox]

    def __post_init__(self):
        for prediction in self.predictions:
            if prediction.scores is None:
                raise ValueError


class MetricType(str, Enum):

    Accuracy = ("Accuracy",)
    Precision = ("Precision",)
    Recall = ("Recall",)
    AP = "AP"
    AR = "AR"
    mAP = "mAP"
    mAR = "mAR"
    APAveragedOverIOUs = "APAveragedOverIOUs"
    mAPAveragedOverIOUs = "mAPAveragedOverIOUs"
    PrecisionRecallCurve = "PrecisionRecallCurve"


@dataclass
class Metric:
    type: str
    value: float
    parameters: dict

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "value": self.value,
            "parameters": self.parameters,
        }


@dataclass
class AP:
    value: float
    iou_threshold: float
    label: tuple[str, str]

    @property
    def metric(self) -> Metric:
        return Metric(
            type=type(self).__name__,
            value=self.value,
            parameters={
                "iou_threshold": self.iou_threshold,
                "label": {
                    "key": self.label[0],
                    "value": self.label[1],
                },
            },
        )

    def to_dict(self) -> dict:
        return self.metric.to_dict()


@dataclass
class mAP:
    value: float
    iou_threshold: float
    label_key: str

    @property
    def metric(self) -> Metric:
        return Metric(
            type=type(self).__name__,
            value=self.value,
            parameters={
                "iou_threshold": self.iou_threshold,
                "label_key": self.label_key,
            },
        )

    def to_dict(self) -> dict:
        return self.metric.to_dict()


@dataclass
class AR:
    value: float
    score_threshold: float
    ious: list[float]
    label: tuple[str, str]

    @property
    def metric(self) -> Metric:
        return Metric(
            type=type(self).__name__,
            value=self.value,
            parameters={
                "score_threshold": self.score_threshold,
                "ious": self.ious,
                "label": {
                    "key": self.label[0],
                    "value": self.label[1],
                },
            },
        )

    def to_dict(self) -> dict:
        return self.metric.to_dict()


@dataclass
class mAR:
    value: float
    score_threshold: float
    ious: list[float]
    label_key: str

    @property
    def metric(self) -> Metric:
        return Metric(
            type=type(self).__name__,
            value=self.value,
            parameters={
                "score_threshold": self.score_threshold,
                "ious": self.ious,
                "label_key": self.label_key,
            },
        )

    def to_dict(self) -> dict:
        return self.metric.to_dict()


@dataclass
class CountWithExamples:
    value: int
    examples: list[str]

    def __post_init__(self):
        self.value = int(self.value)

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "examples": self.examples,
        }


@dataclass
class PrecisionRecallCurvePoint:
    score: float
    tp: CountWithExamples
    fp_misclassification: CountWithExamples
    fp_hallucination: CountWithExamples
    fn_misclassification: CountWithExamples
    fn_missing_prediction: CountWithExamples

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "tp": self.tp.to_dict(),
            "fp_misclassification": self.fp_misclassification.to_dict(),
            "fp_hallucination": self.fp_hallucination.to_dict(),
            "fn_misclassification": self.fn_misclassification.to_dict(),
            "fn_missing_prediction": self.fn_missing_prediction.to_dict(),
        }


@dataclass
class PrecisionRecallCurve:
    iou: float
    value: list[PrecisionRecallCurvePoint]
    label: tuple[str, str]

    def to_dict(self) -> dict:
        return {
            "value": [pt.to_dict() for pt in self.value],
            "iou": self.iou,
            "label": {
                "key": self.label[0],
                "value": self.label[1],
            },
            "type": "PrecisionRecallCurve",
        }


def _compute_iou(data: list[np.ndarray]) -> list[np.ndarray]:

    results = list()
    for datum_idx in range(len(data)):
        datum = data[datum_idx]

        xmin1, xmax1, ymin1, ymax1 = (
            datum[:, 3],
            datum[:, 4],
            datum[:, 5],
            datum[:, 6],
        )
        xmin2, xmax2, ymin2, ymax2 = (
            datum[:, 7],
            datum[:, 8],
            datum[:, 9],
            datum[:, 10],
        )

        xmin = np.maximum(xmin1, xmin2)
        ymin = np.maximum(ymin1, ymin2)
        xmax = np.minimum(xmax1, xmax2)
        ymax = np.minimum(ymax1, ymax2)

        intersection_width = np.maximum(0, xmax - xmin)
        intersection_height = np.maximum(0, ymax - ymin)
        intersection_area = intersection_width * intersection_height

        area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
        area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

        union_area = area1 + area2 - intersection_area

        iou = np.zeros_like(union_area)
        valid_union_mask = union_area >= 1e-9
        iou[valid_union_mask] = (
            intersection_area[valid_union_mask] / union_area[valid_union_mask]
        )

        h = datum.shape[0]
        result = np.zeros((h, 7))
        result[:, 0:3] = datum[:, 0:3]
        result[:, 3] = iou
        result[:, 4:] = datum[:, 11:]

        results.append(result)

    return results


def _compute_ranked_pairs(
    data: np.ndarray,
    label_counts: np.ndarray,
) -> np.ndarray:

    # remove null predictions
    data = data[data[:, 2] >= 0.0]

    # sort by gt_id, iou, score
    indices = np.lexsort(
        (
            data[:, 1],
            -data[:, 3],
            -data[:, 6],
        )
    )
    data = data[indices]

    # remove ignored predictions
    for idx, count in enumerate(label_counts[:, 0]):
        if count > 0:
            continue
        data = data[data[:, 5] != idx]

    # only keep the highest ranked pair
    _, indices = np.unique(data[:, [0, 2]], axis=0, return_index=True)
    return data[np.sort(indices)]


def _compute_average_precision(
    data: np.ndarray,
    label_counts: np.ndarray,
    iou_thresholds: np.ndarray,
) -> np.ndarray:

    data = _compute_ranked_pairs(data, label_counts)

    n_rows = data.shape[0]
    n_labels = label_counts.shape[0]
    n_ious = iou_thresholds.shape[0]

    mask_score = data[:, 6] > 1e-9
    mask_labels_match = np.isclose(data[:, 4], data[:, 5])
    mask_gt_exists = data[:, 1] >= 0.0

    total_count = np.zeros(
        (n_ious, n_rows),
        dtype=np.int32,
    )
    tp_count = np.zeros_like(total_count)
    gt_count = np.zeros_like(total_count)
    pd_labels = data[:, 5].astype(int)

    pr_curve = np.zeros((n_ious, n_labels, 101))  # binned PR curve

    for iou_idx in range(n_ious):

        # create true-positive mask
        mask_iou = data[:, 3] >= iou_thresholds[iou_idx]
        mask = mask_score & mask_iou & mask_labels_match & mask_gt_exists
        tp_candidates = data[mask]
        _, indices_gt_unique = np.unique(
            tp_candidates[:, [0, 1, 4]], axis=0, return_index=True
        )
        mask_gt_unique = np.zeros(tp_candidates.shape[0], dtype=bool)
        mask_gt_unique[indices_gt_unique] = True
        true_positives_mask = np.zeros(n_rows, dtype=bool)
        true_positives_mask[mask] = mask_gt_unique

        # count tp and total
        for pd_label in np.unique(pd_labels):
            mask_pd_label = pd_labels == pd_label
            gt_count[iou_idx][mask_pd_label] = label_counts[pd_label][0]
            total_count[iou_idx][mask_pd_label] = np.arange(
                1, mask_pd_label.sum() + 1
            )
            mask_tp = mask_pd_label & true_positives_mask
            tp_count[iou_idx][mask_tp] = np.arange(1, mask_tp.sum() + 1)

    # calculate precision-recall points
    precision = np.divide(tp_count, total_count, where=total_count > 0)
    recall = np.divide(tp_count, gt_count, where=gt_count > 0)
    recall_index = np.floor(recall * 100.0).astype(int)
    for iou_idx in range(n_ious):
        p = precision[iou_idx]
        r = recall_index[iou_idx]
        pr_curve[iou_idx, pd_labels, r] = np.maximum(
            pr_curve[iou_idx, pd_labels, r], p
        )

    # calculate average precision
    average_precision = np.zeros((n_ious, n_labels))
    running_max = np.zeros((n_ious, n_labels))
    for recall in range(100, -1, -1):
        precision = pr_curve[:, :, recall]
        running_max = np.maximum(precision, running_max)
        average_precision += running_max
    return average_precision / 101.0


def _compute_average_recall(
    data: np.ndarray,
    label_counts: np.ndarray,
    iou_thresholds: np.ndarray,
    score_thresholds: np.ndarray,
) -> np.ndarray:

    data = _compute_ranked_pairs(data, label_counts)

    n_rows = data.shape[0]
    n_labels = label_counts.shape[0]
    n_ious = iou_thresholds.shape[0]
    n_scores = score_thresholds.shape[0]

    mask_labels_match = np.isclose(data[:, 4], data[:, 5])
    mask_gt_exists = data[:, 1] >= 0.0

    average_recall = np.zeros((n_scores, n_labels))

    for score_idx in range(n_scores):

        mask_score_1 = data[:, 6] > 1e-9
        mask_score_2 = data[:, 6] >= score_thresholds[score_idx]
        mask_score = mask_score_1 & mask_score_2

        pd_labels = data[:, 5].astype(int)

        for iou_idx in range(n_ious):

            # create true-positive mask
            mask_iou = data[:, 3] >= iou_thresholds[iou_idx]
            mask = mask_score & mask_iou & mask_labels_match & mask_gt_exists
            tp_candidates = data[mask]
            _, indices_gt_unique = np.unique(
                tp_candidates[:, [0, 1, 4]], axis=0, return_index=True
            )
            mask_gt_unique = np.zeros(tp_candidates.shape[0], dtype=bool)
            mask_gt_unique[indices_gt_unique] = True
            true_positives_mask = np.zeros(n_rows, dtype=bool)
            true_positives_mask[mask] = mask_gt_unique

            # calculate recall
            unique_pd_labels = np.unique(pd_labels)
            gt_count = label_counts[unique_pd_labels, 0]
            tp_count = np.bincount(pd_labels, weights=true_positives_mask)
            tp_count = tp_count[unique_pd_labels]
            average_recall[score_idx][unique_pd_labels] += tp_count / gt_count

    return average_recall / n_ious


def _compute_ap_ar(
    data: np.ndarray,
    label_counts: np.ndarray,
    iou_thresholds: np.ndarray,
    score_thresholds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    data = _compute_ranked_pairs(data, label_counts)

    n_rows = data.shape[0]
    n_labels = label_counts.shape[0]
    n_ious = iou_thresholds.shape[0]
    n_scores = score_thresholds.shape[0]

    mask_score_nonzero = data[:, 6] > 1e-9
    mask_labels_match = np.isclose(data[:, 4], data[:, 5])
    mask_gt_exists = data[:, 1] >= 0.0

    pr_curve = np.zeros((n_ious, n_labels, 101))
    average_recall = np.zeros((n_scores, n_labels))
    pd_labels = data[:, 5].astype(int)
    unique_pd_labels = np.unique(pd_labels)
    gt_count = label_counts[unique_pd_labels, 0]
    running_total_count = np.zeros(
        (n_ious, n_rows),
        dtype=np.int32,
    )
    running_tp_count = np.zeros_like(running_total_count)
    running_gt_count = np.zeros_like(running_total_count)

    for iou_idx in range(n_ious):

        mask_iou = data[:, 3] >= iou_thresholds[iou_idx]
        mask = (
            mask_score_nonzero & mask_iou & mask_labels_match & mask_gt_exists
        )

        for score_idx in range(n_scores):

            mask_score_thresh = data[:, 6] >= score_thresholds[score_idx]

            # create true-positive mask score treshold
            tp_candidates = data[mask & mask_score_thresh]
            _, indices_gt_unique = np.unique(
                tp_candidates[:, [0, 1, 4]], axis=0, return_index=True
            )
            mask_gt_unique = np.zeros(tp_candidates.shape[0], dtype=bool)
            mask_gt_unique[indices_gt_unique] = True
            true_positives_mask = np.zeros(n_rows, dtype=bool)
            true_positives_mask[mask & mask_score_thresh] = mask_gt_unique

            # calculate recall for AR
            tp_count = np.bincount(pd_labels, weights=true_positives_mask)
            tp_count = tp_count[unique_pd_labels]
            average_recall[score_idx][unique_pd_labels] += tp_count / gt_count

        # create true-positive mask score treshold
        tp_candidates = data[mask]
        _, indices_gt_unique = np.unique(
            tp_candidates[:, [0, 1, 4]], axis=0, return_index=True
        )
        mask_gt_unique = np.zeros(tp_candidates.shape[0], dtype=bool)
        mask_gt_unique[indices_gt_unique] = True
        true_positives_mask = np.zeros(n_rows, dtype=bool)
        true_positives_mask[mask] = mask_gt_unique

        # count running tp and total for AP
        for pd_label in unique_pd_labels:
            mask_pd_label = pd_labels == pd_label
            running_gt_count[iou_idx][mask_pd_label] = label_counts[pd_label][
                0
            ]
            running_total_count[iou_idx][mask_pd_label] = np.arange(
                1, mask_pd_label.sum() + 1
            )
            mask_tp = mask_pd_label & true_positives_mask
            running_tp_count[iou_idx][mask_tp] = np.arange(
                1, mask_tp.sum() + 1
            )

    # calculate running precision-recall points for AP
    precision = np.divide(
        running_tp_count, running_total_count, where=running_total_count > 0
    )
    recall = np.divide(
        running_tp_count, running_gt_count, where=running_gt_count > 0
    )
    recall_index = np.floor(recall * 100.0).astype(int)
    for iou_idx in range(n_ious):
        p = precision[iou_idx]
        r = recall_index[iou_idx]
        pr_curve[iou_idx, pd_labels, r] = np.maximum(
            pr_curve[iou_idx, pd_labels, r], p
        )

    # calculate average precision
    average_precision = np.zeros((n_ious, n_labels))
    running_max = np.zeros((n_ious, n_labels))
    for recall in range(100, -1, -1):
        precision = pr_curve[:, :, recall]
        running_max = np.maximum(precision, running_max)
        average_precision += running_max
    average_precision = average_precision / 101.0

    # calculate average recall
    average_recall /= n_ious

    # calculate mAP and mAR
    label_keys = label_counts[unique_pd_labels, 2]
    mAP = np.ones((n_ious, label_keys.shape[0])) * -1.0
    mAR = np.ones((n_ious, label_keys.shape[0])) * -1.0
    for key in np.unique(label_keys):
        labels = unique_pd_labels[label_keys == key]
        key_idx = int(key)
        mAP[:, key_idx] = average_precision[:, labels].mean(axis=1)
        mAR[:, key_idx] = average_recall[:, labels].mean(axis=1)

    return average_precision, average_recall, mAP, mAR


def _compute_pr_curve(
    data: np.ndarray,
    label_counts: np.ndarray,
    iou_thresholds: np.ndarray,
    score_thresholds: np.ndarray,
    n_samples: int,
) -> np.ndarray:

    """
    0  label
    1  tp
    ...
    2  fp - 1
    3  fp - 2
    4  fn - misclassification
    5  fn - hallucination
    """

    n_rows = data.shape[0]
    n_labels = label_counts.shape[0]
    n_ious = iou_thresholds.shape[0]
    n_scores = score_thresholds.shape[0]
    n_metrics = 5 * (n_samples + 1)

    tp_idx = 0
    fp_misclf_idx = tp_idx + n_samples + 1
    fp_halluc_idx = fp_misclf_idx + n_samples + 1
    fn_misclf_idx = fp_halluc_idx + n_samples + 1
    fn_misprd_idx = fn_misclf_idx + n_samples + 1

    detailed_pr_curve = np.ones((n_ious, n_scores, n_labels, n_metrics)) * -1.0

    mask_gt_exists = data[:, 1] > -0.5
    mask_pd_exists = data[:, 2] > -0.5
    mask_label_match = np.isclose(data[:, 4], data[:, 5])

    for iou_idx in range(n_ious):
        mask_iou = data[:, 3] >= iou_thresholds[iou_idx]

        for score_idx in range(n_scores):
            mask_score = data[:, 6] >= score_thresholds[score_idx]

            mask_tp = (
                mask_gt_exists
                & mask_pd_exists
                & mask_iou
                & mask_score
                & mask_label_match
            )
            mask_fp_misclf = (
                mask_gt_exists
                & mask_pd_exists
                & mask_iou
                & mask_score
                & ~mask_label_match
            )
            mask_fn_misclf = (
                mask_gt_exists
                & mask_pd_exists
                & mask_iou
                & ~mask_score
                & mask_label_match
            )
            mask_fp_halluc = (
                ~(mask_tp | mask_fp_misclf | mask_fn_misclf) & mask_pd_exists
            )
            mask_fn_misprd = (
                ~(mask_tp | mask_fp_misclf | mask_fn_misclf) & mask_gt_exists
            )

            tp_slice = data[mask_tp]
            fp_misclf_slice = data[mask_fp_misclf]
            fp_halluc_slice = data[mask_fp_halluc]
            fn_misclf_slice = data[mask_fn_misclf]
            fn_misprd_slice = data[mask_fn_misprd]

            tp_count = np.bincount(
                tp_slice[:, 5].astype(int), minlength=n_labels
            )
            fp_misclf_count = np.bincount(
                fp_misclf_slice[:, 5].astype(int), minlength=n_labels
            )
            fp_halluc_count = np.bincount(
                fp_halluc_slice[:, 5].astype(int), minlength=n_labels
            )
            fn_misclf_count = np.bincount(
                fn_misclf_slice[:, 4].astype(int), minlength=n_labels
            )
            fn_misprd_count = np.bincount(
                fn_misprd_slice[:, 4].astype(int), minlength=n_labels
            )

            detailed_pr_curve[iou_idx, score_idx, :, tp_idx] = tp_count
            detailed_pr_curve[
                iou_idx, score_idx, :, fp_misclf_idx
            ] = fp_misclf_count
            detailed_pr_curve[
                iou_idx, score_idx, :, fp_halluc_idx
            ] = fp_halluc_count
            detailed_pr_curve[
                iou_idx, score_idx, :, fn_misclf_idx
            ] = fn_misclf_count
            detailed_pr_curve[
                iou_idx, score_idx, :, fn_misprd_idx
            ] = fn_misprd_count

            # tp_examples = tp_slice[:n_samples, 0]
            # fp_misclf_examples = fp_misclf_slice[:n_samples, 0]
            # fp_halluc_examples = fp_halluc_slice[:n_samples, 0]
            # fn_misclf_examples = fn_misclf_slice[:n_samples, 0]
            # fn_misprd_examples = fn_misprd_slice[:n_samples, 0]

            if n_samples > 0:
                for label_idx in range(n_labels):
                    tp_examples = tp_slice[
                        tp_slice[:, 5].astype(int) == label_idx
                    ][:n_samples, 0]
                    fp_misclf_examples = fp_misclf_slice[
                        fp_misclf_slice[:, 5].astype(int) == label_idx
                    ][:n_samples, 0]
                    fp_halluc_examples = fp_halluc_slice[
                        fp_halluc_slice[:, 5].astype(int) == label_idx
                    ][:n_samples, 0]
                    fn_misclf_examples = fn_misclf_slice[
                        fn_misclf_slice[:, 4].astype(int) == label_idx
                    ][:n_samples, 0]
                    fn_misprd_examples = fn_misprd_slice[
                        fn_misprd_slice[:, 4].astype(int) == label_idx
                    ][:n_samples, 0]

                    detailed_pr_curve[
                        iou_idx,
                        score_idx,
                        label_idx,
                        tp_idx + 1 : tp_idx + 1 + tp_examples.shape[0],
                    ] = tp_examples
                    detailed_pr_curve[
                        iou_idx,
                        score_idx,
                        label_idx,
                        fp_misclf_idx
                        + 1 : fp_misclf_idx
                        + 1
                        + fp_misclf_examples.shape[0],
                    ] = fp_misclf_examples
                    detailed_pr_curve[
                        iou_idx,
                        score_idx,
                        label_idx,
                        fp_halluc_idx
                        + 1 : fp_halluc_idx
                        + 1
                        + fp_halluc_examples.shape[0],
                    ] = fp_halluc_examples
                    detailed_pr_curve[
                        iou_idx,
                        score_idx,
                        label_idx,
                        fn_misclf_idx
                        + 1 : fn_misclf_idx
                        + 1
                        + fn_misclf_examples.shape[0],
                    ] = fn_misclf_examples
                    detailed_pr_curve[
                        iou_idx,
                        score_idx,
                        label_idx,
                        fn_misprd_idx
                        + 1 : fn_misprd_idx
                        + 1
                        + fn_misprd_examples.shape[0],
                    ] = fn_misprd_examples

    return detailed_pr_curve


def _get_bbox_extrema(
    data: list[list[list[float]]],
) -> tuple[float, float, float, float]:

    x = [point[0] for shape in data for point in shape]
    y = [point[1] for shape in data for point in shape]

    return (
        min(x),
        max(x),
        min(y),
        max(y),
    )


class Manager:
    def __init__(self):

        # metadata
        self.n_datums = 0
        self.n_groundtruths = 0
        self.n_predictions = 0
        self.n_labels = 0
        self.gt_counts = defaultdict(int)
        self.pd_counts = defaultdict(int)

        # datum reference
        self.uid_to_index = dict()
        self.index_to_uid = dict()

        # label reference
        self.label_to_index = dict()
        self.index_to_label = dict()

        # label key reference
        self.index_to_label_key = dict()
        self.label_key_to_index = dict()
        self.label_index_to_label_key_index = dict()

        # ingestion caches
        self.pairs = list()

        # computation caches
        self._label_cache = np.empty((1, 3))

    def __setattr__(self, __name: str, __value: Any) -> None:
        if "_lock" not in self.__dict__:
            super().__setattr__("_lock", False)
        if self._lock and __name in {
            "pairs",
            "gt_counts",
            "pd_counts",
            "uid_to_index",
            "index_to_uid",
            "label_to_index",
            "index_to_label",
            "index_to_label_key",
            "n_datums",
            "n_groundtruths",
            "n_predictions",
            "n_labels",
            "_lock",
        }:
            raise AttributeError(
                f"Cannot manually modify '{__name}' after finalization."
            )
        return super().__setattr__(__name, __value)

    @property
    def ignored_prediction_labels(self) -> list[tuple[str, str]]:
        glabels = set(self.gt_counts.keys())
        plabels = set(self.pd_counts.keys())
        return [
            self.index_to_label[label_id] for label_id in (plabels - glabels)
        ]

    @property
    def missing_prediction_labels(self) -> list[tuple[str, str]]:
        glabels = set(self.gt_counts.keys())
        plabels = set(self.pd_counts.keys())
        return [
            self.index_to_label[label_id] for label_id in (glabels - plabels)
        ]

    @property
    def metadata(self) -> dict:
        return {
            "n_datums": self.n_datums,
            "n_groundtruths": self.n_groundtruths,
            "n_predictions": self.n_predictions,
            "n_labels": self.n_labels,
            "ignored_prediction_labels": self.ignored_prediction_labels,
            "missing_prediction_labels": self.missing_prediction_labels,
        }

    def add_data(
        self,
        detections: list[Detection],
    ):
        for detection in tqdm(detections):

            # update datum uids
            uid = detection.uid
            if uid not in self.uid_to_index:
                index = len(self.uid_to_index)
                self.uid_to_index[uid] = index
                self.index_to_uid[index] = uid
            uid_index = self.uid_to_index[uid]

            # update labels
            glabels = [
                glabel
                for gann in detection.groundtruths
                for glabel in gann.labels
            ]
            glabels_set = set(glabels)

            plabels = [
                plabel
                for pann in detection.predictions
                for plabel in pann.labels
            ]
            plabels_set = set(plabels)

            label_id = len(self.index_to_label)
            label_key_id = len(self.index_to_label_key)
            for label in glabels_set.union(plabels_set):
                if label not in self.label_to_index:
                    self.label_to_index[label] = label_id
                    self.index_to_label[label_id] = label

                    # update label key index
                    if label[0] not in self.label_key_to_index:
                        self.label_key_to_index[label[0]] = label_key_id
                        self.index_to_label_key[label_key_id] = label[0]
                        label_key_id += 1

                    self.label_index_to_label_key_index[
                        label_id
                    ] = self.label_key_to_index[label[0]]
                    label_id += 1

            # count label occurences
            for item in glabels_set:
                self.gt_counts[self.label_to_index[item]] += glabels.count(
                    item
                )
            for item in plabels_set:
                self.pd_counts[self.label_to_index[item]] += plabels.count(
                    item
                )

            # populate numpy array
            if detection.groundtruths and detection.predictions:
                new_data = np.array(
                    [
                        [
                            float(uid_index),
                            float(gidx),
                            float(pidx),
                            *gann.extrema,
                            *pann.extrema,
                            float(self.label_to_index[glabel]),
                            float(self.label_to_index[plabel]),
                            float(pscore),
                        ]
                        for pidx, pann in enumerate(detection.predictions)
                        for plabel, pscore in zip(pann.labels, pann.scores)
                        for gidx, gann in enumerate(detection.groundtruths)
                        for glabel in gann.labels
                    ]
                )
            elif detection.groundtruths:
                new_data = np.array(
                    [
                        [
                            float(uid_index),
                            float(gidx),
                            -1.0,
                            *gann.extrema,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            float(self.label_to_index[glabel]),
                            -1.0,
                            -1.0,
                        ]
                        for gidx, gann in enumerate(detection.groundtruths)
                        for glabel in gann.labels
                    ]
                )
            elif detection.predictions:
                new_data = np.array(
                    [
                        [
                            float(uid_index),
                            -1.0,
                            float(pidx),
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            *pann.extrema,
                            -1.0,
                            float(self.label_to_index[plabel]),
                            float(pscore),
                        ]
                        for pidx, pann in enumerate(detection.predictions)
                        for plabel, pscore in zip(pann.labels, pann.scores)
                    ]
                )
            else:
                new_data = np.array(
                    [
                        [
                            float(uid_index),
                            -1.0,
                            -1.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            -1.0,
                            -1.0,
                            -1.0,
                        ]
                    ]
                )

            self.pairs.append(new_data)

    def add_data_from_valor_core(
        self,
        groundtruths: list[schemas.GroundTruth],
        predictions: list[schemas.Prediction],
    ):
        for groundtruth, prediction in tqdm(zip(groundtruths, predictions)):

            # update metadata
            self.n_datums += 1
            self.n_groundtruths += len(groundtruth.annotations)
            self.n_predictions += len(prediction.annotations)

            # update datum uids
            uid = groundtruth.datum.uid
            if uid not in self.uid_to_index:
                index = len(self.uid_to_index)
                self.uid_to_index[uid] = index
                self.index_to_uid[index] = uid
            uid_index = self.uid_to_index[uid]

            # update labels
            glabels = [
                (glabel.key, glabel.value)
                for gann in groundtruth.annotations
                for glabel in gann.labels
            ]
            glabels_set = set(glabels)

            plabels = [
                (plabel.key, plabel.value)
                for pann in prediction.annotations
                for plabel in pann.labels
            ]
            plabels_set = set(plabels)

            # update label index
            label_id = len(self.index_to_label)
            label_key_id = len(self.index_to_label_key)
            for label in glabels_set.union(plabels_set):
                if label not in self.label_to_index:
                    self.label_to_index[label] = label_id
                    self.index_to_label[label_id] = label

                    # update label key index
                    if label[0] not in self.label_key_to_index:
                        self.label_key_to_index[label[0]] = label_key_id
                        self.index_to_label_key[label_key_id] = label[0]
                        label_key_id += 1

                    self.label_index_to_label_key_index[
                        label_id
                    ] = self.label_key_to_index[label[0]]
                    label_id += 1

            for item in glabels_set:
                self.gt_counts[self.label_to_index[item]] += glabels.count(
                    item
                )
            for item in plabels_set:
                self.pd_counts[self.label_to_index[item]] += plabels.count(
                    item
                )

            if groundtruth.annotations and prediction.annotations:
                new_data = np.array(
                    [
                        [
                            float(uid_index),
                            float(gidx),
                            float(pidx),
                            float(gann.bounding_box.xmin),
                            float(gann.bounding_box.xmax),
                            float(gann.bounding_box.ymin),
                            float(gann.bounding_box.ymax),
                            float(pann.bounding_box.xmin),
                            float(pann.bounding_box.xmax),
                            float(pann.bounding_box.ymin),
                            float(pann.bounding_box.ymax),
                            float(
                                self.label_to_index[(glabel.key, glabel.value)]
                            ),
                            float(
                                self.label_to_index[(plabel.key, plabel.value)]
                            ),
                            float(plabel.score),
                        ]
                        for pidx, pann in enumerate(prediction.annotations)
                        for plabel in pann.labels
                        for gidx, gann in enumerate(groundtruth.annotations)
                        for glabel in gann.labels
                    ]
                )
            elif groundtruth.annotations:
                new_data = np.array(
                    [
                        [
                            float(uid_index),
                            float(gidx),
                            -1.0,
                            float(gann.bounding_box.xmin),
                            float(gann.bounding_box.xmax),
                            float(gann.bounding_box.ymin),
                            float(gann.bounding_box.ymax),
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            float(
                                self.label_to_index[(glabel.key, glabel.value)]
                            ),
                            -1.0,
                            -1.0,
                        ]
                        for gidx, gann in enumerate(groundtruth.annotations)
                        for glabel in gann.labels
                    ]
                )
            elif prediction.annotations:
                new_data = np.array(
                    [
                        [
                            float(uid_index),
                            -1.0,
                            float(pidx),
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            float(pann.bounding_box.xmin),
                            float(pann.bounding_box.xmax),
                            float(pann.bounding_box.ymin),
                            float(pann.bounding_box.ymax),
                            -1.0,
                            float(
                                self.label_to_index[(plabel.key, plabel.value)]
                            ),
                            float(plabel.score),
                        ]
                        for pidx, pann in enumerate(prediction.annotations)
                        for plabel in pann.labels
                    ]
                )
            else:
                new_data = np.array(
                    [
                        [
                            float(uid_index),
                            -1.0,
                            -1.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            -1.0,
                            -1.0,
                            -1.0,
                        ]
                    ]
                )

            self.pairs.append(new_data)

    def add_data_from_valor_dict(
        self,
        groundtruths: list[dict],
        predictions: list[dict],
    ):
        for groundtruth, prediction in tqdm(zip(groundtruths, predictions)):

            # update datum uids
            uid = groundtruth["datum"]["uid"]
            if uid not in self.uid_to_index:
                index = len(self.uid_to_index)
                self.uid_to_index[uid] = index
                self.index_to_uid[index] = uid
            uid_index = self.uid_to_index[uid]

            # update labels
            glabels = [
                (glabel["key"], glabel["value"])
                for gann in groundtruth["annotations"]
                for glabel in gann["labels"]
            ]
            glabels_set = set(glabels)

            plabels = [
                (plabel["key"], plabel["value"])
                for pann in prediction["annotations"]
                for plabel in pann["labels"]
            ]
            plabels_set = set(plabels)

            label_id = len(self.index_to_label)
            label_key_id = len(self.index_to_label_key)
            for label in glabels_set.union(plabels_set):
                if label not in self.label_to_index:
                    self.label_to_index[label] = label_id
                    self.index_to_label[label_id] = label

                    # update label key index
                    if label[0] not in self.label_key_to_index:
                        self.label_key_to_index[label[0]] = label_key_id
                        self.index_to_label_key[label_key_id] = label[0]
                        label_key_id += 1

                    self.label_index_to_label_key_index[
                        label_id
                    ] = self.label_key_to_index[label[0]]
                    label_id += 1

            for item in glabels_set:
                self.gt_counts[self.label_to_index[item]] += glabels.count(
                    item
                )
            for item in plabels_set:
                self.pd_counts[self.label_to_index[item]] += plabels.count(
                    item
                )

            if groundtruth["annotations"] and prediction["annotations"]:
                new_data = np.array(
                    [
                        [
                            float(uid_index),
                            float(gidx),
                            float(pidx),
                            *_get_bbox_extrema(gann["bounding_box"]),
                            *_get_bbox_extrema(pann["bounding_box"]),
                            float(
                                self.label_to_index[
                                    (glabel["key"], glabel["value"])
                                ]
                            ),
                            float(
                                self.label_to_index[
                                    (plabel["key"], plabel["value"])
                                ]
                            ),
                            float(plabel["score"]),
                        ]
                        for pidx, pann in enumerate(prediction["annotations"])
                        for plabel in pann["labels"]
                        for gidx, gann in enumerate(groundtruth["annotations"])
                        for glabel in gann["labels"]
                    ]
                )
            elif groundtruth["annotations"]:
                new_data = np.array(
                    [
                        [
                            float(uid_index),
                            float(gidx),
                            -1.0,
                            *_get_bbox_extrema(gann["bounding_box"]),
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            float(
                                self.label_to_index[
                                    (glabel["key"], glabel["value"])
                                ]
                            ),
                            -1.0,
                            -1.0,
                        ]
                        for gidx, gann in enumerate(groundtruth["annotations"])
                        for glabel in gann["labels"]
                    ]
                )
            elif prediction["annotations"]:
                new_data = np.array(
                    [
                        [
                            float(uid_index),
                            -1.0,
                            float(pidx),
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            *_get_bbox_extrema(pann["bounding_box"]),
                            -1.0,
                            float(
                                self.label_to_index[
                                    (plabel["key"], plabel["value"])
                                ]
                            ),
                            float(plabel["score"]),
                        ]
                        for pidx, pann in enumerate(prediction["annotations"])
                        for plabel in pann["labels"]
                    ]
                )
            else:
                new_data = np.array(
                    [
                        [
                            float(uid_index),
                            -1.0,
                            -1.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            -1.0,
                            -1.0,
                            -1.0,
                        ]
                    ]
                )

            self.pairs.append(new_data)

    def finalize(self) -> None:
        if self._lock:
            return

        if self.pairs is None:
            raise ValueError

        self._label_cache = np.array(
            [
                [
                    float(self.gt_counts.get(label_id, 0)),
                    float(self.pd_counts.get(label_id, 0)),
                    float(self.label_index_to_label_key_index[label_id]),
                ]
                for label_id in range(len(self.index_to_label))
            ]
        )
        self.n_labels = len(self.index_to_label)

        self.detailed_pairs = np.concatenate(
            _compute_iou(self.pairs),
            axis=0,
        )

        # lock the object
        self._lock = True

        # clear ingestion cache
        del self.pairs

    def compute_pr_curve(
        self,
        iou_thresholds: list[float] = [0.5],
        score_thresholds: list[float] = [
            score / 100.0 for score in range(1, 101)
        ],
        n_samples: int = 0,
    ) -> list[PrecisionRecallCurve]:
        if not self._lock:
            raise RuntimeError("Data not finalized.")

        metrics = _compute_pr_curve(
            self.detailed_pairs,
            label_counts=self._label_cache,
            iou_thresholds=np.array(iou_thresholds),
            score_thresholds=np.array(score_thresholds),
            n_samples=n_samples,
        )

        tp_idx = 0
        fp_misclf_idx = tp_idx + n_samples + 1
        fp_halluc_idx = fp_misclf_idx + n_samples + 1
        fn_misclf_idx = fp_halluc_idx + n_samples + 1
        fn_misprd_idx = fn_misclf_idx + n_samples + 1

        results = list()
        for label_idx in range(len(metrics)):
            n_ious, n_scores, _, _ = metrics.shape
            for iou_idx in range(n_ious):
                curve = PrecisionRecallCurve(
                    iou=iou_thresholds[iou_idx],
                    value=list(),
                    label=self.index_to_label[label_idx],
                )
                for score_idx in range(n_scores):
                    curve.value.append(
                        PrecisionRecallCurvePoint(
                            score=(score_idx + 1) / 100.0,
                            tp=CountWithExamples(
                                value=metrics[iou_idx][score_idx][label_idx][
                                    tp_idx
                                ],
                                examples=[
                                    self.index_to_uid[int(datum_idx)]
                                    for datum_idx in metrics[iou_idx][
                                        score_idx
                                    ][label_idx][tp_idx + 1 : fp_misclf_idx]
                                    if int(datum_idx) >= 0
                                ],
                            ),
                            fp_misclassification=CountWithExamples(
                                value=metrics[iou_idx][score_idx][label_idx][
                                    fp_misclf_idx
                                ],
                                examples=[
                                    self.index_to_uid[int(datum_idx)]
                                    for datum_idx in metrics[iou_idx][
                                        score_idx
                                    ][label_idx][
                                        fp_misclf_idx + 1 : fp_halluc_idx
                                    ]
                                    if int(datum_idx) >= 0
                                ],
                            ),
                            fp_hallucination=CountWithExamples(
                                value=metrics[iou_idx][score_idx][label_idx][
                                    fp_halluc_idx
                                ],
                                examples=[
                                    self.index_to_uid[int(datum_idx)]
                                    for datum_idx in metrics[iou_idx][
                                        score_idx
                                    ][label_idx][
                                        fp_halluc_idx + 1 : fn_misclf_idx
                                    ]
                                    if int(datum_idx) >= 0
                                ],
                            ),
                            fn_misclassification=CountWithExamples(
                                value=metrics[iou_idx][score_idx][label_idx][
                                    fn_misclf_idx
                                ],
                                examples=[
                                    self.index_to_uid[int(datum_idx)]
                                    for datum_idx in metrics[iou_idx][
                                        score_idx
                                    ][label_idx][
                                        fn_misclf_idx + 1 : fn_misprd_idx
                                    ]
                                    if int(datum_idx) >= 0
                                ],
                            ),
                            fn_missing_prediction=CountWithExamples(
                                value=metrics[iou_idx][score_idx][label_idx][
                                    fn_misprd_idx
                                ],
                                examples=[
                                    self.index_to_uid[int(datum_idx)]
                                    for datum_idx in metrics[iou_idx][
                                        score_idx
                                    ][label_idx][fn_misprd_idx + 1 :]
                                    if int(datum_idx) >= 0
                                ],
                            ),
                        )
                    )
                results.append(curve)
        return results

    def evaluate(
        self,
        iou_thresholds: list[float] = [0.5, 0.75, 0.9],
        score_thresholds: list[float] = [0.0],
    ) -> dict[MetricType, list[AP] | list[AR] | list[mAP] | list[mAR]]:

        if not self._lock:
            raise RuntimeError("Data not finalized.")

        (
            average_precision,
            average_recall,
            mean_average_precision,
            mean_average_recall,
        ) = _compute_ap_ar(
            data=self.detailed_pairs,
            label_counts=self._label_cache,
            iou_thresholds=np.array(iou_thresholds),
            score_thresholds=np.array(score_thresholds),
        )

        ap_metrics = [
            AP(
                value=value,
                iou_threshold=iou_thresholds[iou_idx],
                label=self.index_to_label[label_idx],
            )
            for label_idx, row in enumerate(range(average_precision.shape[1]))
            if int(self._label_cache[label_idx][0]) > 0.0
            for iou_idx, value in enumerate(average_precision[:, row])
        ]

        map_metrics = [
            mAP(
                value=value,
                iou_threshold=iou_thresholds[iou_idx],
                label_key=self.index_to_label_key[label_key_idx],
            )
            for label_key_idx, row in enumerate(
                range(mean_average_precision.shape[1])
            )
            if label_key_idx < -0.5
            for iou_idx, value in enumerate(mean_average_precision[:, row])
        ]

        ar_metrics = [
            AR(
                value=value,
                ious=iou_thresholds,
                score_threshold=score_thresholds[score_idx],
                label=self.index_to_label[label_idx],
            )
            for label_idx, row in enumerate(range(average_recall.shape[1]))
            if int(self._label_cache[label_idx][0]) > 0.0
            for score_idx, value in enumerate(average_recall[:, row])
        ]

        mar_metrics = [
            mAR(
                value=value,
                ious=iou_thresholds,
                score_threshold=score_thresholds[score_idx],
                label_key=self.index_to_label_key[label_key_idx],
            )
            for label_key_idx, row in enumerate(
                range(mean_average_recall.shape[1])
            )
            if label_key_idx < -0.5
            for score_idx, value in enumerate(mean_average_recall[:, row])
        ]

        return {
            MetricType.AP: ap_metrics,
            MetricType.mAP: map_metrics,
            MetricType.AR: ar_metrics,
            MetricType.mAR: mar_metrics,
        }

    def benchmark_iou(self):
        if not self.pairs:
            raise ValueError
        _compute_iou(self.pairs)
