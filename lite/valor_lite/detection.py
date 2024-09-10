from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from valor_lite import schemas

"""
Usage
-----

manager = DataLoader()
manager.add_data(
    groundtruths=groundtruths,
    predictions=predictions,
)
evaluator = manager.finalize()

metrics = evaluator.evaluate(iou_thresholds=[0.5])

ap_metrics = metrics[MetricType.AP]
ar_metrics = metrics[MetricType.AR]

filter_mask = evaluator.create_filter(datum_uids=["uid1", "uid2"])
filtered_metrics = evaluator.evaluate(iou_thresholds=[0.5], filter_mask=filter_mask)
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
    def extrema(self) -> NDArray[np.floating]:
        return np.array([self.xmin, self.xmax, self.ymin, self.ymax])


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
    TP = "TruePositiveCount"
    FP = "FalsePositiveCount"
    FN = "FalseNegativeCount"
    Accuracy = "Accuracy"
    Precision = "Precision"
    Recall = "Recall"
    F1 = "F1"
    AP = "AP"
    AR = "AR"
    mAP = "mAP"
    mAR = "mAR"
    APAveragedOverIOUs = "APAveragedOverIOUs"
    mAPAveragedOverIOUs = "mAPAveragedOverIOUs"
    PrecisionRecallCurve = "PrecisionRecallCurve"
    DetailedPrecisionRecallCurve = "DetailedPrecisionRecallCurve"


@dataclass
class Metric:
    type: str
    value: float | dict | list
    parameters: dict

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "value": self.value,
            "parameters": self.parameters,
        }


@dataclass
class CountingClassMetric:
    value: int
    label: tuple[str, str]
    iou_threhsold: float
    score_threshold: float

    @property
    def metric(self) -> Metric:
        return Metric(
            type=type(self).__name__,
            value=self.value,
            parameters={
                "iou_threshold": self.iou_threhsold,
                "score_threshold": self.score_threshold,
                "label": {
                    "key": self.label[0],
                    "value": self.label[1],
                },
            },
        )

    def to_dict(self) -> dict:
        return self.metric.to_dict()


@dataclass
class ClassMetric:
    value: float
    label: tuple[str, str]
    iou_threhsold: float
    score_threshold: float

    @property
    def metric(self) -> Metric:
        return Metric(
            type=type(self).__name__,
            value=self.value,
            parameters={
                "iou_threshold": self.iou_threhsold,
                "score_threshold": self.score_threshold,
                "label": {
                    "key": self.label[0],
                    "value": self.label[1],
                },
            },
        )

    def to_dict(self) -> dict:
        return self.metric.to_dict()


class Precision(ClassMetric):
    pass


class Recall(ClassMetric):
    pass


class Accuracy(ClassMetric):
    pass


class F1(ClassMetric):
    pass


class TruePositiveCount(CountingClassMetric):
    pass


class FalsePositiveCount(CountingClassMetric):
    pass


class FalseNegativeCount(CountingClassMetric):
    pass


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
class InterpolatedPrecisionRecallCurve:
    """
    Interpolated over recalls 0.0, 0.01, ..., 1.0.
    """

    precision: list[float]
    iou_threshold: float
    label: tuple[str, str]

    @property
    def metric(self) -> Metric:
        return Metric(
            type=type(self).__name__,
            value=self.precision,
            parameters={
                "iou_threshold": self.iou_threshold,
                "label": {"key": self.label[0], "value": self.label[1]},
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
class DetailedPrecisionRecallPoint:
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
class DetailedPrecisionRecallCurve:
    iou: float
    value: list[DetailedPrecisionRecallPoint]
    label: tuple[str, str]

    def to_dict(self) -> dict:
        return {
            "value": [pt.to_dict() for pt in self.value],
            "iou": self.iou,
            "label": {
                "key": self.label[0],
                "value": self.label[1],
            },
            "type": "DetailedPrecisionRecallCurve",
        }


def _compute_iou(data: NDArray[np.floating]) -> NDArray[np.floating]:

    xmin1, xmax1, ymin1, ymax1 = (
        data[:, 0],
        data[:, 1],
        data[:, 2],
        data[:, 3],
    )
    xmin2, xmax2, ymin2, ymax2 = (
        data[:, 4],
        data[:, 5],
        data[:, 6],
        data[:, 7],
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

    iou = np.zeros(data.shape[0])
    valid_union_mask = union_area >= 1e-9
    iou[valid_union_mask] = (
        intersection_area[valid_union_mask] / union_area[valid_union_mask]
    )
    return iou


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


def _compute_metrics(
    data: np.ndarray,
    label_counts: np.ndarray,
    iou_thresholds: np.ndarray,
    score_thresholds: np.ndarray,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Computes Object Detection metrics.

    Returns
    -------
    np.ndarray
        Average Precision.
    np.ndarray
        Average Recall.
    np.ndarray
        mAP.
    np.ndarray
        mAR.
    np.ndarray
        Precision, Recall, TP, FP, FN, F1 Score, Accuracy.
    np.ndarray
        Interpolated Precision-Recall Curves.
    """

    n_rows = data.shape[0]
    n_labels = label_counts.shape[0]
    n_ious = iou_thresholds.shape[0]
    n_scores = score_thresholds.shape[0]

    average_precision = np.zeros((n_ious, n_labels))
    average_recall = np.zeros((n_scores, n_labels))
    precision_recall = np.zeros((n_ious, n_scores, n_labels, 7))

    pd_labels = data[:, 5].astype(int)
    unique_pd_labels = np.unique(pd_labels)
    gt_count = label_counts[unique_pd_labels, 0]
    running_total_count = np.zeros(
        (n_ious, n_rows),
        dtype=np.int32,
    )
    running_tp_count = np.zeros_like(running_total_count)
    running_gt_count = np.zeros_like(running_total_count)
    pr_curve = np.zeros((n_ious, n_labels, 101))

    mask_score_nonzero = data[:, 6] > 1e-9
    mask_labels_match = np.isclose(data[:, 4], data[:, 5])
    mask_gt_exists = data[:, 1] >= 0.0

    for iou_idx in range(n_ious):

        mask_iou = data[:, 3] >= iou_thresholds[iou_idx]
        mask = (
            mask_score_nonzero & mask_iou & mask_labels_match & mask_gt_exists
        )

        for score_idx in range(n_scores):

            mask_score_thresh = data[:, 6] >= score_thresholds[score_idx]

            # create true-positive mask score threshold
            tp_candidates = data[mask & mask_score_thresh]
            _, indices_gt_unique = np.unique(
                tp_candidates[:, [0, 1, 4]], axis=0, return_index=True
            )
            mask_gt_unique = np.zeros(tp_candidates.shape[0], dtype=bool)
            mask_gt_unique[indices_gt_unique] = True
            true_positives_mask = np.zeros(n_rows, dtype=bool)
            true_positives_mask[mask & mask_score_thresh] = mask_gt_unique

            # calculate intermediates
            pd_count = np.bincount(pd_labels)
            pd_count = pd_count[unique_pd_labels]
            tp_count = np.bincount(
                pd_labels,
                weights=true_positives_mask,
            )
            tp_count = tp_count[unique_pd_labels]

            # calculate component metrics
            precision = tp_count / pd_count
            recall = tp_count / gt_count
            fp_count = pd_count - tp_count
            fn_count = gt_count - tp_count
            f1_score = np.divide(
                np.multiply(precision, recall),
                (precision + recall),
                where=(precision + recall) > 1e-9,
            )
            accuracy = tp_count / (gt_count + pd_count)
            precision_recall[iou_idx][score_idx][
                unique_pd_labels
            ] = np.concatenate(
                (
                    tp_count[:, np.newaxis],
                    fp_count[:, np.newaxis],
                    fn_count[:, np.newaxis],
                    precision[:, np.newaxis],
                    recall[:, np.newaxis],
                    f1_score[:, np.newaxis],
                    accuracy[:, np.newaxis],
                ),
                axis=1,
            )

            # calculate recall for AR
            average_recall[score_idx][unique_pd_labels] += recall

        # create true-positive mask score threshold
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
    mAR = np.ones((n_scores, label_keys.shape[0])) * -1.0
    for key in np.unique(label_keys):
        labels = unique_pd_labels[label_keys == key]
        key_idx = int(key)
        mAP[:, key_idx] = average_precision[:, labels].mean(axis=1)
        mAR[:, key_idx] = average_recall[:, labels].mean(axis=1)

    return (
        average_precision,
        average_recall,
        mAP,
        mAR,
        precision_recall,
        pr_curve,
    )


def _compute_detailed_pr_curve(
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
) -> NDArray[np.floating]:

    x = [point[0] for shape in data for point in shape]
    y = [point[1] for shape in data for point in shape]

    return np.array(
        [
            min(x),
            max(x),
            min(y),
            max(y),
        ]
    )


class Evaluator:
    def __init__(self):

        # metadata
        self.n_datums = 0
        self.n_groundtruths = 0
        self.n_predictions = 0
        self.n_labels = 0
        self.gt_counts = defaultdict(int)
        self.pd_counts = defaultdict(int)

        # datum reference
        self.uid_to_index: dict[str, int] = dict()
        self.index_to_uid: dict[int, str] = dict()

        # label reference
        self.label_to_index: dict[tuple[str, str], int] = dict()
        self.index_to_label: dict[int, tuple[str, str]] = dict()

        # label key reference
        self.index_to_label_key: dict[int, str] = dict()
        self.label_key_to_index: dict[str, int] = dict()
        self.label_index_to_label_key_index: dict[int, int] = dict()

        # computation caches
        self._detailed_pairs = np.empty((1, 7))
        self._ranked_pairs = np.empty((1, 7))
        self._label_metadata = np.empty((1, 3))

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

    def create_filter(
        self,
        datum_uids: list[str] | None = None,
        labels: list[tuple[str, str]] | None = None,
        label_keys: list[str] | None = None,
    ) -> NDArray[np.bool_]:
        """
        Creates a boolean mask that can be passed to an evaluation.
        """

        mask = np.ones_like(self._ranked_pairs, dtype=np.bool_)
        if datum_uids is not None:
            indices = np.array([self.uid_to_index[uid] for uid in datum_uids])
            mask &= self._ranked_pairs[:, 0].astype(int) == indices

        if labels is not None:
            indices = np.array(
                [self.label_to_index[label] for label in labels]
            )
            mask &= (
                self._ranked_pairs[:, 4].astype(int) == indices
            )  # groundtruth filter
            mask &= (
                self._ranked_pairs[:, 5].astype(int) == indices
            )  # prediction filter

        if label_keys is not None:
            key_indices = np.array(
                [self.label_key_to_index[key] for key in label_keys]
            )
            label_indices = np.where(self._label_metadata[:, 2] == key_indices)
            mask &= (
                self._ranked_pairs[:, 4].astype(int) == label_indices
            )  # groundtruth filter
            mask &= (
                self._ranked_pairs[:, 5].astype(int) == label_indices
            )  # prediction filter

        return mask

    def evaluate(
        self,
        iou_thresholds: list[float] = [0.5, 0.75, 0.9],
        score_thresholds: list[float] = [0.0],
        filter_mask: NDArray[np.bool_] | None = None,
    ) -> dict[MetricType, list]:
        """
        Runs evaluation over cached data.

        Parameters
        ----------
        iou_thresholds : list[float]
            A list of iou thresholds to compute over.
        score_thresholds : list[float]
            A list of score thresholds to compute over.
        filter_mask : NDArray[bool], optional
            A boolean mask that filters the cached data.
        """

        data = self._ranked_pairs
        if filter_mask is not None:
            data = data[filter_mask]

        (
            average_precision,
            average_recall,
            mean_average_precision,
            mean_average_recall,
            precision_recall,
            pr_curves,
        ) = _compute_metrics(
            data=self._ranked_pairs,
            label_counts=self._label_metadata,
            iou_thresholds=np.array(iou_thresholds),
            score_thresholds=np.array(score_thresholds),
        )

        metrics = defaultdict(list)

        metrics[MetricType.AP] = [
            AP(
                value=value,
                iou_threshold=iou_thresholds[iou_idx],
                label=self.index_to_label[label_idx],
            )
            for label_idx, row in enumerate(range(average_precision.shape[1]))
            if int(self._label_metadata[label_idx][0]) > 0.0
            for iou_idx, value in enumerate(average_precision[:, row])
        ]

        metrics[MetricType.mAP] = [
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

        metrics[MetricType.AR] = [
            AR(
                value=value,
                ious=iou_thresholds,
                score_threshold=score_thresholds[score_idx],
                label=self.index_to_label[label_idx],
            )
            for label_idx, row in enumerate(range(average_recall.shape[1]))
            if int(self._label_metadata[label_idx][0]) > 0.0
            for score_idx, value in enumerate(average_recall[:, row])
        ]

        metrics[MetricType.mAR] = [
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

        metrics[MetricType.PrecisionRecallCurve] = [
            InterpolatedPrecisionRecallCurve(
                precision=list(pr_curves[iou_idx][label_idx]),
                iou_threshold=iou_threshold,
                label=label,
            )
            for iou_idx, iou_threshold in enumerate(iou_thresholds)
            for label_idx, label in self.index_to_label.items()
        ]

        for iou_idx, iou_threshold in enumerate(iou_thresholds):
            for score_idx, score_threshold in enumerate(score_thresholds):
                for label_idx, label in self.index_to_label.items():
                    row = precision_recall[iou_idx][score_idx][label_idx]
                    kwargs = {
                        "label": label,
                        "iou_threhsold": iou_threshold,
                        "score_threshold": score_threshold,
                    }
                    metrics[MetricType.TP].append(
                        TruePositiveCount(
                            value=int(row[0]),
                            **kwargs,
                        )
                    )
                    metrics[MetricType.FP].append(
                        FalsePositiveCount(
                            value=int(row[1]),
                            **kwargs,
                        )
                    )
                    metrics[MetricType.FN].append(
                        FalseNegativeCount(
                            value=int(row[2]),
                            **kwargs,
                        )
                    )
                    metrics[MetricType.Precision].append(
                        Precision(
                            value=row[3],
                            **kwargs,
                        )
                    )
                    metrics[MetricType.Recall].append(
                        Recall(
                            value=row[4],
                            **kwargs,
                        )
                    )
                    metrics[MetricType.F1].append(
                        F1(
                            value=row[5],
                            **kwargs,
                        )
                    )
                    metrics[MetricType.Accuracy].append(
                        Accuracy(
                            value=row[6],
                            **kwargs,
                        )
                    )

        return metrics

    def compute_detailed_pr_curve(
        self,
        iou_thresholds: list[float] = [0.5],
        score_thresholds: list[float] = [
            score / 100.0 for score in range(1, 101)
        ],
        n_samples: int = 0,
    ) -> list[DetailedPrecisionRecallCurve]:

        if not self._detailed_pairs:
            raise ValueError

        metrics = _compute_detailed_pr_curve(
            self._detailed_pairs,
            label_counts=self._label_metadata,
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
                curve = DetailedPrecisionRecallCurve(
                    iou=iou_thresholds[iou_idx],
                    value=list(),
                    label=self.index_to_label[label_idx],
                )
                for score_idx in range(n_scores):
                    curve.value.append(
                        DetailedPrecisionRecallPoint(
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


class DataLoader:
    def __init__(self):
        self._evaluator = Evaluator()
        self.pairs = list()

    def add_data(
        self,
        detections: list[Detection],
        show_progress: bool = False,
    ):
        for detection in tqdm(detections, disable=show_progress):

            # update datum uids
            uid = detection.uid
            if uid not in self._evaluator.uid_to_index:
                index = len(self._evaluator.uid_to_index)
                self._evaluator.uid_to_index[uid] = index
                self._evaluator.index_to_uid[index] = uid
            uid_index = self._evaluator.uid_to_index[uid]

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
            label_id = len(self._evaluator.index_to_label)
            label_key_id = len(self._evaluator.index_to_label_key)
            for label in glabels_set.union(plabels_set):
                if label not in self._evaluator.label_to_index:
                    self._evaluator.label_to_index[label] = label_id
                    self._evaluator.index_to_label[label_id] = label

                    # update label key index
                    if label[0] not in self._evaluator.label_key_to_index:
                        self._evaluator.label_key_to_index[
                            label[0]
                        ] = label_key_id
                        self._evaluator.index_to_label_key[
                            label_key_id
                        ] = label[0]
                        label_key_id += 1

                    self._evaluator.label_index_to_label_key_index[
                        label_id
                    ] = self._evaluator.label_key_to_index[label[0]]
                    label_id += 1

            # count label occurences
            for item in glabels_set:
                self._evaluator.gt_counts[
                    self._evaluator.label_to_index[item]
                ] += glabels.count(item)
            for item in plabels_set:
                self._evaluator.pd_counts[
                    self._evaluator.label_to_index[item]
                ] += plabels.count(item)

            # populate numpy array
            if detection.groundtruths and detection.predictions:
                boxes = np.array(
                    [
                        np.concatenate((gann.extrema, pann.extrema), axis=1)
                        for pann in detection.predictions
                        for gann in detection.groundtruths
                    ]
                )
                ious = _compute_iou(boxes)
                new_data = np.array(
                    [
                        [
                            float(uid_index),
                            float(gidx),
                            float(pidx),
                            float(
                                ious[pidx * len(detection.groundtruths) + gidx]
                            ),
                            float(self._evaluator.label_to_index[glabel]),
                            float(self._evaluator.label_to_index[plabel]),
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
                            0.0,
                            float(self._evaluator.label_to_index[glabel]),
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
                            -1.0,
                            float(self._evaluator.label_to_index[plabel]),
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
            self._evaluator.n_datums += 1
            self._evaluator.n_groundtruths += len(groundtruth.annotations)
            self._evaluator.n_predictions += len(prediction.annotations)

            # update datum uids
            uid = groundtruth.datum.uid
            if uid not in self._evaluator.uid_to_index:
                index = len(self._evaluator.uid_to_index)
                self._evaluator.uid_to_index[uid] = index
                self._evaluator.index_to_uid[index] = uid
            uid_index = self._evaluator.uid_to_index[uid]

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
            label_id = len(self._evaluator.index_to_label)
            label_key_id = len(self._evaluator.index_to_label_key)
            for label in glabels_set.union(plabels_set):
                if label not in self._evaluator.label_to_index:
                    self._evaluator.label_to_index[label] = label_id
                    self._evaluator.index_to_label[label_id] = label

                    # update label key index
                    if label[0] not in self._evaluator.label_key_to_index:
                        self._evaluator.label_key_to_index[
                            label[0]
                        ] = label_key_id
                        self._evaluator.index_to_label_key[
                            label_key_id
                        ] = label[0]
                        label_key_id += 1

                    self._evaluator.label_index_to_label_key_index[
                        label_id
                    ] = self._evaluator.label_key_to_index[label[0]]
                    label_id += 1

            for item in glabels_set:
                self._evaluator.gt_counts[
                    self._evaluator.label_to_index[item]
                ] += glabels.count(item)
            for item in plabels_set:
                self._evaluator.pd_counts[
                    self._evaluator.label_to_index[item]
                ] += plabels.count(item)

            if groundtruth.annotations and prediction.annotations:
                boxes = np.array(
                    [
                        np.array(
                            [
                                gann.bounding_box.xmin,
                                gann.bounding_box.xmax,
                                gann.bounding_box.ymin,
                                gann.bounding_box.ymax,
                                pann.bounding_box.xmin,
                                pann.bounding_box.xmax,
                                pann.bounding_box.ymin,
                                pann.bounding_box.ymax,
                            ]
                        )
                        for pann in prediction.annotations
                        for gann in groundtruth.annotations
                    ]
                )
                ious = _compute_iou(boxes)
                new_data = np.array(
                    [
                        [
                            float(uid_index),
                            float(gidx),
                            float(pidx),
                            float(
                                ious[
                                    pidx * len(groundtruth.annotations) + gidx
                                ]
                            ),
                            float(
                                self._evaluator.label_to_index[
                                    (glabel.key, glabel.value)
                                ]
                            ),
                            float(
                                self._evaluator.label_to_index[
                                    (plabel.key, plabel.value)
                                ]
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
                            0.0,
                            float(
                                self._evaluator.label_to_index[
                                    (glabel.key, glabel.value)
                                ]
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
                            -1.0,
                            float(
                                self._evaluator.label_to_index[
                                    (plabel.key, plabel.value)
                                ]
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
            if uid not in self._evaluator.uid_to_index:
                index = len(self._evaluator.uid_to_index)
                self._evaluator.uid_to_index[uid] = index
                self._evaluator.index_to_uid[index] = uid
            uid_index = self._evaluator.uid_to_index[uid]

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

            label_id = len(self._evaluator.index_to_label)
            label_key_id = len(self._evaluator.index_to_label_key)
            for label in glabels_set.union(plabels_set):
                if label not in self._evaluator.label_to_index:
                    self._evaluator.label_to_index[label] = label_id
                    self._evaluator.index_to_label[label_id] = label

                    # update label key index
                    if label[0] not in self._evaluator.label_key_to_index:
                        self._evaluator.label_key_to_index[
                            label[0]
                        ] = label_key_id
                        self._evaluator.index_to_label_key[
                            label_key_id
                        ] = label[0]
                        label_key_id += 1

                    self._evaluator.label_index_to_label_key_index[
                        label_id
                    ] = self._evaluator.label_key_to_index[label[0]]
                    label_id += 1

            for item in glabels_set:
                self._evaluator.gt_counts[
                    self._evaluator.label_to_index[item]
                ] += glabels.count(item)
            for item in plabels_set:
                self._evaluator.pd_counts[
                    self._evaluator.label_to_index[item]
                ] += plabels.count(item)

            if groundtruth["annotations"] and prediction["annotations"]:
                boxes = np.array(
                    [
                        np.concatenate(
                            (
                                _get_bbox_extrema(gann["bounding_box"]),
                                _get_bbox_extrema(pann["bounding_box"]),
                            ),
                            axis=0,
                        )
                        for pann in prediction["annotations"]
                        for gann in groundtruth["annotations"]
                    ]
                )
                ious = _compute_iou(boxes)
                new_data = np.array(
                    [
                        [
                            float(uid_index),
                            float(gidx),
                            float(pidx),
                            float(
                                ious[
                                    pidx * len(groundtruth["annotations"])
                                    + gidx
                                ]
                            ),
                            float(
                                self._evaluator.label_to_index[
                                    (glabel["key"], glabel["value"])
                                ]
                            ),
                            float(
                                self._evaluator.label_to_index[
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
                            0.0,
                            float(
                                self._evaluator.label_to_index[
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
                            -1.0,
                            float(
                                self._evaluator.label_to_index[
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
                            -1.0,
                            -1.0,
                            -1.0,
                        ]
                    ]
                )

            self.pairs.append(new_data)

    def finalize(self) -> Evaluator:

        if self.pairs is None:
            raise ValueError

        self._evaluator._label_metadata = np.array(
            [
                [
                    float(self._evaluator.gt_counts.get(label_id, 0)),
                    float(self._evaluator.pd_counts.get(label_id, 0)),
                    float(
                        self._evaluator.label_index_to_label_key_index[
                            label_id
                        ]
                    ),
                ]
                for label_id in range(len(self._evaluator.index_to_label))
            ]
        )
        self._evaluator.n_labels = len(self._evaluator.index_to_label)

        self._evaluator._detailed_pairs = np.concatenate(
            self.pairs,
            axis=0,
        )

        self._evaluator._ranked_pairs = _compute_ranked_pairs(
            data=self._evaluator._detailed_pairs,
            label_counts=self._evaluator._label_metadata,
        )

        return self._evaluator
