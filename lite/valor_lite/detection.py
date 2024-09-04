from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numba
import numpy as np
from tqdm import tqdm
from valor_lite import schemas
from valor_lite.enums import MetricType

# datum id  0
# gt        1
# pd        2
# gt xmin   3
# gt xmax   4
# gt ymin   5
# gt ymax   6
# pd xmin   7
# pd xmax   8
# pd ymin   9
# pd ymax   10
# gt label  11
# pd label  12
# pd score  13


# datum id  0
# gt        1
# pd        2
# iou       3
# gt label  4
# pd label  5
# score     6


@dataclass
class ValueAtIoU:
    value: float
    iou: float


@dataclass
class AP:
    values: list[ValueAtIoU]
    label: tuple[str, str]

    def to_dict(self) -> dict:
        return {
            "type": "AP",
            "values": {str(value.iou): value.value for value in self.values},
            "label": {
                "key": self.label[0],
                "value": self.label[1],
            },
        }


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


@numba.njit(parallel=False)
def _compute_iou(data: list[np.ndarray]) -> list[np.ndarray]:
    """
    Returns a list of ious in form [(p1, g1), (p1, g2), ..., (p1, gN), (p2, g1), ...].
    """
    results = [
        np.empty((1, 7)).astype(data[0].dtype) for i in range(len(data))
    ]
    for datum_idx in numba.prange(len(data)):
        datum = data[datum_idx]
        h = datum.shape[0]
        results[datum_idx] = np.zeros((h, 7))
        for row in numba.prange(h):
            if (
                datum[row][3] >= datum[row][8]
                or datum[row][4] <= datum[row][7]
                or datum[row][5] >= datum[row][10]
                or datum[row][6] <= datum[row][9]
            ):
                iou = 0.0
            else:

                xmin1 = datum[row][3]
                xmax1 = datum[row][4]
                ymin1 = datum[row][5]
                ymax1 = datum[row][6]
                xmin2 = datum[row][7]
                xmax2 = datum[row][8]
                ymin2 = datum[row][9]
                ymax2 = datum[row][10]

                gt_area = (xmax1 - xmin1) * (ymax1 - ymin1)
                pd_area = (xmax2 - xmin2) * (ymax2 - ymin2)

                xmin = max([xmin1, xmin2])
                xmax = min([xmax1, xmax2])
                ymin = max([ymin1, ymin2])
                ymax = min([ymax1, ymax2])

                intersection = (xmax - xmin) * (ymax - ymin)
                union = gt_area + pd_area - intersection
                iou = intersection / union if union > 0 else 0.0

            results[datum_idx][row][0] = datum[row][0]
            results[datum_idx][row][1] = datum[row][1]
            results[datum_idx][row][2] = datum[row][2]
            results[datum_idx][row][3] = iou
            results[datum_idx][row][4] = datum[row][11]
            results[datum_idx][row][5] = datum[row][12]
            results[datum_idx][row][6] = datum[row][13]

    return results


# @numba.njit(parallel=False)
def _compute_ap(
    data: list[np.ndarray],
    gt_counts: np.ndarray,
    iou_thresholds: np.ndarray,
) -> np.ndarray:

    n_thresh = iou_thresholds.shape[0]
    results = np.zeros((len(data), n_thresh + 1))  # AP metrics + label
    for label_idx in numba.prange(len(data)):
        h, _ = data[label_idx].shape

        number_of_groundtruths = gt_counts[label_idx]

        # if no groundtruths exist for the label, set to null results and continue
        if number_of_groundtruths == 0.0:
            for thresh_idx in range(n_thresh):
                results[label_idx][thresh_idx] = 0.0
            results[label_idx][n_thresh] = -1.0
            continue

        # rank pairs
        visited_pds = set()
        ranked_pairs = np.zeros((h, 7))
        ranked_row = 0
        for row in range(h):

            datum = data[label_idx][row]

            if datum[5] != label_idx:  # label mismatch
                continue

            pd_tuple = (
                datum[0],  # datum
                datum[2],  # pd id
                datum[5],  # pd label
            )

            if pd_tuple not in visited_pds or datum[1] < 0.0:
                is_match = 1.0 if datum[4] == datum[5] else 0.0
                ranked_pairs[ranked_row][0] = datum[0]  # datum
                ranked_pairs[ranked_row][1] = datum[1]  # gt
                ranked_pairs[ranked_row][2] = datum[2]  # pd
                ranked_pairs[ranked_row][3] = datum[3]  # iou
                ranked_pairs[ranked_row][4] = datum[5]  # label
                ranked_pairs[ranked_row][5] = datum[6]  # score
                ranked_pairs[ranked_row][6] = is_match  # is_match
                visited_pds.add(pd_tuple)
                ranked_row += 1

        ranked_pairs = ranked_pairs[:ranked_row]
        h = ranked_pairs.shape[0]

        for thresh_idx in range(n_thresh):
            total_count = 0
            tp_count = 0
            pr_curve = np.zeros((101,))  # binned PR curve
            visited_gts = set()
            for row in range(h):

                gt_tuple = (
                    ranked_pairs[row][0],  # datum
                    ranked_pairs[row][1],  # gt
                )

                if ranked_pairs[row][2] < 0:  # no prediction
                    continue

                total_count += 1
                if (
                    ranked_pairs[row][5] > 0.0  # score > 0
                    and ranked_pairs[row][1] > -0.5  # groundtruth exists
                    and ranked_pairs[row][3]  # passes iou threshold
                    > iou_thresholds[thresh_idx]
                    and ranked_pairs[row][6] > 0.5  # matching label
                    and gt_tuple not in visited_gts
                ):
                    tp_count += 1
                    visited_gts.add(gt_tuple)

                precision = tp_count / total_count
                recall = (
                    tp_count / number_of_groundtruths
                    if number_of_groundtruths > 0
                    else 0.0
                )

                recall = int(np.floor(recall * 100.0))
                pr_curve[recall] = (
                    precision
                    if precision > pr_curve[recall]
                    else pr_curve[recall]
                )

            # interpolate the PR-Curve
            auc = 0.0
            running_max = 0.0
            for recall in range(100, -1, -1):
                precision = pr_curve[recall]
                if precision > running_max:
                    running_max = precision
                auc += running_max
            results[label_idx][thresh_idx] = auc / 101
        results[label_idx][n_thresh] = label_idx  # record label

    return results


# @numba.njit(parallel=False)
def _compute_detailed_pr_curve(
    data: list[np.ndarray],
    label_counts: np.ndarray,
    iou_thresholds: np.ndarray,
    n_samples: int,
) -> list[np.ndarray]:

    """
    0  label
    1  tp
    ...
    2  fp - 1
    3  fp - 2
    4  fn - misclassification
    5  fn - hallucination
    """

    n_ious = iou_thresholds.shape[0]

    w = 5 * (n_samples + 1) + 1
    tp_idx = 1
    fp_misclf_idx = tp_idx + n_samples + 1
    fp_hall_idx = fp_misclf_idx + n_samples + 1
    fn_misclf_idx = fp_hall_idx + n_samples + 1
    fn_misprd_idx = fn_misclf_idx + n_samples + 1

    results = [
        np.empty((len(iou_thresholds), 100, w)) for i in range(len(data))
    ]
    for label_idx in numba.prange(len(data)):
        result = np.zeros((len(iou_thresholds), 100, w))
        datum = data[label_idx]
        n_rows = datum.shape[0]
        for iou_idx in numba.prange(n_ious):
            for score_idx in numba.prange(100):
                score_threshold = (score_idx + 1) / 100.0
                tp, fp_misclf, fp_hall, fn_misclf, fn_misprd = 0, 0, 0, 0, 0
                for row in numba.prange(n_rows):

                    gt_exists = datum[row][1] > -0.5
                    pd_exists = datum[row][2] > -0.5
                    iou = datum[row][3]
                    gt_label = int(datum[row][4])
                    pd_label = int(datum[row][5])
                    score = datum[row][6]

                    tp_conditional = (
                        gt_exists
                        and pd_exists
                        and iou >= iou_thresholds[iou_idx]
                        and score >= score_threshold
                        and gt_label == pd_label
                        and gt_label == label_idx
                    )
                    fp_misclf_conditional = (
                        gt_exists
                        and pd_exists
                        and iou >= iou_thresholds[iou_idx]
                        and score >= score_threshold
                        and gt_label != pd_label
                        and pd_label == label_idx
                    )
                    fn_misclf_conditional = (
                        gt_exists
                        and pd_exists
                        and iou >= iou_thresholds[iou_idx]
                        and score <= score_threshold
                        and gt_label == pd_label
                        and gt_label == label_idx
                    )

                    if tp_conditional:
                        tp += 1
                    elif fp_misclf_conditional:
                        fp_misclf += 1
                    elif fn_misclf_conditional:
                        fn_misclf += 1
                    elif pd_exists:
                        fp_hall += 1
                    elif gt_exists:
                        fn_misprd += 1
                    else:
                        raise RuntimeError

                result[iou_idx][score_idx][0] = label_idx
                result[iou_idx][score_idx][tp_idx] = tp
                result[iou_idx][score_idx][fp_misclf_idx] = fp_misclf
                result[iou_idx][score_idx][fp_hall_idx] = fp_hall
                result[iou_idx][score_idx][fn_misclf_idx] = fn_misclf
                result[iou_idx][score_idx][fn_misprd_idx] = fn_misprd

        results[label_idx] = result
    return results


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


class DetectionManager:
    def __init__(self):

        # metadata
        self.n_datums = 0
        self.n_groundtruths = 0
        self.n_predictions = 0
        self.n_labels = 0
        self.ignored_prediction_labels = list()
        self.missing_prediction_labels = list()

        # datum reference
        self.uid_to_index = dict()
        self.index_to_uid = dict()

        # label reference
        self.label_to_index = dict()
        self.index_to_label = dict()

        # ingestion caches
        self.pairs = list()
        self.gt_counts = defaultdict(int)
        self.pd_counts = defaultdict(int)

        # computation caches
        self._label_counts = np.empty((1, 2))
        self._pairs_sorted_by_label = list()

    def __setattr__(self, __name: str, __value: Any) -> None:
        if "_lock" not in self.__dict__:
            super().__setattr__("_lock", False)
        if self._lock and __name in {
            "pr_pairs",
            "detailed_pairs",
            "gt_count",
            "pd_count",
            "uid_to_index",
            "index_to_uid",
            "label_to_index",
            "index_to_label",
            "ignored_prediction_labels",
            "missing_groundtruth_labels",
            "n_datums",
            "n_annotations",
            "n_labels",
            "_lock",
        }:
            raise AttributeError(
                f"Cannot manually modify '{__name}' after finalization."
            )
        return super().__setattr__(__name, __value)

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

            label_id = len(self.index_to_label)
            for label in glabels_set.union(plabels_set):
                if label not in self.label_to_index:
                    self.label_to_index[label] = label_id
                    self.index_to_label[label_id] = label
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

    def add_data_from_dict(
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
            for label in glabels_set.union(plabels_set):
                if label not in self.label_to_index:
                    self.label_to_index[label] = label_id
                    self.index_to_label[label_id] = label
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

        self._label_counts = np.array(
            [
                [
                    float(self.gt_counts.get(label_id, 0)),
                    float(self.pd_counts.get(label_id, 0)),
                ]
                for label_id in range(len(self.index_to_label))
            ]
        )

        glabels = set(self.gt_counts.keys())
        plabels = set(self.pd_counts.keys())

        self.ignored_prediction_labels = [
            schemas.Label(
                key=self.index_to_label[idx][0],
                value=self.index_to_label[idx][1],
            )
            for idx in (plabels - glabels)
        ]
        self.missing_prediction_labels = [
            schemas.Label(
                key=self.index_to_label[idx][0],
                value=self.index_to_label[idx][1],
            )
            for idx in (glabels - plabels)
        ]
        self.n_labels = len(self.index_to_label)

        detailed_pairs = np.concatenate(
            _compute_iou(self.pairs),
            axis=0,
        )

        indices = np.lexsort(
            (
                detailed_pairs[:, 1],
                -detailed_pairs[:, 3],
                -detailed_pairs[:, 6],
            )
        )
        detailed_pairs = detailed_pairs[indices]

        self._pairs_sorted_by_label = [
            detailed_pairs[
                (detailed_pairs[:, 4] == label_id)
                | (detailed_pairs[:, 5] == label_id)
            ]
            for label_id in range(len(self.index_to_label))
        ]

        # lock the object
        self._lock = True

    def compute_ap(
        self,
        iou_thresholds: list[float] = [0.5, 0.75, 0.9],
    ) -> list[AP]:
        if not self._lock:
            raise RuntimeError("Data not finalized.")

        metrics = _compute_ap(
            data=self._pairs_sorted_by_label,
            gt_counts=self._label_counts[:, 0],
            iou_thresholds=np.array(iou_thresholds),
        )

        return [
            AP(
                values=[
                    ValueAtIoU(
                        iou=iou_thresholds[iou_idx],
                        value=value,
                    )
                    for iou_idx, value in enumerate(row[:-1])
                ],
                label=self.index_to_label[row[-1]],
            )
            for row in metrics
            if row[-1] > -0.5
        ]

    def compute_pr_curve(
        self,
        iou_thresholds: list[float] = [0.5],
        n_samples: int = 0,
    ) -> list[PrecisionRecallCurve]:
        if not self._lock:
            raise RuntimeError("Data not finalized.")

        metrics = _compute_detailed_pr_curve(
            self._pairs_sorted_by_label,
            label_counts=self._label_counts,
            iou_thresholds=np.array(iou_thresholds),
            n_samples=n_samples,
        )

        tp_idx = 1
        fp_misclf_idx = n_samples + 1
        fp_hall_idx = fp_misclf_idx + n_samples + 1
        fn_misclf_idx = fp_hall_idx + n_samples + 1
        fn_misprd_idx = fn_misclf_idx + n_samples + 1

        results = list()
        for label_idx in range(len(metrics)):
            n_ious, n_scores, _ = metrics[label_idx].shape
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
                                value=metrics[label_idx][iou_idx][score_idx][
                                    tp_idx
                                ],
                                examples=[],
                            ),
                            fp_misclassification=CountWithExamples(
                                value=metrics[label_idx][iou_idx][score_idx][
                                    fp_misclf_idx
                                ],
                                examples=[],
                            ),
                            fp_hallucination=CountWithExamples(
                                value=metrics[label_idx][iou_idx][score_idx][
                                    fp_hall_idx
                                ],
                                examples=[],
                            ),
                            fn_misclassification=CountWithExamples(
                                value=metrics[label_idx][iou_idx][score_idx][
                                    fn_misclf_idx
                                ],
                                examples=[],
                            ),
                            fn_missing_prediction=CountWithExamples(
                                value=metrics[label_idx][iou_idx][score_idx][
                                    fn_misprd_idx
                                ],
                                examples=[],
                            ),
                        )
                    )
                results.append(curve)
        return results

    def evaluate(
        self,
        metrics: list[MetricType],
    ):
        return list()
