from collections import defaultdict
from typing import Any

import numba
import numpy as np
from tqdm import tqdm
from valor_core import schemas

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
        h, _ = datum.shape
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

        # rank pairs
        visited_gts, visited_pds = set(), set()
        ranked_pairs = np.zeros((h, 6))
        ranked_row = 0
        for row in range(h):

            datum = data[label_idx][row]

            label_id = datum[4] if datum[4] > -1.0 else datum[5]

            if datum[4] != datum[5]:  # label mismatch
                continue

            gt_tuple = (
                datum[0],  # datum
                datum[1],  # gt_id
                label_id,
            )
            pd_tuple = (
                datum[0],
                datum[2],
                label_id,
            )

            if gt_tuple in visited_gts or pd_tuple in visited_pds:
                continue

            ranked_pairs[ranked_row][0] = datum[0]  # datum
            ranked_pairs[ranked_row][1] = datum[1]  # gt
            ranked_pairs[ranked_row][2] = datum[2]  # pd
            ranked_pairs[ranked_row][3] = datum[3]  # iou
            ranked_pairs[ranked_row][4] = label_id  # label
            ranked_pairs[ranked_row][5] = datum[6]  # score
            ranked_row += 1

            visited_gts.add(gt_tuple)
            visited_pds.add(pd_tuple)

        ranked_pairs = ranked_pairs[:ranked_row]
        h, _ = ranked_pairs.shape

        for thresh_idx in range(n_thresh):
            total_count = 0
            tp_count = 0
            pr_curve = np.zeros((101,))  # binned PR curve
            for row in range(h):

                if ranked_pairs[row][2] < 0:  # no prediction
                    continue

                total_count += 1
                if (
                    ranked_pairs[row][5] > 0  # score > 0
                    and ranked_pairs[row][1] > -0.5  # groundtruth exists
                    and ranked_pairs[row][3]  # passes iou threshold
                    > iou_thresholds[thresh_idx]
                ):
                    tp_count += 1

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
            for precision in pr_curve:
                if precision > running_max:
                    running_max = precision
                auc += running_max
            results[label_idx][thresh_idx] = auc / 101
        results[label_idx][n_thresh] = label_idx  # record label

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

        # datum reference
        self.uid_to_index = dict()
        self.index_to_uid = dict()

        # label reference
        self.label_to_index = dict()
        self.index_to_label = dict()

        # ingestion caches
        self.pr_pairs = list()
        self.detailed_pairs = list()
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
            "_lock",
        }:
            raise AttributeError(
                f"Cannot manually modify '{__name}' after finalization."
            )
        return super().__setattr__(__name, __value)

    def add_data(
        self,
        groundtruths: list[schemas.GroundTruth],
        predictions: list[schemas.Prediction],
    ):
        for groundtruth, prediction in tqdm(zip(groundtruths, predictions)):

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

            indices = (
                (new_data[:, 11] == new_data[:, 12])
                | (new_data[:, 11] == -1.0)
                | (new_data[:, 12] == -1.0)
            )
            self.pr_pairs.append(new_data[indices].copy())

            self.detailed_pairs.append(new_data)

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

            indices = (
                (new_data[:, 11] == new_data[:, 12])
                | (new_data[:, 11] == -1.0)
                | (new_data[:, 12] == -1.0)
            )
            self.pr_pairs.append(new_data[indices].copy())

            self.detailed_pairs.append(new_data)

    def finalize(self) -> None:
        if self._lock:
            return

        if self.pr_pairs is None:
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

        detailed_pairs = np.concatenate(
            _compute_iou(self.detailed_pairs),
            axis=0,
        )

        indices = np.lexsort(
            (
                detailed_pairs[:, 6],
                detailed_pairs[:, 3],
                detailed_pairs[:, 1],
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
    ):
        if not self._lock:
            raise RuntimeError("Data not finalized.")

        metrics = _compute_ap(
            data=self._pairs_sorted_by_label,
            gt_counts=self._label_counts[:, 0],
            iou_thresholds=np.array(iou_thresholds),
        )

        results = list()
        for row in metrics:
            label_id = int(row[-1])
            if label_id == -1:
                continue

            values = {
                str(round(thresh, 5)): round(value, 5)
                for thresh, value in zip(iou_thresholds, row[:-1])
            }
            values["label"] = self.index_to_label[label_id]
            values["type"] = "AP"
            results.append(values)

        return results
