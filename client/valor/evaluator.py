import numba
import numpy as np
from tqdm import tqdm

from valor import GroundTruth, Prediction

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
# label     4
# score     5


@numba.njit(parallel=True)
def _compute_iou(
    data: list[np.ndarray],
) -> list[np.ndarray]:
    """
    Returns a list of ious in form [(p1, g1), (p1, g2), ..., (p1, gN), (p2, g1), ...].
    """
    results = [np.empty((1,)).astype(data[0].dtype) for i in range(len(data))]
    for datum_idx in numba.prange(len(data)):
        datum = data[datum_idx]
        h, _ = datum.shape
        results[datum_idx] = np.zeros((h,))
        for row in numba.prange(h):
            if (
                datum[row][3] >= datum[row][8]
                or datum[row][4] <= datum[row][7]
                or datum[row][5] >= datum[row][10]
                or datum[row][6] <= datum[row][9]
            ):
                results[datum_idx][row] = 0.0
            else:
                gt_area = (datum[row][4] - datum[row][3]) * (
                    datum[row][6] - datum[row][5]
                )
                pd_area = (datum[row][8] - datum[row][7]) * (
                    datum[row][10] - datum[row][9]
                )

                xmin = (
                    datum[row][6]
                    if datum[row][6] > datum[row][7]
                    else datum[row][7]
                )
                xmax = (
                    datum[row][4]
                    if datum[row][4] < datum[row][8]
                    else datum[row][8]
                )
                ymin = (
                    datum[row][5]
                    if datum[row][5] > datum[row][9]
                    else datum[row][9]
                )
                ymax = (
                    datum[row][6]
                    if datum[row][6] < datum[row][10]
                    else datum[row][10]
                )

                intersection = (xmax - xmin) * (ymax - ymin)
                union = gt_area + pd_area - intersection
                results[datum_idx][row] = (
                    intersection / union if union > 0 else 0.0
                )
    return results


def compute_iou(data: list[np.ndarray]):
    results = _compute_iou(data)
    return np.concatenate(results, axis=0)


@numba.njit(parallel=True)
def _compute_ranked_pairs(
    data: list[np.ndarray],
) -> list[np.ndarray]:
    """
    Returns a list of ious in form [(p1, g1), (p1, g2), ..., (p1, gN), (p2, g1), ...].
    """
    results = [np.empty((1,6)) for _ in range(len(data))]
    for datum_idx in numba.prange(len(data)):
        datum = data[datum_idx]
        h, _ = datum.shape
        ious = np.zeros((h,))
        for row in numba.prange(h):
            if (
                datum[row][3] >= datum[row][8]
                or datum[row][4] <= datum[row][7]
                or datum[row][5] >= datum[row][10]
                or datum[row][6] <= datum[row][9]
            ):
                ious[row] = 0.0
            else:
                gt_area = (datum[row][4] - datum[row][3]) * (
                    datum[row][6] - datum[row][5]
                )
                pd_area = (datum[row][8] - datum[row][7]) * (
                    datum[row][10] - datum[row][9]
                )

                xmin = (
                    datum[row][6]
                    if datum[row][6] > datum[row][7]
                    else datum[row][7]
                )
                xmax = (
                    datum[row][4]
                    if datum[row][4] < datum[row][8]
                    else datum[row][8]
                )
                ymin = (
                    datum[row][5]
                    if datum[row][5] > datum[row][9]
                    else datum[row][9]
                )
                ymax = (
                    datum[row][6]
                    if datum[row][6] < datum[row][10]
                    else datum[row][10]
                )

                intersection = (xmax - xmin) * (ymax - ymin)
                union = gt_area + pd_area - intersection
                ious[row] = (
                    intersection / union if union > 0 else 0.0
                )

        visited_gts, visited_pds = set(), set()
        ranked_pairs = np.zeros((h, 6))
        for row in range(h):
            
            ranked_pairs[row][0] = datum[row][0]
            ranked_pairs[row][1] = datum[row][1]
            ranked_pairs[row][2] = datum[row][2]
            ranked_pairs[row][4] = datum[row][11] if datum[row][11] > -1.0 else datum[row][12]
            ranked_pairs[row][5] = datum[row][13]

            gt_tuple = (
                datum[row][0],
                datum[row][1],
                datum[row][4],
            )
            pd_tuple = (
                datum[row][0],
                datum[row][2],
                datum[row][4],
            )

            if (
                gt_tuple in visited_gts 
                or pd_tuple in visited_pds
            ):
                continue

            visited_gts.add(gt_tuple)
            visited_pds.add(pd_tuple)

            ranked_pairs[row][3] = ious[row]
        results[datum_idx] = ranked_pairs
    return results


@numba.njit(parallel=True)
def _compute_ap(
    data: list[np.ndarray],
    iou_thresholds: np.ndarray,
) -> np.ndarray:
    
    n_thresh = iou_thresholds.shape[0]
    results = np.zeros((len(data), n_thresh + 1)) # AP metrics per label
    for label_idx in numba.prange(len(data)):
        h, _ = data[label_idx].shape
        for thresh_idx in range(n_thresh):
            tp, fp, fn = 0, 0, 0
            pr_curve = np.zeros((101,)) # binned PR curve
            for row in range(h):
                if (
                    data[label_idx][row][1] < 0
                    and data[label_idx][row][2] < 0
                ):  # neither a groundtruth or prediction exist
                    continue
                elif data[label_idx][row][1] < 0:  # no groundtruth exists
                    fp += 1
                elif data[label_idx][row][2] < 0:  # no prediction exists
                    fn += 1
                elif data[label_idx][row][3] >= iou_thresholds[thresh_idx]:
                    tp += 1
                else:
                    fp += 1
                
                precision = (
                    tp / (tp + fp) if (tp + fp) else 0.0
                )
                recall = (
                    tp / (tp + fn) if (tp + fn) else 0.0
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
            for recall in range(101, -1, -1):
                if pr_curve[recall] > running_max:
                    running_max = pr_curve[recall]
                else:
                    pr_curve[recall] = running_max
                auc += pr_curve[recall]
            results[label_idx][thresh_idx] = auc / 101
        results[label_idx][n_thresh] = data[label_idx][0][4]  # record label
        
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
        self.pr_pairs = list()
        self.detailed_pairs = list()

        self.gt_count = 0
        self.pd_count = 0

        self.gt_cache = dict()
        self.pd_cache = dict()

        self.uid_to_index = dict()
        self.index_to_uid = dict()

        self.label_to_index = dict()
        self.index_to_label = dict()

    def add_data(
        self,
        groundtruths: list[dict],
        predictions: list[dict],
    ):
        for groundtruth, prediction in tqdm(zip(groundtruths, predictions)):

            self.gt_count += len(groundtruth["annotations"])
            self.pd_count += len(prediction["annotations"])

            # update datum uids
            uid = groundtruth["datum"]["uid"]
            if uid not in self.uid_to_index:
                index = len(self.uid_to_index)
                self.uid_to_index[uid] = index
                self.index_to_uid[index] = uid
            uid_index = self.uid_to_index[uid]

            # update labels
            glabels = {
                (glabel["key"], glabel["value"])
                for gann in groundtruth["annotations"]
                for glabel in gann["labels"]
            }
            plabels = {
                (plabel["key"], plabel["value"])
                for pann in prediction["annotations"]
                for plabel in pann["labels"]
            }
            label_count = len(self.label_to_index)
            for idx, label in enumerate(glabels.union(plabels)):
                if label not in self.label_to_index:
                    self.label_to_index[label] = label_count + idx
                    self.index_to_label[label_count + idx] = label

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

    def benchmark_iou(self):
        if self.detailed_pairs is None:
            raise ValueError
        compute_iou(self.detailed_pairs)

    def compute_ap(
        self,
        iou_thresholds: list[float] = [0.3, 0.6, 0.9],
    ):
        if self.pr_pairs is None:
            raise ValueError

        pairs = np.concatenate(
            _compute_ranked_pairs(self.pr_pairs),
            axis=0,
        )

        indices = np.lexsort(
            (pairs[:, 4], pairs[:, 5], pairs[:, 3], pairs[:, 1])
        )
        sorted_pairs = pairs[indices]

        pairs_by_label = [
            sorted_pairs[sorted_pairs[:, 4] == label]
            for label in np.unique(sorted_pairs[:, 4])
        ]

        metrics = _compute_ap(pairs_by_label, iou_thresholds=np.array(iou_thresholds))

        results = []
        n_labels, n_cols = metrics.shape
        for row in metrics:
            label_id = int(row[-1])
            if label_id == -1:
                continue

            values = {
                str(round(thresh, 2)): round(value, 2)
                for thresh, value in zip(iou_thresholds, row[:-1])
            }
            values["label"] = self.index_to_label[label_id]
            results.append(values)

        return results
