import numba
import numpy as np

from valor import GroundTruth, Prediction


@numba.njit()
def compute_ious(
    gt_boxes: np.ndarray,
    pd_boxes: np.ndarray,
    n_gt: int,
    n_pd: int,
    result: np.ndarray,
) -> np.ndarray:
    """
    Returns a list of ious in form [(p1, g1), (p1, g2), ..., (p1, gN), (p2, g1), ...].
    """
    for pidx in numba.prange(n_pd):
        for gidx in range(n_gt):
            if (
                gt_boxes[gidx][0] >= pd_boxes[pidx][1]
                or gt_boxes[gidx][1] <= pd_boxes[pidx][0]
                or gt_boxes[gidx][2] >= pd_boxes[pidx][3]
                or gt_boxes[gidx][3] <= pd_boxes[pidx][2]
            ):
                result[pidx * n_gt + gidx] = 0.0
            else:
                gt_area = (gt_boxes[gidx][1] - gt_boxes[gidx][0]) * (
                    gt_boxes[gidx][3] - gt_boxes[gidx][2]
                )
                pd_area = (pd_boxes[pidx][1] - pd_boxes[pidx][0]) * (
                    pd_boxes[pidx][3] - pd_boxes[pidx][2]
                )

                xmin = (
                    gt_boxes[gidx][0]
                    if gt_boxes[gidx][0] > pd_boxes[pidx][0]
                    else pd_boxes[pidx][0]
                )
                xmax = (
                    gt_boxes[gidx][1]
                    if gt_boxes[gidx][1] < pd_boxes[pidx][1]
                    else pd_boxes[pidx][1]
                )
                ymin = (
                    gt_boxes[gidx][2]
                    if gt_boxes[gidx][2] > pd_boxes[pidx][2]
                    else pd_boxes[pidx][2]
                )
                ymax = (
                    gt_boxes[gidx][3]
                    if gt_boxes[gidx][3] < pd_boxes[pidx][3]
                    else pd_boxes[pidx][3]
                )

                intersection = (xmax - xmin) * (ymax - ymin)
                union = gt_area + pd_area - intersection

                result[pidx * n_gt + gidx] = (
                    intersection / union if union > 0 else 0
                )
    return result


def preprocess_pair(
    datum_index: int,
    groundtruth: GroundTruth,
    prediction: Prediction,
    label_to_index: dict[tuple[str, str], int],
):
    if not groundtruth.annotations and not prediction.annotations:
        return np.array(
            [
                [
                    float(datum_index),
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    0.0,
                ]
            ]
        )
    elif not prediction.annotations:
        return np.array(
            [
                [
                    float(datum_index),
                    float(gidx),
                    -1.0,
                    float(label_to_index[(glabel.key, glabel.value)]),
                    float(-1),
                    -1.0,
                    0.0,
                ]
                for gidx, gann in enumerate(groundtruth.annotations)
                for glabel in gann.labels
            ]
        )
    elif not groundtruth.annotations:
        return np.array(
            [
                [
                    float(datum_index),
                    -1.0,
                    float(pidx),
                    -1.0,
                    float(label_to_index[(plabel.key, plabel.value)]),
                    float(plabel.score),
                    0.0,
                ]
                for pidx, pann in enumerate(prediction.annotations)
                for plabel in pann.labels
            ]
        )
    else:
        n_gt = len(groundtruth.annotations)
        n_pd = len(prediction.annotations)
        results = np.zeros((n_gt * n_pd,))
        ious = compute_ious(
            np.array(
                [
                    [
                        ann.bounding_box.xmin,
                        ann.bounding_box.xmax,
                        ann.bounding_box.ymin,
                        ann.bounding_box.ymax,
                    ]
                    if ann.bounding_box
                    else [0, 0, 0, 0]
                    for ann in groundtruth.annotations
                ]
            ),
            np.array(
                [
                    [
                        ann.bounding_box.xmin,
                        ann.bounding_box.xmax,
                        ann.bounding_box.ymin,
                        ann.bounding_box.ymax,
                    ]
                    if ann.bounding_box
                    else [0, 0, 0, 0]
                    for ann in prediction.annotations
                ]
            ),
            n_gt,
            n_pd,
            results,
        )
        return np.array(
            [
                [
                    float(datum_index),
                    float(gidx),
                    float(pidx),
                    float(label_to_index[(glabel.key, glabel.value)]),
                    float(label_to_index[(plabel.key, plabel.value)]),
                    float(plabel.score),
                    float(ious[pidx * len(groundtruth.annotations) + gidx]),
                ]
                for pidx, pann in enumerate(prediction.annotations)
                for plabel in pann.labels
                for gidx, gann in enumerate(groundtruth.annotations)
                for glabel in gann.labels
            ]
        )


class DetectionManager:
    def __init__(self):
        self.data = None

        self.gt_cache = dict()
        self.pd_cache = dict()

        self.uid_to_index = dict()
        self.index_to_uid = dict()

        self.label_to_index = dict()
        self.index_to_label = dict()

    def add_groundtruths(
        self,
        groundtruths: list[GroundTruth],
    ):
        for groundtruth in groundtruths:

            uid = groundtruth.datum.uid

            # update datum uids
            if uid not in self.uid_to_index:
                index = len(self.uid_to_index)
                self.uid_to_index[uid] = index
                self.index_to_uid[index] = uid

            # update labels
            labels = {
                (label.key, label.value)
                for ann in groundtruth.annotations
                for label in ann.labels
                if (label.key, label.value) not in self.label_to_index
            }
            existing_total = len(self.label_to_index)
            for idx, label in enumerate(labels):
                self.label_to_index[label] = existing_total + idx
                self.index_to_label[existing_total + idx] = label

            # if a prediction is cached, preprocess the pair
            if uid in self.pd_cache:

                dt_index = self.uid_to_index[uid]
                prediction = self.pd_cache[uid]

                if self.data is None:
                    self.data = preprocess_pair(
                        dt_index, groundtruth, prediction, self.label_to_index
                    )
                else:
                    self.data = np.concatenate(
                        (
                            self.data,
                            preprocess_pair(
                                dt_index,
                                groundtruth,
                                prediction,
                                self.label_to_index,
                            ),
                        ),
                        axis=0,
                    )

            # else, cache the groundtruth
            else:
                if uid in self.gt_cache:
                    raise ValueError(
                        f"GroundTruth already exists for datum with UID {uid}."
                    )
                self.gt_cache[uid] = groundtruth

    def add_predictions(
        self,
        predictions: list[Prediction],
    ):
        for prediction in predictions:

            uid = prediction.datum.uid

            # update datum uids
            if uid not in self.uid_to_index:
                index = len(self.uid_to_index)
                self.uid_to_index[uid] = index
                self.index_to_uid[index] = uid

            # update labels
            labels = {
                (label.key, label.value)
                for ann in prediction.annotations
                for label in ann.labels
                if (label.key, label.value) not in self.label_to_index
            }
            existing_total = len(self.label_to_index)
            for idx, label in enumerate(labels):
                self.label_to_index[label] = existing_total + idx
                self.index_to_label[existing_total + idx] = label

            # if a prediction is cached, preprocess the pair
            if uid in self.gt_cache:

                dt_index = self.uid_to_index[uid]
                groundtruth = self.gt_cache[uid]

                if self.data is None:
                    self.data = preprocess_pair(
                        dt_index, groundtruth, prediction, self.label_to_index
                    )
                else:
                    self.data = np.concatenate(
                        (
                            self.data,
                            preprocess_pair(
                                dt_index,
                                groundtruth,
                                prediction,
                                self.label_to_index,
                            ),
                        ),
                        axis=0,
                    )

            # else, cache the prediction
            else:
                if uid in self.pd_cache:
                    raise ValueError(
                        f"Prediction already exists for datum with UID {uid}."
                    )
                self.pd_cache[uid] = prediction
