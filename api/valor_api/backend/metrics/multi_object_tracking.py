from collections import OrderedDict

import motmetrics as mm
import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame

from valor_api import schemas

OBJECT_ID_LABEL_KEY = "object_id"
MOT_METRICS_NAMES = [
    "num_frames",
    "idf1",
    "idp",
    "idr",
    "recall",
    "precision",
    "num_objects",
    "mostly_tracked",
    "partially_tracked",
    "mostly_lost",
    "num_false_positives",
    "num_misses",
    "num_switches",
    "num_fragmentations",
    "mota",
    "motp",
]


class MOTDetection:
    """
    Class to convert detection data into multi-object tracking (MOT) format. See https://arxiv.org/abs/1603.00831

    Parameters
    ----------
    frame_number: int
        The frame number for a video frame.
    object_id: str
        The id of the object
    bbox: schemas.BoundingBox
        A bounding box for the frame.
    confidence: float
        The confidence level for the bounding box.
    """

    def __init__(
        self,
        frame_number: int | float,
        object_id: str,
        bbox: schemas.BoundingBox,
        confidence: float,
    ):
        self.frame_number = frame_number
        self.object_id = object_id
        self.bbox = bbox
        self.confidence = confidence

        assert (
            confidence <= 1 and confidence >= 0
        ), "Confidence must be in [0,1]"

    def to_list(self) -> list[int | float | str]:
        """
        Convert the MOT object to a list as expected by MOT metrics calculators.

        Returns
        ----------
        List[int | float | str]
            A list of MOT attributes.
        """
        return [
            self.frame_number,
            self.object_id,
            self.bbox.left,
            self.bbox.top,
            self.bbox.width,
            self.bbox.height,
            self.confidence,
            -1,
            -1,
            -1,
        ]


def _ground_truth_det_to_mot(
    datum: schemas.Datum,
    gt: schemas.Annotation,
    obj_id_to_int: dict,
) -> NDArray:
    """Helper to convert a ground truth detection into MOT format"""
    if "frame" not in datum.metadata:
        raise ValueError("Datum does not contain a video frame number.")
    if not gt.labels:
        raise ValueError("GroundTruth does not contain labels.")

    for label in gt.labels:
        if label.key == OBJECT_ID_LABEL_KEY:
            break
    bbox = gt.bounding_box

    if not bbox:
        raise ValueError("GroundTruth is missing bounding box.")

    mot_det = MOTDetection(
        frame_number=datum.metadata["frame"],  # type: ignore - we don't need to explicitely type the "frame" key of the metadata dict
        object_id=obj_id_to_int[
            label.value  # type: ignore - label shouldn't be unbound if gt.labels isn't empty
        ],  # Label's value is used as object id
        bbox=bbox,
        confidence=1,
    )
    return np.array(mot_det.to_list())


def _pred_det_to_mot(
    datum: schemas.Datum,
    pred: schemas.Annotation,
    obj_id_to_int: dict,
    object_id_label_key: str = OBJECT_ID_LABEL_KEY,
) -> NDArray:
    """Helper to convert a predicted detection into MOT format"""
    if "frame" not in datum.metadata:
        raise ValueError("Datum does not contain a video frame number.")
    if not pred.labels:
        raise ValueError("Prediction does not contain labels.")
    for scored_label in pred.labels:
        if scored_label.key == object_id_label_key:
            break

    bbox = pred.bounding_box

    if not bbox:
        raise ValueError("Prediction is missing bounding box.")

    mot_det = MOTDetection(
        frame_number=datum.metadata["frame"],  # type: ignore - we don't need to explicitely type the "frame" key of the metadata dict
        object_id=obj_id_to_int[
            scored_label.value  # type: ignore - label shouldn't be unbound if gt.labels isn't empty
        ],
        bbox=bbox,
        confidence=scored_label.score,  # type: ignore - label shouldn't be unbound if pred.labels isn't empty
    )
    return np.array(mot_det.to_list())


def compute_mot_metrics(
    predictions: list[schemas.Prediction],
    groundtruths: list[schemas.GroundTruth],
) -> DataFrame | dict | OrderedDict:
    """
    Compute the multi-object tracking (MOT) metrics given predictions and groundtruths. See https://arxiv.org/abs/1603.00831 for details on MOT.

    Parameters
    ----------
    predictions: list[schemas.Prediction]
        A list of predictions.
    groundtruths: list[schemas.GroundTruth]
        A list of groundtruths.

    Returns
    ----------
    dict
        A dictionary of MOT metrics.
    """

    # Build obj_id_to_int map
    obj_ids = set()
    for annotated_datum in groundtruths:
        for gt in annotated_datum.annotations:
            for label in gt.labels:
                if label.key == OBJECT_ID_LABEL_KEY:
                    obj_ids.add(label.value)
    for annotated_datum in predictions:
        for pd in annotated_datum.annotations:
            for scored_label in pd.labels:
                if scored_label.key == OBJECT_ID_LABEL_KEY:
                    obj_ids.add(scored_label.value)
    obj_id_to_int = {_id: i for i, _id in enumerate(obj_ids)}

    # Convert to MOT format
    gt_mots = []
    for annotated_datum in groundtruths:
        gt_mots.extend(
            [
                _ground_truth_det_to_mot(
                    annotated_datum.datum, gt, obj_id_to_int
                )
                for gt in annotated_datum.annotations
            ]
        )
    groundtruths_total = np.array(gt_mots)
    pd_mots = []
    for annotated_datum in predictions:
        pd_mots.extend(
            [
                _pred_det_to_mot(annotated_datum.datum, pred, obj_id_to_int)
                for pred in annotated_datum.annotations
            ]
        )
    predicted_total = np.array(pd_mots)

    acc = mm.MOTAccumulator(auto_id=True)

    for frame in range(int(groundtruths_total[:, 0].max())):
        frame += 1

        gt_dets = groundtruths_total[
            groundtruths_total[:, 0] == frame, 1:6
        ]  # select all detections in groundtruths_total
        t_dets = predicted_total[
            predicted_total[:, 0] == frame, 1:6
        ]  # select all detections in predicted_total

        C = mm.distances.iou_matrix(gt_dets[:, 1:], t_dets[:, 1:], max_iou=0.5)

        acc.update(
            gt_dets[:, 0].astype("int").tolist(),
            t_dets[:, 0].astype("int").tolist(),
            C,
        )

    mh = mm.metrics.create()

    summary = mh.compute(
        acc,
        metrics=MOT_METRICS_NAMES,
        name="acc",
    )

    return summary
