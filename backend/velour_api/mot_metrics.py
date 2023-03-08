import motmetrics as mm
import numpy as np
from velour_api import schemas

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


class BoundingBox:
    def __init__(self, ymin: float, xmin: float, ymax: float, xmax: float):
        self.ymin = ymin
        self.xmin = xmin
        self.ymax = ymax
        self.xmax = xmax

    @property
    def left(self):
        return self.xmin

    @property
    def top(self):
        return self.ymin

    @property
    def width(self):
        return self.xmax - self.xmin

    @property
    def height(self):
        return self.ymax - self.ymin

    @classmethod
    def from_polygon(self, polygon: list[tuple[float, float]]):
        """Convert from polygon to BoundingBox"""
        polygon = schemas.validate_single_polygon(polygon)
        xmin = np.infty
        ymin = np.infty
        xmax = -np.infty
        ymax = -np.infty
        for coord in polygon:
            xmin = min(xmin, coord[0])
            ymin = min(ymin, coord[1])
            xmax = max(xmax, coord[0])
            ymax = max(ymax, coord[1])

        return BoundingBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)


class MOTDetection:
    """Class to convert detection data into MOT format.
    See https://arxiv.org/abs/1603.00831
    """

    def __init__(
        self,
        frame_number: int = None,
        object_id: str = None,
        bbox: BoundingBox = None,
        confidence: float = None,
    ):
        self.frame_number = frame_number
        self.object_id = object_id
        self.bbox = bbox
        self.confidence = confidence

        assert (
            confidence <= 1 and confidence >= 0
        ), "Confidence must be in [0,1]"

    def to_list(self):
        """Convert to a list as expected by MOT metrics calculators."""
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


def ground_truth_det_to_mot(
    gt: schemas.GroundTruthDetection, obj_id_to_int: dict
) -> list[float]:
    """Helper to convert a ground truth detection into MOT format"""

    for label in gt.labels:
        if label.key == OBJECT_ID_LABEL_KEY:
            break
    bbox = BoundingBox.from_polygon(gt.boundary)
    mot_det = MOTDetection(
        frame_number=gt.image.frame,
        object_id=obj_id_to_int[
            label.value
        ],  # Label's value is used as object id
        bbox=bbox,
        confidence=1,
    )

    return np.array(mot_det.to_list())


def pred_det_to_mot(
    pred: schemas.PredictedDetection, obj_id_to_int: dict
) -> list[float]:
    """Helper to convert a predicted detection into MOT format"""

    for scored_label in pred.scored_labels:
        if scored_label.label.key == OBJECT_ID_LABEL_KEY:
            break

    bbox = BoundingBox.from_polygon(pred.boundary)
    mot_det = MOTDetection(
        frame_number=pred.image.frame,
        object_id=obj_id_to_int[
            scored_label.label.value
        ],  # Label's value is used as object id
        bbox=bbox,
        confidence=scored_label.score,
    )

    return np.array(mot_det.to_list())


def compute_mot_metrics(
    predictions: list[schemas.PredictedDetection],
    groundtruths: list[schemas.GroundTruthDetection],
):
    """Compute the MOT metrics given predictions and ground truths.
    See https://arxiv.org/abs/1603.00831 for details on MOT.
    """

    # Build obj_id_to_int map
    obj_ids = set()
    for pred in predictions:
        for scored_label in pred.scored_labels:
            if scored_label.label.key == OBJECT_ID_LABEL_KEY:
                obj_ids.add(scored_label.label.value)
    for gt in groundtruths:
        for label in gt.labels:
            if label.key == OBJECT_ID_LABEL_KEY:
                obj_ids.add(label.value)
    obj_id_to_int = {_id: i for i, _id in enumerate(obj_ids)}

    # Conver to MOT format
    predicted_total = np.array(
        [pred_det_to_mot(pred, obj_id_to_int) for pred in predictions]
    )
    groundtruths_total = np.array(
        [ground_truth_det_to_mot(gt, obj_id_to_int) for gt in groundtruths]
    )

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

    return summary.to_dict(orient="records")[0]
