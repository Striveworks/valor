import motmetrics as mm
import numpy as np
import schemas


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


def ground_truth_det_to_mot(gt: schemas.GroundTruthDetection) -> list[float]:
    """Helper to convert a ground truth detection into MOT format"""
    mot_lines = []

    for label in gt.labels:
        bbox = BoundingBox.from_polygon(gt.boundary)
        mot_det = MOTDetection(
            frame_number=gt.image.frame,
            object_id=label.value,  # Label's value is used as object id
            bbox=bbox,
            confidence=1,
        )
        mot_lines.append(mot_det.to_list())

    return np.array(mot_lines)


def pred_det_to_mot(pred: schemas.PredictedDetection) -> list[float]:
    """Helper to convert a predicted detection into MOT format"""
    mot_lines = []

    for scored_label in pred.scored_labels:
        bbox = BoundingBox.from_polygon(pred.boundary)
        mot_det = MOTDetection(
            frame_number=pred.image.frame,
            object_id=scored_label.label.value,  # Label's value is used as object id
            bbox=bbox,
            confidence=scored_label.score,
        )
        mot_lines.append(mot_det.to_list())

    return np.array(mot_lines)


def compute_mot_metrics(
    predictions: list[list[schemas.PredictedDetection]],
    groundtruths: list[list[schemas.GroundTruthDetection]],
):
    """Compute the MOT metrics given predictions and ground truths.
    See https://arxiv.org/abs/1603.00831 for details on MOT.
    """
    predicted_total = np.concatenate(list(map(pred_det_to_mot, predictions)))
    groundtruths_total = np.concatenate(
        list(map(ground_trutn_det_to_mot, groundtruths))
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

        C = mm.distances.iou_matrix(
            gt_dets[:, 1:], t_dets[:, 1:], max_iou=0.5
        )

        acc.update(
            gt_dets[:, 0].astype("int").tolist(),
            t_dets[:, 0].astype("int").tolist(),
            C,
        )

    mh = mm.metrics.create()

    summary = mh.compute(
        acc,
        metrics=[
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
        ],
        name="acc",
    )

    return summary
