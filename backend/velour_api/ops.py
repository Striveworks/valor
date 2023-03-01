import ast

from geoalchemy2.functions import ST_Area, ST_Intersection, ST_ValueCount
from sqlalchemy.orm import Session

from .models import (
    GroundTruthDetection,
    GroundTruthSegmentation,
    PredictedDetection,
    PredictedSegmentation,
)

DetectionType = GroundTruthDetection | PredictedDetection


def iou(db: Session, det1: DetectionType, det2: DetectionType) -> float:
    """Computes the IOU between two detections"""
    cap_area = intersection_area_of_dets(db, det1, det2)
    return cap_area / (det_area(db, det1) + det_area(db, det2) - cap_area)


def intersection_area_of_dets(
    db: Session, det1: DetectionType, det2: DetectionType
) -> float:
    """Computes the area of the intersection between two detections"""
    return db.scalar(ST_Area(ST_Intersection(det1.boundary, det2.boundary)))


def _area_from_value_counts(vcs: list[str]) -> float:
    """
    vcs
        list of the form  ['(1, N)', '(0, M)'] where N is the number of
        pixels with value 1 and M is the number of pixels with value 0
    """
    # convert strings to tuples
    vcs = [ast.literal_eval(vc) for vc in vcs]

    # get value count for pixel value 1
    vc1 = [vc for vc in vcs if vc[0] == 1]

    if len(vc1) == 0:
        return 0.0

    vc1 = vc1[0]
    return vc1[1]


def intersection_area_of_gt_seg_and_pred_seg(
    db: Session,
    gt_seg: GroundTruthSegmentation,
    pred_seg: PredictedSegmentation,
) -> float:
    return _area_from_value_counts(
        db.scalars(
            ST_ValueCount(ST_Intersection(gt_seg.shape, pred_seg.shape))
        ).fetchall()
    )


def det_area(db: Session, det: DetectionType) -> float:
    """Computes the area of the intersection between two detections"""
    return db.scalar(ST_Area(det.boundary))


def seg_area(
    db: Session, seg: PredictedSegmentation | GroundTruthSegmentation
) -> float:
    return _area_from_value_counts(
        db.scalars(ST_ValueCount(seg.shape)).fetchall()
    )
