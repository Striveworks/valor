import ast

from geoalchemy2.functions import ST_Area, ST_Intersection, ST_ValueCount
from sqlalchemy.orm import Session

from .models import (
    GroundTruthDetection,
    PredictedDetection,
    PredictedSegmentation,
)

DetectionType = GroundTruthDetection | PredictedDetection


def iou(db: Session, det1: DetectionType, det2: DetectionType) -> float:
    """Computes the IOU between two detections"""
    cap_area = intersection_area(db, det1, det2)
    return cap_area / (det_area(db, det1) + det_area(db, det2) - cap_area)


def intersection_area(
    db: Session, det1: DetectionType, det2: DetectionType
) -> float:
    """Computes the area of the intersection between two detections"""
    return db.scalar(ST_Area(ST_Intersection(det1.boundary, det2.boundary)))


def det_area(db: Session, det: DetectionType) -> float:
    """Computes the area of the intersection between two detections"""
    return db.scalar(ST_Area(det.boundary))


def pred_seg_area(db: Session, seg: PredictedSegmentation) -> float:
    # list of the form  ['(1, N)', '(0, M)'] where N is the number of
    # pixels with value 1 and M is the number of pixels with value 0
    vcs = db.scalars(ST_ValueCount(seg.shape)).fetchall()
    # convert strings to tuples
    vcs = [ast.literal_eval(vc) for vc in vcs]

    # get value count for pixel value 1
    vc1 = [vc for vc in vcs if vc[0] == 1][0]

    return vc1[1]
