from geoalchemy2.functions import ST_Area, ST_Intersection
from sqlalchemy.orm import Session

from .models import GroundTruthDetection, PredictedDetection

DetectionType = GroundTruthDetection | PredictedDetection


def iou(db: Session, det1: DetectionType, det2: DetectionType) -> float:
    """Computes the IOU between two detections"""
    cap_area = intersection_area(db, det1, det2)
    return cap_area / (area(db, det1) + area(db, det2) - cap_area)


def intersection_area(
    db: Session, det1: DetectionType, det2: DetectionType
) -> float:
    """Computes the area of the intersection between two detections"""
    return db.scalar(ST_Area(ST_Intersection(det1.boundary, det2.boundary)))


def area(db: Session, det: DetectionType) -> float:
    """Computes the area of the intersection between two detections"""
    return db.scalar(ST_Area(det.boundary))
