from geoalchemy2.functions import ST_Area, ST_Intersection
from sqlalchemy.orm import Session

from .models import Detection


def iou(session: Session, det1: Detection, det2: Detection) -> float:
    """Computes the IOU between two detections"""
    return intersection_area(session, det1, det2) / (
        area(session, det1) + area(session, det2)
    )


def intersection_area(
    session: Session, det1: Detection, det2: Detection
) -> float:
    """Computes the area of the intersection between two detections"""
    return session.scalar(
        ST_Area(ST_Intersection(det1.boundary, det2.boundary))
    )


def area(session: Session, det: Detection) -> float:
    """Computes the area of the intersection between two detections"""
    return session.scalar(ST_Area(det.boundary))


def tp_fp_counts(
    session: Session,
    detections: list[Detection],
    groundtruths: list[Detection],
    iou_thres: float,
):
    """Computes the number of true positives and false positives for detections in a single image"""
    pass
