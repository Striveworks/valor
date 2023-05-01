from geoalchemy2.elements import RasterElement, WKBElement
from geoalchemy2.functions import (
    ST_Area,
    ST_Boundary,
    ST_ConvexHull,
    ST_Count,
    ST_Envelope,
    ST_Intersection,
    ST_Polygon,
)
from sqlalchemy.orm import Session

from .models import (
    GroundTruthDetection,
    GroundTruthSegmentation,
    LabeledGroundTruthDetection,
    LabeledGroundTruthSegmentation,
    LabeledPredictedDetection,
    LabeledPredictedSegmentation,
    PredictedDetection,
    PredictedSegmentation,
)

DetectionType = GroundTruthDetection | PredictedDetection
LabeledDetectionType = LabeledGroundTruthDetection | LabeledPredictedDetection

SegmentationType = GroundTruthSegmentation | PredictedSegmentation
LabeledSegmentationType = (
    LabeledGroundTruthSegmentation | LabeledPredictedSegmentation
)


def iou(
    db: Session,
    det_or_seg1: LabeledDetectionType | LabeledSegmentationType,
    det_or_seg2: LabeledDetectionType | LabeledSegmentationType,
) -> float:
    if isinstance(det_or_seg1, LabeledDetectionType) and isinstance(
        det_or_seg2, LabeledDetectionType
    ):
        return iou_two_dets(db, det_or_seg1.detection, det_or_seg2.detection)

    if isinstance(det_or_seg1, LabeledDetectionType) and isinstance(
        det_or_seg2, LabeledSegmentationType
    ):
        return iou_det_and_seg(
            db, det=det_or_seg1.detection, seg=det_or_seg2.segmentation
        )

    if isinstance(det_or_seg1, LabeledSegmentationType) and isinstance(
        det_or_seg2, LabeledDetectionType
    ):
        return iou_det_and_seg(
            db, det=det_or_seg2.detection, seg=det_or_seg1.segmentation
        )

    if isinstance(det_or_seg1, LabeledSegmentationType) and isinstance(
        det_or_seg2, LabeledSegmentationType
    ):
        return iou_two_segs(
            db, det_or_seg1.segmentation, det_or_seg2.segmentation
        )

    raise ValueError("Unexpected arguments")


def iou_two_dets(
    db: Session, det1: DetectionType, det2: DetectionType
) -> float:
    """Computes the IOU between two detections. If one is a bounding box detection
    and the other a polygon detection, then the polygon will be converted to a bounding box.
    """
    boundary1 = det1.boundary
    boundary2 = det2.boundary

    # check if any boundaries need to be converted to bboxes
    if det1.is_bbox and not det2.is_bbox:
        # convert polygon to a bounding box
        boundary2 = ST_Envelope(boundary2)
    elif not det1.is_bbox and det2.is_bbox:
        boundary1 = ST_Envelope(boundary1)

    cap_area = _intersection_area_of_det_boundaries(db, boundary1, boundary2)

    return cap_area / (
        _det_boundary_area(db, boundary1)
        + _det_boundary_area(db, boundary2)
        - cap_area
    )


def iou_two_segs(
    db: Session, seg1: SegmentationType, seg2: SegmentationType
) -> float:
    cap_area = intersection_area_of_segs(db, seg1, seg2)
    return cap_area / (
        _raster_area(db, seg1.shape) + _raster_area(db, seg2.shape) - cap_area
    )


def iou_det_and_seg(
    db: Session, det: DetectionType, seg: SegmentationType
) -> float:
    """Computes the IOU of a detection and segmentation. In the case that
    the detection is a bounding box, we take the interection of the detection
    with the bounding box that circumscribes the segmentation. If the detection is a polygon
    then we take the intersection of the detection with the convex hull of the segmentation
    """
    seg_boundary = ST_Polygon(seg.shape)
    if det.is_bbox:
        seg_boundary = ST_Envelope(seg_boundary)
    else:
        seg_boundary = ST_ConvexHull(ST_Boundary(seg_boundary))
    cap_area = _intersection_area_of_det_boundaries(
        db, det.boundary, seg_boundary
    )
    return cap_area / (
        _det_boundary_area(db, det.boundary)
        + _det_boundary_area(db, seg_boundary)
        - cap_area
    )


def _intersection_area_of_det_boundaries(
    db: Session, boundary1: WKBElement, boundary2: WKBElement
) -> float:
    """Computes the area of the intersection between two detections
    If one is a bounding box detection and the other a polygon detection, then
    the polygon will be converted to a bounding box.
    """

    return db.scalar(ST_Area(ST_Intersection(boundary1, boundary2)))


def intersection_area_of_segs(
    db: Session,
    seg1: SegmentationType,
    seg2: SegmentationType,
) -> float:
    return _intersection_area_of_rasters(db, seg1.shape, seg2.shape)


def _intersection_area_of_rasters(
    db: Session, rast1: RasterElement, rast2: RasterElement
) -> float:
    return db.scalar(ST_Count(ST_Intersection(rast1, rast2)))


def _det_boundary_area(db: Session, boundary: WKBElement) -> float:
    """Computes the area of the intersection between two detections"""
    return db.scalar(ST_Area(boundary))


def _raster_area(db: Session, rast: RasterElement) -> float:
    return db.scalar(ST_Count(rast))
