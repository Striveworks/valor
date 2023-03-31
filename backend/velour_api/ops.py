import ast

from geoalchemy2.elements import RasterElement
from geoalchemy2.functions import (
    ST_Area,
    ST_Boundary,
    ST_Envelope,
    ST_Intersection,
    ST_MakePolygon,
    ST_Polygon,
    ST_ValueCount,
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
        return iou_two_segs(db, det_or_seg1, det_or_seg2)

    raise ValueError("Unexpected arguments")


def iou_two_dets(
    db: Session, det1: DetectionType, det2: DetectionType
) -> float:
    """Computes the IOU between two detections"""
    cap_area = intersection_area_of_dets(db, det1, det2)
    return cap_area / (det_area(db, det1) + det_area(db, det2) - cap_area)


def iou_two_segs(
    db: Session, seg1: SegmentationType, seg2: SegmentationType
) -> float:
    cap_area = intersection_area_of_segs(db, seg1, seg2)
    return cap_area / (seg_area(db, seg1) + seg_area(db, seg2) - cap_area)


def iou_det_and_seg(
    db: Session, det: DetectionType, seg: SegmentationType
) -> float:
    cap_area = intersection_area_of_det_and_seg(db, det=det, seg=seg)
    return cap_area / (det_area(db, det) + seg_area(db, seg) - cap_area)


def intersection_area_of_dets(
    db: Session, det1: DetectionType, det2: DetectionType
) -> float:
    """Computes the area of the intersection between two detections
    If one is a bounding box detection and the other a polygon detection, then
    the polygon will be converted to a bounding box.
    """
    boundary1 = det1.boundary
    boundary2 = det2.boundary

    # check if any boundaries need to be converted to bboxes
    if det1.is_bbox and not det2.is_bbox:
        # convert polygon to a bounding box
        boundary2 = ST_Envelope(boundary2)
    elif not det1.is_bbox and det2.is_bbox:
        boundary1 = ST_Envelope(boundary1)

    return db.scalar(ST_Area(ST_Intersection(boundary1, boundary2)))


def intersection_area_of_det_and_seg(
    db: Session, det: DetectionType, seg: SegmentationType
):
    seg_boundary = ST_Polygon(seg.shape)
    if det.is_bbox:
        seg_boundary = ST_Envelope(seg_boundary)
    else:
        seg_boundary = ST_MakePolygon(ST_Boundary(seg_boundary))

    return db.scalar(ST_Area(ST_Intersection(det.boundary, seg_boundary)))


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


def intersection_area_of_segs(
    db: Session,
    seg1: SegmentationType,
    seg2: SegmentationType,
) -> float:
    return _intersection_area_of_rasters(db, seg1.shape, seg2.shape)


def _intersection_area_of_rasters(
    db: Session, rast1: RasterElement, rast2: RasterElement
) -> float:
    return _area_from_value_counts(
        db.scalars(ST_ValueCount(ST_Intersection(rast1, rast2))).fetchall()
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
