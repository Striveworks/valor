import ast

from geoalchemy2.functions import ST_Area, ST_Intersection, ST_ValueCount
from sqlalchemy.orm import Session
from sqlalchemy.sql import text

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
    """Computes the area of the intersection between two detections"""
    return db.scalar(ST_Area(ST_Intersection(det1.boundary, det2.boundary)))


def intersection_area_of_det_and_seg(
    db: Session, det: DetectionType, seg: SegmentationType
):
    # not exactly sure why geoalchemy2.functions don't work here. they seem to have issues with mixing vector/raster types
    # see e.g. the warning here: https://geoalchemy-2.readthedocs.io/en/latest/spatial_functions.html

    return db.execute(
        text(
            f"""
        SELECT ST_Area((ST_Intersection(ST_SetGeoReference(ST_SetBandNoDataValue({seg.__class__.__tablename__}.shape, 0), '1 0 0 1 0 0', 'GDAL'), {det.__class__.__tablename__}.boundary)).geom)
        FROM {seg.__class__.__tablename__}, {det.__class__.__tablename__}
        WHERE {seg.__class__.__tablename__}.id={seg.id} AND {det.__class__.__tablename__}.id={det.id}
        """
        )
    ).scalar()


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
    return _area_from_value_counts(
        db.scalars(
            ST_ValueCount(ST_Intersection(seg1.shape, seg2.shape))
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
