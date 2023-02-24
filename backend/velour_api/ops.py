import ast

from geoalchemy2.functions import ST_Area, ST_Intersection, ST_ValueCount
from sqlalchemy.orm import Session
from sqlalchemy.sql import text

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


# SELECT ST_Area((ST_Intersection(ST_SetGeoReference(predicted_segmentation.shape, '1 0 0 1 0 0', 'GDAL'), ground_truth_segmentation.shape)).geom)
# FROM predicted_segmentation, ground_truth_segmentation


def intersection_area_of_gt_seg_and_pred_seg(
    db: Session,
    gt_seg: GroundTruthSegmentation,
    pred_seg: PredictedSegmentation,
) -> float:
    # not exactly sure why geoalchemy2.functions don't work here. they seem to have issues with mixing vector/raster types
    # see e.g. the warning here: https://geoalchemy-2.readthedocs.io/en/latest/spatial_functions.html

    return db.execute(
        text(
            f"""
        SELECT ST_Area((ST_Intersection(ST_SetGeoReference(ST_SetBandNoDataValue({PredictedSegmentation.__tablename__}.shape, 0), '1 0 0 1 0 0', 'GDAL'), {GroundTruthSegmentation.__tablename__}.shape)).geom)
        FROM {PredictedSegmentation.__tablename__}, {GroundTruthSegmentation.__tablename__}
        WHERE {PredictedSegmentation.__tablename__}.id={pred_seg.id} AND {GroundTruthSegmentation.__tablename__}.id={gt_seg.id}
        """
        )
    ).scalar()


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
