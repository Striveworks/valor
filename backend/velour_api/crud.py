import json

from sqlalchemy.orm import Session

from . import models, schemas


def _wkt_polygon_from_detection(det: schemas.DetectionBase) -> str:
    """Returns the "Well-known text" format of a detection"""
    return (
        "POLYGON (("
        + ", ".join(
            [" ".join([str(pt[0]), str(pt[1])]) for pt in det.boundary]
        )
        + "))"
    )


def _list_of_points_from_wkt_polygon(
    db: Session,
    det: models.Detection,
) -> list[tuple[int, int]]:
    geo = json.loads(db.scalar(det.boundary.ST_AsGeoJSON()))
    assert len(geo["coordinates"]) == 1
    return [tuple(p) for p in geo["coordinates"][0]]


def create_gt_detection(
    db: Session, detection: schemas.GroundTruthDetectionCreate
) -> models.Detection:
    db_detection = models.Detection(
        boundary=_wkt_polygon_from_detection(detection),
        score=-1,
        class_label=detection.class_label,
    )

    db.add(db_detection)
    db.commit()
    db.refresh(db_detection)
    # need to convert to a schemas.Detection object here
    return schemas.GroundTruthDetection(
        boundary=_list_of_points_from_wkt_polygon(db=db, det=db_detection),
        class_label=db_detection.class_label,
        id=db_detection.id,
    )
