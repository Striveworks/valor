import json

from sqlalchemy.orm import Session

from . import models, schemas


def _wkt_polygon_from_detection(det: schemas.DetectionBase) -> str:
    """Returns the "Well-known text" format of a detection"""
    pts = det.boundary
    # in PostGIS polygon has to begin and end at the same point
    if pts[0] != pts[-1]:
        pts = pts + [pts[0]]
    return (
        "POLYGON (("
        + ", ".join([" ".join([str(pt[0]), str(pt[1])]) for pt in pts])
        + "))"
    )


def _list_of_points_from_wkt_polygon(
    db: Session,
    det: models.Detection,
) -> list[tuple[int, int]]:
    geo = json.loads(db.scalar(det.boundary.ST_AsGeoJSON()))
    assert len(geo["coordinates"]) == 1
    return [tuple(p) for p in geo["coordinates"][0]]


def create_groundtruth_detections(
    db: Session, detections: list[schemas.GroundTruthDetectionCreate]
) -> list[int]:
    """Adds groundtruth detections to the database"""
    mappings = [
        {
            "boundary": _wkt_polygon_from_detection(detection),
            "score": -1,
            "class_label": detection.class_label,
        }
        for detection in detections
    ]

    db.bulk_insert_mappings(models.Detection, mappings, return_defaults=True)
    db.commit()

    # need to convert to a schemas.Detection object here
    return [m["id"] for m in mappings]


def create_predicted_detections(
    db: Session, detections: list[schemas.PredictedDetectionCreate]
) -> list[int]:
    """Adds predicted detections to the database"""
    mappings = [
        {
            "boundary": _wkt_polygon_from_detection(detection),
            "score": detection.score,
            "class_label": detection.class_label,
        }
        for detection in detections
    ]
    db.bulk_insert_mappings(models.Detection, mappings, return_defaults=True)
    db.commit()

    return [m["id"] for m in mappings]
