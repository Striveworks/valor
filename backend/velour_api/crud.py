from sqlalchemy import insert, select
from sqlalchemy.orm import Session

from . import models, schemas
from .database import Base


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


def bulk_insert_and_return_ids(
    db: Session, model: Base, mappings: list[dict]
) -> list[int]:
    added_ids = db.scalars(insert(model).returning(model.id), mappings)
    db.commit()
    return added_ids.all()


def _create_detections(
    db: Session,
    detections: list[schemas.GroundTruthDetectionCreate]
    | list[schemas.PredictedDetectionCreate],
    det_model_class: type,
    labeled_det_model_class: type,
):
    # add the images to the db and add their ids to the pydantic objects
    add_pydantic_objs_to_db(
        db=db,
        model_class=models.Image,
        pydantic_objs=[detection.image for detection in detections],
    )

    # create gt detections
    det_mappings = [
        {
            "boundary": _wkt_polygon_from_detection(detection),
            "image": detection.image._db_id,
        }
        for detection in detections
    ]

    det_ids = bulk_insert_and_return_ids(db, det_model_class, det_mappings)

    # add the labels to the db and add their ids to the pydantic objects
    add_pydantic_objs_to_db(
        db=db,
        model_class=models.Label,
        pydantic_objs=[
            label for detection in detections for label in detection.labels
        ],
    )

    labeled_mappings = [
        {"detection_id": gt_det_id, "label_id": label._db_id}
        | ({"score": detection.score} if hasattr(detection, "score") else {})
        for gt_det_id, detection in zip(det_ids, detections)
        for label in detection.labels
    ]

    return bulk_insert_and_return_ids(
        db, labeled_det_model_class, labeled_mappings
    )


def create_groundtruth_detections(
    db: Session, detections: list[schemas.GroundTruthDetectionCreate]
) -> list[int]:
    return _create_detections(
        db=db,
        detections=detections,
        det_model_class=models.GroundTruthDetection,
        labeled_det_model_class=models.LabeledGroundTruthDetection,
    )


def create_predicted_detections(
    db: Session, detections: list[schemas.PredictedDetectionCreate]
) -> list[int]:
    return _create_detections(
        db=db,
        detections=detections,
        det_model_class=models.PredictedDetection,
        labeled_det_model_class=models.LabeledPredictedDetection,
    )


def get_or_create_row(
    db: Session,
    model_class: type,
    mapping: dict,
) -> int:
    """Tries to get the row defined by mapping. If that exists then
    its id is returned. Otherwise a row is created by `mapping` and the newly created
    row's id is returned
    """
    # create the query from the mapping
    where_expressions = [
        (getattr(model_class, k) == v) for k, v in mapping.items()
    ]
    where_expression = where_expressions[0]
    for exp in where_expressions[1:]:
        where_expression = where_expression & exp

    db_element = db.scalar(select(model_class).where(where_expression))

    if not db_element:
        db_element = model_class(**mapping)
        db.add(db_element)
        db.flush()
        db.commit()

    return db_element.id


def add_pydantic_objs_to_db(
    db: Session, model_class: type, pydantic_objs: list[schemas.BaseModel]
) -> None:
    for obj in pydantic_objs:
        if not hasattr(obj, "_db_id"):
            setattr(
                obj,
                "_db_id",
                get_or_create_row(
                    db=db, model_class=model_class, mapping=obj.dict()
                ),
            )
