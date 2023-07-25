from sqlalchemy import and_, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from velour_api import enums, exceptions, schemas
from velour_api.backend import core, models, ops


def create_groundtruth(
    db: Session,
    groundtruth: schemas.GroundTruth,
):
    dataset = core.get_dataset(db, name=groundtruth.dataset_name)
    if not dataset:
        raise exceptions.DatasetDoesNotExistError(groundtruth.dataset_name)

    datum = core.create_datum(db, dataset=dataset, datum=groundtruth.datum)

    rows = []
    for gt in groundtruth.annotations:
        annotation = core.create_annotation(
            db,
            gt.annotation,
            datum=datum,
        )
        rows += [
            models.GroundTruth(
                annotation=annotation,
                label=label,
            )
            for label in core.create_labels(db, gt.labels)
        ]
    try:
        db.add_all(rows)
        db.commit()
    except IntegrityError:
        db.rollback()
        raise exceptions.GroundTruthAlreadyExistsError

    return rows


def get_groundtruth(
    db: Session,
    groundtruth: models.GroundTruth,
) -> schemas.GroundTruth:

    # Get annotation
    annotation = (
        db.query(models.Annotation)
        .where(models.Annotation.id == groundtruth.annotation_id)
        .one_or_none()
    )

    # Get datum
    datum = (
        db.query(models.Datum)
        .where(models.Datum.id == annotation.datum_id)
        .one_or_none()
    )

    # Get dataset
    dataset = (
        db.query(models.Dataset)
        .where(models.Dataset.id == datum.dataset_id)
        .one_or_none()
    )

    # Get annotations with metadata
    annotations = db.query(models.Annotation).where(
        and_(
            models.Annotation.datum_id == datum.id,
            models.Annotation.model_id.is_(None),
        )
    )

    return schemas.GroundTruth(
        dataset_name=dataset.name,
        datum=schemas.Datum(
            uid=datum.uid,
            metadata=core.get_metadata(db, datum=datum),
        ),
        annotations=[
            schemas.GroundTruthAnnotation(
                labels=core.get_labels(db, annotation=annotation),
                annotation=core.get_annotation(
                    db, datum=datum, annotation=annotation
                ),
            )
            for annotation in annotations.all()
        ],
    )


def get_groundtruths(
    db: Session,
    request: schemas.Filter,
) -> list[schemas.GroundTruth]:
    
    gts = (
        ops.BackendQuery.groundtruth()
        .filter(request)
        .all(db)
    )

    return [get_groundtruth(db, gt) for gt in gts]
