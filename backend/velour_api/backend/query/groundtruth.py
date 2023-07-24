from sqlalchemy import and_, or_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from velour_api import enums, exceptions, schemas
from velour_api.backend import core, models, query


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
    dataset_name: str,
    datum_uid: str,
    filter_by_task_type: list[enums.TaskType] = [],
    filter_by_metadata: list[schemas.MetaDatum] = [],
) -> schemas.GroundTruth:

    # Get dataset
    dataset = core.get_dataset(db, name=dataset_name)

    # Get datum
    datum = core.get_datum(db, uid=datum_uid)

    # Validity check
    if dataset.id != datum.dataset_id:
        raise ValueError(
            f"Datum '{datum_uid}' does not belong to dataset '{dataset_name}'."
        )

    # Get annotations with metadata
    annotations = db.query(models.Annotation).where(
        and_(
            models.Annotation.datum_id == datum.id,
            models.Annotation.model_id.is_(None),
        )
    )

    # Filter by task type
    if filter_by_task_type:
        task_filters = [
            models.Annotation.task_type == task_type.value
            for task_type in filter_by_task_type
        ]
        annotations = annotations.where(
            or_(
                *task_filters,
            )
        )

    # Filter by metadata
    if filter_by_metadata:
        metadata_filters = [
            query.compare_metadata(metadatum)
            for metadatum in filter_by_metadata
        ]
        annotations = annotations.where(
            and_(
                models.MetaDatum.annotation_id == models.Annotation.id,
                or_(*metadata_filters),
            )
        )

    return schemas.GroundTruth(
        dataset_name=dataset.name,
        datum=schemas.Datum(
            uid=datum.uid,
            metadata=query.get_metadata(db, datum=datum),
        ),
        annotations=[
            schemas.GroundTruthAnnotation(
                labels=query.get_labels(db, annotation=annotation),
                annotation=query.get_annotation(
                    db, datum=datum, annotation=annotation
                ),
            )
            for annotation in annotations.all()
        ],
    )


def get_groundtruths(
    db: Session,
    dataset_name: str = None,
) -> list[schemas.GroundTruth]:

    if dataset_name:
        dataset = core.get_dataset(db, dataset_name)
        datums = (
            db.query(models.Datum)
            .where(models.Datum.dataset_id == dataset.id)
            .all()
        )
    else:
        datums = db.query(models.Datum.all())

    return [get_groundtruth(db, datum.uid) for datum in datums]
