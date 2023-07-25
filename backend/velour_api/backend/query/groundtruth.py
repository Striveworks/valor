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
    dataset_name: str,
    datum_uid: str,
):
    dataset = core.get_dataset(db, dataset_name)
    datum = core.get_datum(db, datum_uid)

    # validation
    try:
        assert datum.dataset_id == dataset.id
    except AssertionError:
        raise ValueError(f"Datum `{datum_uid}` exists for different dataset.")

    return schemas.GroundTruth(
        dataset_name=dataset_name,
        datum=schemas.Datum(
            uid=datum.uid,
            metadata=core.get_metadata(db, datum=datum),
        ),
        annotations=core.get_groundtruth_annotations(db, datum)
    )


def get_groundtruths(
    db: Session,
    request: schemas.Filter,
) -> list[schemas.GroundTruth]:
    
    datums = (
        ops.BackendQuery.datum()
        .filter(request)
        .all(db)
    )

    return [core.get_groundtruth(db, datum) for datum in datums]
