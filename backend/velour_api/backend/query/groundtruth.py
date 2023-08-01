from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from velour_api import exceptions, schemas
from velour_api.backend import core, models, ops


def create_groundtruth(
    db: Session,
    groundtruth: schemas.GroundTruth,
):
    dataset = core.get_dataset(db, name=groundtruth.dataset)
    if not dataset:
        raise exceptions.DatasetDoesNotExistError(groundtruth.dataset)

    datum = core.create_datum(db, dataset=dataset, datum=groundtruth.datum)

    rows = []
    for groundtruth_annotation in groundtruth.annotations:
        annotation = core.create_annotation(
            db,
            annotation=groundtruth_annotation,
            datum=datum,
        )
        rows += [
            models.GroundTruth(
                annotation=annotation,
                label=label,
            )
            for label in core.create_labels(db, groundtruth_annotation.labels)
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
        dataset=dataset_name,
        datum=schemas.Datum(
            uid=datum.uid,
            metadata=core.get_metadata(db, datum=datum),
        ),
        annotations=core.get_annotations(db, datum),
    )


def get_groundtruths(
    db: Session,
    request: schemas.Filter,
) -> list[schemas.GroundTruth]:

    datums = ops.BackendQuery.datum().filter(request).all(db)

    return [core.get_annotations(db, datum) for datum in datums]
