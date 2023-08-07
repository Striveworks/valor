from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from velour_api import exceptions, schemas
from velour_api.backend import core, models, ops


def create_groundtruth(
    db: Session,
    groundtruth: schemas.GroundTruth,
):
    # create datum
    datum = core.create_datum(db, groundtruth.datum)

    rows = []
    for groundtruth_annotation in groundtruth.annotations:
        annotation = core.create_annotation(
            db,
            annotation=groundtruth_annotation,
            datum=datum,
        )
        rows += [
            models.GroundTruth(
                annotation_id=annotation.id,
                label_id=label.id,
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
    # get dataset
    dataset = core.get_dataset(db, dataset_name)

    # get datum
    datum = core.get_datum(db, datum_uid)

    # validate
    if dataset.id != datum.dataset_id:
        raise exceptions.DatumDoesNotBelongToDatasetError(
            dataset_name, datum_uid
        )

    return schemas.GroundTruth(
        datum=schemas.Datum(
            uid=datum.uid,
            dataset=dataset_name,
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
