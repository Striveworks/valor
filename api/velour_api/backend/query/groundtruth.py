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

    # Create annotation, groundtruths, labels
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
    # retrieve from table
    dataset = core.get_dataset(db, name=dataset_name)
    datum = core.get_datum(db, dataset_id=dataset.id, uid=datum_uid)
    return schemas.GroundTruth(
        datum=schemas.Datum(
            uid=datum.uid,
            dataset=dataset_name,
            metadata=core.serialize_metadatums(datum.meta),
        ),
        annotations=core.get_annotations(db, datum),
    )


def get_groundtruths(
    db: Session,
    request: schemas.Filter,
) -> list[schemas.GroundTruth]:
    datums = ops.BackendQuery.datum().filter(request).all(db)
    return [core.get_annotations(db, datum) for datum in datums]
