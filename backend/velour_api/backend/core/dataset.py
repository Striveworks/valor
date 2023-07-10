from sqlalchemy.exc import IntegrityError, InvalidRequestError
from sqlalchemy.orm import Session

from velour_api import exceptions, schemas
from velour_api.backend import models
from velour_api.backend.core.annotation import create_annotation
from velour_api.backend.core.label import create_labels
from velour_api.backend.core.metadata import create_metadata


def get_dataset(
    db: Session,
    name: str,
) -> models.Dataset:
    return (
        db.query(models.Dataset)
        .where(models.Dataset.name == name)
        .one_or_none()
    )


def get_datum(
    db: Session,
    datum: schemas.Datum,
) -> models.Datum:
    return (
        db.query(models.Datum)
        .where(models.Datum.uid == datum.uid)
        .one_or_none()
    )


def create_datum(
    db: Session,
    datum: schemas.Datum,
    dataset: models.Dataset,
) -> models.Datum:
    
    # Create datum
    try:
        row = models.Datum(uid=datum.uid, dataset_id=dataset.id)
        db.add(row)
        db.commit()
    except IntegrityError:
        db.rollback()
        raise exceptions.DatumAlreadyExistsError(datum.uid)

    # Create metadata
    create_metadata(db, datum.metadata, datum=row)
    return row


def create_groundtruths(
    db: Session,
    groundtruths: schemas.GroundTruth,
    dataset: models.Dataset,
):
    datum = create_datum(db, datum=groundtruths.datum, dataset=dataset)
    
    rows = []
    for annotation in groundtruths.annotations:
        arow = create_annotation(db, annotation.annotation)
        rows += [
            models.GroundTruth(
                datum=datum,
                annotation=arow,
                label=label,
            )
            for label in create_labels(db, annotation.labels)
        ]
    try:
        db.add_all(rows)
        db.commit()
    except IntegrityError:
        db.rollback()
        raise exceptions.GroundTruthAlreadyExistsError

    return rows


def create_dataset(
    db: Session,
    dataset: schemas.Dataset,
):
    # Check if dataset already exists.
    if (
        db.query(models.Dataset)
        .where(models.Dataset.name == dataset.name)
        .one_or_none()
    ):
        raise exceptions.DatasetAlreadyExistsError(dataset.name)

    # Create dataset
    try:
        row = models.Dataset(name=dataset.name)
        db.add(row)
        db.commit()
    except IntegrityError:
        db.rollback()
        raise exceptions.DatasetAlreadyExistsError(dataset.name)

    # Create metadata
    create_metadata(db, dataset.metadata, dataset=row)
    return row


def delete_dataset(
    db: Session,
    name: str,
):
    try:
        ds = db.query(models.Dataset).where(models.Dataset.name == name).one_or_none()
        db.delete(ds)
        db.commit()
    except IntegrityError:
        db.rollback()
        raise RuntimeError