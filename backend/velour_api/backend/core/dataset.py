from sqlalchemy.exc import IntegrityError, InvalidRequestError
from sqlalchemy.orm import Session

from velour_api import exceptions, schemas
from velour_api.backend import models
from velour_api.backend.core.geometry import create_geometric_annotation
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
        row = models.Datum(uid=datum.uid, dataset=dataset.id)
        db.add(row)
        db.commit()
    except IntegrityError:
        db.rollback()
        raise exceptions.DatumAlreadyExistsError(datum.uid)

    # Create metadata
    create_metadata(db, datum.metadata, datum_id=row)
    return row


def create_groundtruths(
    db: Session,
    dataset: models.Dataset,
    annotated_datums: list[schemas.AnnotatedDatum],
):
    rows = []
    for annotated_datum in annotated_datums:
        datum = create_datum(db, datum=annotated_datum.datum, dataset=dataset)
        for gt in annotated_datum.groundtruths:
            geometry = (
                create_geometric_annotation(db, gt.geometry)
                if gt.geometry
                else None
            )
            rows += [
                models.GroundTruth(
                    datum_id=datum.id,
                    task_type=gt.task_type,
                    geometry=geometry.id,
                    label=label.id,
                )
                for label in create_labels(db, gt.labels)
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
        db.delete(models.Dataset).where(models.Dataset.name == name)
        db.commit()
    except IntegrityError:
        db.rollback()
        raise RuntimeError