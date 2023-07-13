from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from velour_api import exceptions, schemas
from velour_api.backend import models
from velour_api.backend.core.annotation import _create_annotation
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
    uid: str,
) -> models.Datum:
    return db.query(models.Datum).where(models.Datum.uid == uid).one_or_none()


def _create_datum(
    db: Session,
    dataset: models.Dataset,
    datum: schemas.Datum,
) -> models.Datum:

    # Create datum
    try:
        row = models.Datum(
            uid=datum.uid,
            dataset_id=dataset.id,
        )
        db.add(row)
        db.commit()
    except IntegrityError:
        db.rollback()
        raise exceptions.DatumAlreadyExistsError(datum.uid)

    # Create metadata
    create_metadata(db, datum.metadata, datum=row)
    return row


def create_groundtruth(
    db: Session,
    groundtruth: schemas.GroundTruth,
):
    dataset = get_dataset(db, name=groundtruth.dataset_name)
    datum = _create_datum(db, dataset=dataset, datum=groundtruth.datum)

    rows = []
    for gt in groundtruth.annotations:
        annotation = _create_annotation(
            db,
            gt.annotation,
            datum=datum,
        )
        rows += [
            models.GroundTruth(
                annotation=annotation,
                label=label,
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
    ds = (
        db.query(models.Dataset)
        .where(models.Dataset.name == name)
        .one_or_none()
    )
    if not ds:
        raise exceptions.DatasetDoesNotExistError(name)
    try:
        db.delete(ds)
        db.commit()
    except IntegrityError:
        db.rollback()
        raise RuntimeError
