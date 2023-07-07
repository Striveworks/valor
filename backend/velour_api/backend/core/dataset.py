from sqlalchemy.exc import IntegrityError
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


def create_dataset_info(
    db: Session,
    info: schemas.DatasetInfo,
) -> models.Dataset:

    # Create dataset
    row = models.Dataset(name=info.name)
    try:
        db.add(row)
        db.commit()
    except IntegrityError:
        db.rollback()
        raise exceptions.DatasetAlreadyExistsError(info.name)

    # Create metadata
    create_metadata(db, info.metadata, dataset=row)
    return row


def create_datum(
    db: Session,
    datum: schemas.Datum,
    dataset: models.Dataset,
) -> models.Datum:

    # Create datum
    row = models.Datum(uid=datum.uid, dataset=dataset.id)
    try:
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
        not db.query(models.Dataset)
        .where(models.Dataset.name == dataset.info.name)
        .one_or_none()
    ):
        raise exceptions.DatasetAlreadyExistsError(dataset.info.name)

    row = create_dataset_info(db, dataset.info)
    create_groundtruths(db, dataset=row, annotated_datums=dataset.datums)


def delete_dataset(
    db: Session,
    name: str,
):
    pass