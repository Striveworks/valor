from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from velour_api import exceptions, schemas
from velour_api.backend import models
from velour_api.backend.core.metadata import create_metadata


def create_datum(
    db: Session,
    datum: schemas.Datum,
    dataset: models.Dataset,
) -> models.Datum:
    
    row = (
        db.query(models.Datum)
        .where(models.Datum.uid == datum.uid)
        .one_or_none()
    )
    if row:
        # datum already exists
        return row
    

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


def get_datum(
    db: Session,
    uid: str,
) -> models.Datum:
    datum = (
        db.query(models.Datum)
        .where(models.Datum.uid == uid)
        .one_or_none()
    )
    if datum is None:
        raise exceptions.DatumDoesNotExistError(uid)
    return datum


def get_dataset(
    db: Session,
    name: str,
) -> models.Dataset:
    dataset = (
        db.query(models.Dataset)
        .where(models.Dataset.name == name)
        .one_or_none()
    )
    if dataset is None:
        raise exceptions.DatasetDoesNotExistError(name)
    return dataset
