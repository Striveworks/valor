from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from velour_api import exceptions, schemas
from velour_api.backend import models
from velour_api.backend.core.metadata import create_metadata


def get_datum(
    db: Session,
    uid: str,
) -> models.Datum:
    datum = db.query(models.Datum).where(models.Datum.uid == uid).one_or_none()
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


def create_datum(
    db: Session,
    datum: schemas.Datum,
) -> models.Datum:

    # check if datum already exists
    row = (
        db.query(models.Datum)
        .where(models.Datum.uid == datum.uid)
        .one_or_none()
    )
    if row is not None:
        return row

    # get dataset
    dataset = get_dataset(db, datum.dataset)

    # create datum
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

    # create metadata
    create_metadata(db, datum.metadata, datum=row)

    return row
