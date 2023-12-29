from sqlalchemy import and_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from velour_api import exceptions, schemas
from velour_api.backend import models


def get_datum(
    db: Session,
    dataset_id: int,
    uid: str,
) -> models.Datum:
    """
    Fetch a datum from the database.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    dataset_id : int
        The ID of the dataset.
    uid : str
        The UID of the datum.

    Returns
    ----------
    models.Datum
        The requested datum.

    """
    datum = (
        db.query(models.Datum)
        .where(
            and_(
                models.Datum.dataset_id == dataset_id,
                models.Datum.uid == uid,
            )
        )
        .one_or_none()
    )
    if datum is None:
        raise exceptions.DatumDoesNotExistError(uid)
    return datum


def get_dataset(
    db: Session,
    name: str,
) -> models.Dataset:
    """
    Fetch a dataset from the database.

    Parameters
    ----------
    db : Session
        The database Session you want to query against.
    name : str
        The name of the dataset.

    Returns
    ----------
    models.Dataset
        The requested dataset.

    """

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
    """
    Create a datum in the database.

    Parameters
    ----------
    db : Session
        The database Session you want to query against.
    datum : schemas.Datum
        The datum to add to the database.

    Returns
    ----------
    models.Datum
        The datum.

    """
    # retrieve dataset
    dataset = get_dataset(db, datum.dataset)

    # create datum
    try:
        row = models.Datum(
            uid=datum.uid,
            dataset_id=dataset.id,
            meta=datum.metadata,
            geo=datum.geospatial.wkt() if datum.geospatial else None,
        )
        db.add(row)
        db.commit()
    except IntegrityError:
        db.rollback()
        raise exceptions.DatumAlreadyExistsError(datum.uid)

    return row
