from sqlalchemy import and_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from velour_api import exceptions, schemas
from velour_api.backend import models
from velour_api.backend.core.dataset import fetch_dataset
from velour_api.backend.ops import Query


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
    dataset = fetch_dataset(db, datum.dataset_name)

    # create datum
    try:
        row = models.Datum(
            uid=datum.uid,
            dataset_id=dataset.id,
            meta=datum.metadata,
        )
        db.add(row)
        db.commit()
    except IntegrityError:
        db.rollback()
        raise exceptions.DatumAlreadyExistsError(datum.uid)

    return row


def fetch_datum(
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


def get_datums(
    db: Session,
    filters: schemas.Filter | None = None,
) -> list[schemas.Datum]:
    """
    Fetch all datums.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    filters : schemas.Filter
        An optional filter to apply.

    Returns
    ----------
    List[schemas.Datum]
        A list of all datums.
    """
    subquery = Query(models.Datum.id).filter(filters).any()
    if subquery is None:
        raise RuntimeError("Subquery is unexpectedly None.")

    datums = (
        db.query(models.Datum).where(models.Datum.id == subquery.c.id).all()
    )
    return [
        schemas.Datum(
            dataset_name=db.scalar(
                select(models.Dataset.name).where(
                    models.Dataset.id == datum.dataset_id
                )
            ),  # type: ignore - sqlalchemy typing
            uid=datum.uid,
            metadata=datum.meta,
        )
        for datum in datums
    ]
