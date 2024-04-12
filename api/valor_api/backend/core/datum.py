from sqlalchemy import and_, desc, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from valor_api import exceptions, schemas
from valor_api.backend import models
from valor_api.backend.core.dataset import fetch_dataset
from valor_api.backend.query import Query


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


def get_paginated_datums(
    db: Session,
    filters: schemas.Filter | None = None,
    offset: int = 0,
    limit: int = -1,
) -> tuple[list[schemas.Datum], dict[str, str]]:
    """
    Fetch all datums.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    filters : schemas.Filter
        An optional filter to apply.
    offset : int, optional
        The start index of the items to return.
    limit : int, optional
        The number of items to return. Returns all items when set to -1.

    Returns
    ----------
    tuple[list[schemas.Datum], dict[str, str]]
        A tuple containing the datums and response headers to return to the user.
    """
    if offset < 0 or limit < -1:
        raise ValueError(
            "Offset should be an int greater than or equal to zero. Limit should be an int greater than or equal to -1."
        )

    subquery = Query(models.Datum.id).filter(filters).any()
    if subquery is None:
        raise RuntimeError("Subquery is unexpectedly None.")

    count = (
        db.query(func.count(models.Datum.id))
        .where(models.Datum.id == subquery.c.id)
        .scalar()
    )

    if offset > count:
        raise ValueError(
            "Offset is greater than the total number of items returned in the query."
        )

    # return all rows when limit is -1
    if limit == -1:
        limit = count

    datums = (
        db.query(models.Datum)
        .where(models.Datum.id == subquery.c.id)
        .order_by(desc(models.Datum.created_at))
        .offset(offset)
        .limit(limit)
        .all()
    )

    content = [
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

    if datums:
        end_index = (
            offset + len(datums) - 1
        )  # subtract one to make it zero-indexed

        range_indicator = f"{offset}-{end_index}"
    else:
        range_indicator = "*"

    headers = {"content-range": f"items {range_indicator}/{count}"}

    return (content, headers)
