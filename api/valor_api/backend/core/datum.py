from sqlalchemy import and_, desc, func
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from valor_api import api_utils, exceptions, schemas
from valor_api.backend import models
from valor_api.backend.query import generate_select


def create_datums(
    db: Session,
    datums: list[schemas.Datum],
    datasets: list[models.Dataset],
    ignore_existing_datums: bool,
) -> list[models.Datum]:
    """Creates datums in bulk

    Parameters
    ----------
    db
        The database Session you want to query against.
    datums
        The datums to add to the database.
    datasets
        The datasets to link to the datums. This list should be the same length as the datums list.
    ignore_existing_datums
        If True, will ignore datums that already exist in the database.
        If False, will raise an error if any datums already exist.
        Default is False.

    Returns
    -------
    list[models.Datum]
        The datums that were created.
    """
    rows = [
        models.Datum(
            uid=datum.uid,
            dataset_id=dataset.id,
            meta=datum.metadata,
        )
        for datum, dataset in zip(datums, datasets)
    ]

    try:
        if not ignore_existing_datums:
            db.add_all(rows)
            db.commit()
            return rows

        values = [
            {
                "uid": row.uid,
                "dataset_id": row.dataset_id,
                "meta": row.meta,
            }
            for row in rows
        ]
        insert_stmt = (
            insert(models.Datum)
            .values(values)
            .on_conflict_do_nothing(index_elements=["dataset_id", "uid"])
            .returning(models.Datum.id, models.Datum.uid)
        )

        ids_uids = db.execute(insert_stmt)
        db.commit()
        uid_to_id = {uid: id_ for id_, uid in ids_uids}
        new_rows = []
        for row in rows:
            if row.uid in uid_to_id:
                row.id = uid_to_id[row.uid]
                new_rows.append(row)
        return new_rows

    except IntegrityError as e:
        db.rollback()
        if (
            "duplicate key value violates unique constraint" not in str(e)
            or ignore_existing_datums
        ):
            raise e

        # get existing datums
        existing_datums: list[models.Datum] = []
        for datum, dataset in zip(datums, datasets):
            try:
                existing_datums.append(fetch_datum(db, dataset.id, datum.uid))
            except exceptions.DatumDoesNotExistError:
                pass

        raise exceptions.DatumsAlreadyExistError(
            [datum.uid for datum in existing_datums]
        )


def create_datum(
    db: Session,
    datum: schemas.Datum,
    dataset: models.Dataset,
) -> models.Datum:
    """
    Create a datum in the database.

    Parameters
    ----------
    db : Session
        The database Session you want to query against.
    datum : schemas.Datum
        The datum to add to the database.
    dataset : models.Dataset
        The dataset to link to the datum.

    Returns
    ----------
    models.Datum
        The datum.

    Raises
    ----------
    exceptions.DatumAlreadyExistsError
        If the datum already exists in the database.
    """
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

    subquery = generate_select(
        models.Datum.id,
        filters=filters,
    ).subquery()
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
            uid=datum.uid,
            metadata=datum.meta,
        )
        for datum in datums
    ]

    headers = api_utils._get_pagination_header(
        offset=offset,
        number_of_returned_items=len(datums),
        total_number_of_items=count,
    )

    return (content, headers)
