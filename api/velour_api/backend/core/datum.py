import json

from geoalchemy2.functions import ST_AsGeoJSON
from sqlalchemy import and_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from velour_api import exceptions, schemas
from velour_api.backend import models, ops
from velour_api.backend.core.dataset import fetch_dataset_row


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
    dataset = fetch_dataset_row(db, datum.dataset)

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
    q = ops.Query(models.Datum).filter(filters).any()
    datums = db.query(q).all()

    output = []

    for datum in datums:
        geo_dict = (
            schemas.geojson.from_dict(
                json.loads(db.scalar(ST_AsGeoJSON(datum.geo)))
            )
            if datum.geo
            else None
        )

        output.append(
            schemas.Datum(
                dataset=db.scalar(
                    select(models.Dataset.name).where(
                        models.Dataset.id == datum.dataset_id
                    )
                ),
                uid=datum.uid,
                metadata=datum.meta,
                geospatial=geo_dict,
            )
        )
    return output
