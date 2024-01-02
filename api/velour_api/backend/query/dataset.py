import json

from geoalchemy2.functions import ST_AsGeoJSON
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from velour_api import exceptions, schemas
from velour_api.backend import core, models, ops


def create_dataset(
    db: Session,
    dataset: schemas.Dataset,
):
    """
    Creates a dataset.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    dataset : schemas.Dataset
        The dataset to create.
    """
    try:
        row = models.Dataset(
            name=dataset.name,
            meta=dataset.metadata,
            geo=dataset.geospatial.wkt() if dataset.geospatial else None,
        )
        db.add(row)
        db.commit()
        return row
    except IntegrityError:
        db.rollback()
        raise exceptions.DatasetAlreadyExistsError(dataset.name)


def get_dataset(
    db: Session,
    name: str,
) -> schemas.Dataset:
    """
    Fetch a dataset.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    name : str
        The name of the dataset.

    Returns
    ----------
    schemas.Dataset
        The requested dataset.
    """
    dataset = core.get_dataset(db, name=name)
    geo_dict = (
        schemas.geojson.from_dict(
            json.loads(db.scalar(ST_AsGeoJSON(dataset.geo)))
        )
        if dataset.geo
        else None
    )
    return schemas.Dataset(
        id=dataset.id,
        name=dataset.name,
        metadata=dataset.meta,
        geospatial=geo_dict,
    )


def get_datasets(
    db: Session,
) -> list[schemas.Dataset]:
    """
    Fetch all datasets.

    Parameters
    ----------
    db : Session
        The database Session to query against.

    Returns
    ----------
    List[schemas.Dataset]
        A list of all datasets.
    """
    return [
        get_dataset(db, name)
        for name in db.scalars(select(models.Dataset.name)).all()
    ]


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


def delete_dataset(
    db: Session,
    name: str,
):
    """
    Delete a dataset.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    name : str
        The name of the dataset.
    """
    dataset = core.get_dataset(db, name=name)
    try:
        db.delete(dataset)
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise e
