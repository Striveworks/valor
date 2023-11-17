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
    shape = (
        schemas.GeoJSON.from_dict(data=dataset.geospatial).shape().wkt()
        if dataset.geospatial
        else None
    )

    try:
        row = models.Dataset(
            name=dataset.name, meta=dataset.metadata, geo=shape
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
    # retrieve dataset
    dataset = core.get_dataset(db, name=name)
    geo_dict = (
        json.loads(db.scalar(ST_AsGeoJSON(dataset.geo))) if dataset.geo else {}
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
    return [
        get_dataset(db, name)
        for name in db.scalars(select(models.Dataset.name)).all()
    ]


def get_datums(
    db: Session,
    filters: schemas.Filter | None = None,
) -> list[schemas.Datum]:
    """Get datums, optional filter."""
    q = ops.Query(models.Datum).filter(filters).any()
    datums = db.query(q).all()

    output = []

    for datum in datums:
        geo_dict = (
            json.loads(db.scalar(ST_AsGeoJSON(datum.geo))) if datum.geo else {}
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
    dataset = core.get_dataset(db, name=name)
    try:
        db.delete(dataset)
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise e
