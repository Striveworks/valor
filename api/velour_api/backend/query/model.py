import json

from geoalchemy2.functions import ST_AsGeoJSON
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from velour_api import exceptions, schemas
from velour_api.backend import core, models


def create_model(
    db: Session,
    model: schemas.Model,
):
    shape = (
        schemas.GeoJSON.from_dict(data=model.geospatial).shape().wkt()
        if model.geospatial
        else None
    )

    try:
        row = models.Model(name=model.name, meta=model.metadata, geo=shape)
        db.add(row)
        db.commit()
        return row
    except IntegrityError:
        db.rollback()
        raise exceptions.ModelAlreadyExistsError(model.name)


def get_model(
    db: Session,
    name: str,
) -> schemas.Model:
    model = core.get_model(db, name=name)
    geo_dict = (
        json.loads(db.scalar(ST_AsGeoJSON(model.geo))) if model.geo else {}
    )
    return schemas.Model(
        id=model.id, name=model.name, metadata=model.meta, geospatial=geo_dict
    )


def get_models(
    db: Session,
) -> list[schemas.Model]:
    return [
        get_model(db, name) for name in db.scalars(select(models.Model.name))
    ]


def delete_model(
    db: Session,
    name: str,
):
    model = core.get_model(db, name=name)
    try:
        db.delete(model)
        db.commit()
    except IntegrityError:
        db.rollback()
        raise RuntimeError
