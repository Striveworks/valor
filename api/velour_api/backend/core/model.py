import json

from geoalchemy2.functions import ST_AsGeoJSON
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from velour_api import exceptions, schemas
from velour_api.backend import models


def create_model(
    db: Session,
    model: schemas.Model,
):
    """
    Creates a model.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    model : schemas.Model
        The model to create.
    """
    try:
        row = models.Model(
            name=model.name,
            meta=model.metadata,
            geo=model.geospatial.wkt() if model.geospatial else None,
        )
        db.add(row)
        db.commit()
        return row
    except IntegrityError:
        db.rollback()
        raise exceptions.ModelAlreadyExistsError(model.name)


def fetch_model(
    db: Session,
    name: str,
) -> models.Model:
    """
    Fetch a model from the database.

    Parameters
    ----------
    db : Session
        The database Session you want to query against.
    name : str
        The name of the model.

    Returns
    ----------
    models.Model
        The requested model.

    """
    model = (
        db.query(models.Model).where(models.Model.name == name).one_or_none()
    )
    if model is None:
        raise exceptions.ModelDoesNotExistError(name)
    return model


def get_model(
    db: Session,
    name: str,
) -> schemas.Model:
    """
    Fetch a model.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    name : str
        The name of the model.

    Returns
    ----------
    schemas.Model
        The requested model.
    """
    model = fetch_model(db, name=name)
    geodict = (
        schemas.geojson.from_dict(
            json.loads(db.scalar(ST_AsGeoJSON(model.geo)))
        )
        if model.geo
        else None
    )
    return schemas.Model(
        id=model.id,
        name=model.name,
        metadata=model.meta,
        geospatial=geodict,
    )


def get_models(
    db: Session,
) -> list[schemas.Model]:
    """
    Fetch all models.

    Parameters
    ----------
    db : Session
        The database Session to query against.

    Returns
    ----------
    List[schemas.Model]
        A list of all models.
    """
    return [
        get_model(db, name) for name in db.scalars(select(models.Model.name))
    ]


def delete_model(
    db: Session,
    name: str,
):
    """
    Delete a model.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    name : str
        The name of the model.
    """
    model = fetch_model(db, name=name)
    try:
        db.delete(model)
        db.commit()
    except IntegrityError:
        db.rollback()
        raise RuntimeError
