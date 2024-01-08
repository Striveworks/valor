import json

from geoalchemy2.functions import ST_AsGeoJSON
from sqlalchemy import and_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from velour_api import enums, exceptions, schemas
from velour_api.backend import models
from velour_api.backend.core.evaluation import check_for_active_evaluations


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
            status=enums.TableStatus.CREATING,
        )
        db.add(row)
        db.commit()
        return row
    except IntegrityError:
        db.rollback()
        raise exceptions.DatasetAlreadyExistsError(dataset.name)


def fetch_dataset(
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
        .where(
            and_(
                models.Dataset.name == name
                and models.Dataset.status != enums.TableStatus.DELETING
            )
        )
        .one_or_none()
    )
    if dataset is None:
        raise exceptions.DatasetDoesNotExistError(name)
    return dataset


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
    dataset = fetch_dataset(db, name=name)
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


def get_dataset_status(
    db: Session,
    name: str,
) -> enums.TableStatus:
    """
    Get the status of a dataset.
    """
    dataset = fetch_dataset(db, name)
    return enums.TableStatus(dataset.status)


def set_dataset_status(
    db: Session,
    name: str,
    status: enums.TableStatus,
):
    """
    Sets the status of a dataset.
    """
    dataset = fetch_dataset(db, name)
    active_status = enums.TableStatus(dataset.status)
    
    if status == active_status:
        return
    
    if status not in active_status.next():
        raise exceptions.DatasetStateError(name, active_status, status)
    
    # TODO - write test for this after evaluation status is implemented
    if status == enums.TableStatus.DELETING:
        if check_for_active_evaluations(db=db, dataset_name=name):
            raise exceptions.EvaluationRunningError(dataset_name=name)

    try:
        dataset.status = status
        db.commit()
    except Exception as e:
        db.rollback()
        raise e


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
    if check_for_active_evaluations(db=db, model_name=name):
        raise exceptions.EvaluationRunningError(name)
    set_dataset_status(db, name, enums.TableStatus.DELETING)
    dataset = fetch_dataset(db, name=name)
    
    try:
        db.delete(dataset)
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise e
