import json

from geoalchemy2.functions import ST_AsGeoJSON
from sqlalchemy import select, and_, func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from velour_api import exceptions, schemas, enums
from velour_api.backend import models
from velour_api.backend.core.dataset import fetch_dataset, get_dataset_status
from velour_api.backend.core.annotation import create_skipped_annotations
from velour_api.backend.core.evaluation import check_for_active_evaluations


def _count_disjoint_datums(
    db: Session,
    dataset_name: str,
    model_name: str
) -> int:
    """
    Count all datums that the model has not provided predictions for.

    Parameters
    ----------
    db : Session
        The database session.
    dataset_name : str
        The name of the dataset.
    model_name : str
        The name of the model.

    Returns
    -------
    int
        Number of disjoint datums.
    """
    dataset = fetch_dataset(db=db, name=dataset_name)
    model = fetch_model(db=db, name=model_name)
    disjoint_datums = (
        select(func.count())
        .select_from(models.Datum)
        .join(
            models.Annotation,
            and_(
                models.Annotation.datum_id == models.Datum.id,
                models.Annotation.model_id == model.id,
            ),
            isouter=True,
        )
        .where(models.Datum.dataset_id == dataset.id)
        .filter(models.Annotation.id.is_(None))
    )
    return db.scalar(disjoint_datums)


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
            status=enums.ModelStatus.READY,
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


def fetch_disjoint_datums(
    db: Session,
    dataset_name: str,
    model_name: str
) -> list[models.Datum]:
    """
    Fetch all datums that the model has not provided predictions for.

    Parameters
    ----------
    db : Session
        The database session.
    dataset_name : str
        The name of the dataset.
    model_name : str
        The name of the model.

    Returns
    -------
    list[models.Datum]
        List of Datums.
    """
    dataset = fetch_dataset(db=db, name=dataset_name)
    model = fetch_model(db=db, name=model_name)
    disjoint_datums = (
        select(models.Datum)
        .join(
            models.Annotation,
            and_(
                models.Annotation.datum_id == models.Datum.id,
                models.Annotation.model_id == model.id,
            ),
            isouter=True,
        )
        .where(models.Datum.dataset_id == dataset.id)
        .filter(models.Annotation.id.is_(None))
        .subquery()
    )
    return db.query(disjoint_datums).all()


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
        The database session.

    Returns
    ----------
    List[schemas.Model]
        A list of all models.
    """
    return [
        get_model(db, name) for name in db.scalars(select(models.Model.name))
    ]


def get_model_status(
    db: Session,
    dataset_name: str,
    model_name: str,
) -> enums.TableStatus:
    """
    Get status of model.

    Parameters
    ----------
    db : Session
        The database session.
    name : str
        Name of the model.

    Returns
    -------
    enums.TableStatus
    """
    # check if deleting
    model = fetch_model(db, model_name)
    if model.status == enums.ModelStatus.DELETING:
        return enums.TableStatus.DELETING
    
    # check dataset status
    dataset_status = get_dataset_status(db, dataset_name)
    if dataset_status == enums.TableStatus.DELETING:
        raise exceptions.DatasetDoesNotExistError(dataset_name)
    elif dataset_status == enums.TableStatus.CREATING:
        return enums.TableStatus.CREATING

    # check if model maps to all datums in the dataset
    number_of_disjoint_datums = _count_disjoint_datums(db, dataset_name, model_name)

    if number_of_disjoint_datums != 0:
        return enums.TableStatus.CREATING
    else:
        return enums.TableStatus.FINALIZED
    

def set_model_status(
    db: Session,
    dataset_name: str,
    model_name: str,
    status: enums.TableStatus,
):
    """
    Set the status of the model.
    """
    dataset_status = get_dataset_status(db, dataset_name)
    if dataset_status == enums.TableStatus.DELETING:
        raise exceptions.DatasetDoesNotExistError(dataset_name)

    model_status = get_model_status(db, dataset_name, model_name)
    if status == model_status:
        return
    
    model = fetch_model(db, model_name)

    # check if transition is valid
    if status not in model_status.next():
        raise exceptions.ModelStateError(model_name, model_status, status)
    
    # verify model-dataset parity
    if (
        model_status == enums.TableStatus.CREATING
        and status == enums.TableStatus.FINALIZED
    ):
        if dataset_status != enums.TableStatus.FINALIZED:
            raise exceptions.DatasetNotFinalizedError(dataset_name)
        # edge case - check that there exists at least one prediction per datum
        create_skipped_annotations(
            db=db,
            datums=fetch_disjoint_datums(db, dataset_name, model_name),
            model=model,
        )

    # TODO - write test for this after evaluation status is implemented
    elif status == enums.TableStatus.DELETING:
        if check_for_active_evaluations(db=db, model_name=model_name):
            raise exceptions.EvaluationRunningError(dataset_name=dataset_name, model_name=model_name)
        
    try:
        model.status = status
        db.commit()
    except Exception as e:
        db.rollback()
        raise e


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
    if check_for_active_evaluations(db=db, model_name=name):
        raise exceptions.EvaluationRunningError(name)
    
    try:
        model.status = enums.ModelStatus.DELETING
        db.commit()
    except Exception as e:
        db.rollback()
        raise e

    try:
        db.delete(model)
        db.commit()
    except IntegrityError:
        db.rollback()
        raise RuntimeError
