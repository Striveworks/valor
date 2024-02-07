from sqlalchemy import and_, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from velour_api import exceptions, schemas
from velour_api.backend import models
from velour_api.backend.core.annotation import (
    create_skipped_annotations,
    delete_model_annotations,
)
from velour_api.backend.core.dataset import fetch_dataset, get_dataset_status
from velour_api.backend.core.evaluation import (
    count_active_evaluations,
    delete_evaluations,
)
from velour_api.backend.core.prediction import delete_model_predictions
from velour_api.backend.ops import Query
from velour_api.enums import ModelStatus, TableStatus


def _load_model_schema(
    db: Session,
    model: models.Model,
) -> schemas.Model:
    """Convert database row to schema."""
    return schemas.Model(
        id=model.id,
        name=model.name,
        metadata=model.meta,
    )


def _fetch_disjoint_datums(
    db: Session, dataset_name: str, model_name: str
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


def create_model(
    db: Session,
    model: schemas.Model,
) -> models.Model:
    """
    Creates a model.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    model : schemas.Model
        The model to create.

    Returns
    -------
    models.Model
        The created model row.

    Raises
    ------
    exceptions.ModelAlreadyExistsError
        If a model with the provided name already exists.
    """
    try:
        row = models.Model(
            name=model.name,
            meta=model.metadata,
            status=ModelStatus.READY,
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

    Raises
    ------
    exceptions.ModelDoesNotExistError
        If a model with the provided name does not exist.
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
    model = fetch_model(db=db, name=name)
    return _load_model_schema(db=db, model=model)


def get_models(
    db: Session,
    filters: schemas.Filter | None = None,
) -> list[schemas.Model]:
    """
    Get models with optional filter constraint.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    filters : schemas.Filter, optional
        Optional filter to constrain against.

    Returns
    ----------
    list[schemas.Model]
        A list of all models.
    """
    subquery = Query(models.Model.id.label("id")).filter(filters).any()
    models_ = (
        db.query(models.Model).where(models.Model.id == subquery.c.id).all()
    )
    return [_load_model_schema(db=db, model=model) for model in models_]


def get_model_status(
    db: Session,
    dataset_name: str,
    model_name: str,
) -> TableStatus:
    """
    Get status of model.

    Parameters
    ----------
    db : Session
        The database session.
    name : str
        The name of the model.

    Returns
    -------
    enums.TableStatus
        The status of the model.
    """
    dataset = fetch_dataset(db, dataset_name)
    model = fetch_model(db, model_name)

    # format statuses
    dataset_status = TableStatus(dataset.status)
    model_status = ModelStatus(model.status)

    # check if deleting
    if model_status == ModelStatus.DELETING:
        return TableStatus.DELETING

    # check dataset status
    if dataset_status == TableStatus.DELETING:
        raise exceptions.DatasetDoesNotExistError(dataset.name)
    elif dataset_status == TableStatus.CREATING:
        return TableStatus.CREATING

    # query the number of datums that do not have any prediction annotations
    query_num_disjoint_datums = (
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

    # finalization is determined by the existence of at least one annotation per datum.
    if db.scalar(query_num_disjoint_datums) != 0:
        return TableStatus.CREATING
    else:
        return TableStatus.FINALIZED


def set_model_status(
    db: Session,
    dataset_name: str,
    model_name: str,
    status: TableStatus,
):
    """
    Sets the status of a model.

    Parameters
    ----------
    db : Session
        The database session.
    dataset_name : str
        The name of the dataset.
    model_name : str
        The name of the model.
    status : enums.TableStatus
        The desired dataset state.

    Raises
    ------
    exceptions.DatasetDoesNotExistError
        If the dataset does not exist or is being deleted.
    exceptions.ModelStateError
        If an illegal transition is requested.
    exceptions.EvaluationRunningError
        If the requested state is DELETING while an evaluation is running.
    """
    dataset_status = get_dataset_status(db, dataset_name)
    if dataset_status == TableStatus.DELETING:
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
        model_status == TableStatus.CREATING
        and status == TableStatus.FINALIZED
    ):
        if dataset_status != TableStatus.FINALIZED:
            raise exceptions.DatasetNotFinalizedError(
                dataset_name, "finalize inferences"
            )
        # edge case - check that there exists at least one prediction per datum
        create_skipped_annotations(
            db=db,
            datums=_fetch_disjoint_datums(db, dataset_name, model_name),
            model=model,
        )

    elif status == TableStatus.DELETING:
        if count_active_evaluations(
            db=db,
            model_names=[model_name],
        ):
            raise exceptions.EvaluationRunningError(
                dataset_name=dataset_name, model_name=model_name
            )

    try:
        model.status = (
            ModelStatus.READY
            if status != TableStatus.DELETING
            else ModelStatus.DELETING
        )
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

    # set status
    try:
        model.status = ModelStatus.DELETING
        db.commit()
    except Exception as e:
        db.rollback()
        raise e

    delete_evaluations(db=db, model_names=[name])
    delete_model_predictions(db=db, model=model)
    delete_model_annotations(db=db, model=model)

    try:
        db.delete(model)
        db.commit()
    except IntegrityError:
        db.rollback()
        raise RuntimeError
