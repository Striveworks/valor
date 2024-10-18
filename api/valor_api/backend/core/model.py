from sqlalchemy import and_, delete, desc, func, select, insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from valor_api import schemas
from valor_api.backend import models
from valor_api.backend.core.metadata import insert_metadata


def insert_model(
    db: Session,
    model: schemas.Model,
):   
    """
    Creates a entry in the model table.

    Parameters
    ----------
    db : Session
        The database session.
    model : schemas.Model
        The model schema.

    Returns
    -------
    int
        The model ID.

    Raises
    ------
    modelAlreadyExistsError
        If a datum with the same UID already exists for the model.
    """
    values = [
        {
            "name": model.name,
            "metadata_id": insert_metadata(db=db, metadata=model.metadata),
        }
    ]

    try:
        insert_stmt = (
            insert(models.Model)
            .values(values)
            .returning(models.Model.id)
        )
        row_ids = db.execute(insert_stmt).all()
        db.commit()
        return [
            row_id[0] for row_id in row_ids
        ]
    except IntegrityError as e:
        db.rollback()
        raise e


def delete_model(
    db: Session,
    model_id: int,
):
    """
    Deletes a model.

    Parameters
    ----------
    db : Session
        The database session.
    model_id : int
        The row id of the model.
    """
    try:
        db.execute(
            delete(models.Model)
            .where(models.Model.id == model_id)
        )
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise e