from sqlalchemy import and_, delete, desc, func, select, insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from valor_api import api_utils, enums, exceptions, schemas
from valor_api.backend import core, models

from valor_api.backend.core.metadata import insert_metadata


def insert_dataset(
    db: Session,
    dataset: schemas.Dataset,
):   
    """
    Creates a entry in the dataset table.

    Parameters
    ----------
    db : Session
        The database session.
    dataset : schemas.Dataset
        The dataset schema.

    Returns
    -------
    int
        The dataset ID.

    Raises
    ------
    DatasetAlreadyExistsError
        If a datum with the same UID already exists for the dataset.
    """
    values = [
        {
            "name": dataset.name,
            "metadata_id": insert_metadata(db=db, metadata=dataset.metadata),
        }
    ]

    try:
        insert_stmt = (
            insert(models.Dataset)
            .values(values)
            .returning(models.Dataset.id)
        )
        row_ids = db.execute(insert_stmt).all()
        db.commit()
        return [
            row_id[0] for row_id in row_ids
        ]
    except IntegrityError as e:
        db.rollback()
        raise e


def delete_dataset(
    db: Session,
    dataset_id: int,
):
    """
    Delete all datums in a dataset.

    Parameters
    ----------
    db : Session
        The database session.
    dataset_id : int
        The ID of the dataset that owns this datum.
    """
    metadata_ids = (
        select(models.Metadata.id)
        .join(
            models.Datum, 
            models.Datum.metadata_id == models.Metadata.id
        )
        .where(models.Datum.dataset_id == dataset_id)
        .cte()
    )

    try:
        db.execute(
            delete(models.String)
            .where(models.String.metadata_id == metadata_ids.c.id)
        )
        db.execute(
            delete(models.Integer)
            .where(models.Integer.metadata_id == metadata_ids.c.id)
        )
        db.execute(
            delete(models.Float)
            .where(models.Float.metadata_id == metadata_ids.c.id)
        )
        db.execute(
            delete(models.Geospatial)
            .where(models.Geospatial.metadata_id == metadata_ids.c.id)
        )
        db.execute(
            delete(models.DateTime)
            .where(models.DateTime.metadata_id == metadata_ids.c.id)
        )
        db.execute(
            delete(models.Date)
            .where(models.Date.metadata_id == metadata_ids.c.id)
        )
        db.execute(
            delete(models.Time)
            .where(models.Time.metadata_id == metadata_ids.c.id)
        )
        db.execute(
            delete(models.Metadata)
            .where(models.Metadata.id == metadata_ids.c.id)
        )
        db.execute(
            delete(models.Datum)
            .where(models.Datum.dataset_id == dataset_id)
        )
        db.execute(
            delete(models.Dataset)
            .where(models.Dataset.id == dataset_id)
        )
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise e