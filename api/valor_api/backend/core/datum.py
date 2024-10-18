from sqlalchemy import and_, delete, desc, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from valor_api import api_utils, exceptions, schemas
from valor_api.backend import models
from valor_api.backend.core.metadata import insert_metadata, delete_metadata


def insert_datum(
    db: Session,
    dataset_id: int,
    datum: schemas.Datum,
    ignore_existing_datums: bool,
) -> list[int]:    
    """
    Inserts multiple datums from a dataset.

    Parameters
    ----------
    db : Session
        The database session.
    dataset_id : int
        The ID of the dataset that owns this datum.
    datums : list[schemas.Classification] | list[schemas.ObjectDetection] | list[schemas.SemanticSegmentation]
        The objects containing datum information.
    ignore_existing_datums : bool
        Option to ignore duplicates instead of raising an error.

    Returns
    -------
    list[int]
        The IDs of the provided datums.

    Raises
    ------
    DatumAlreadyExistsError
        If a datum with the same UID already exists for the dataset.
    """
    values = [
        {
            "dataset_id": dataset_id,
            "uid": datum.uid,
            "metadata_id": insert_metadata(db=db, metadata=datum.metadata),
        }
    ]

    try:
        insert_stmt = insert(models.Datum).values(values)

        if ignore_existing_datums:
            insert_stmt = insert_stmt.on_conflict_do_nothing(
                index_elements=["dataset_id", "uid"]
            )

        insert_stmt = insert_stmt.returning(models.Datum.id)
        row_ids = db.execute(insert_stmt).all()
        db.commit()
        return [
            row_id[0] for row_id in row_ids
        ]

    except IntegrityError as e:
        db.rollback()
        raise e
