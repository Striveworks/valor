from sqlalchemy import and_, delete, desc, func
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from valor_api import api_utils, exceptions, schemas
from valor_api.backend import models
from valor_api.backend.core.metadata import insert_metadata
from valor_api.backend.query import generate_select
from valor_api.enums import TableStatus


def insert_datum(
    db: Session,
    dataset_id: int,
    datum: schemas.Classification
    | schemas.ObjectDetection
    | schemas.SemanticSegmentation,
    ignore_existing_datums: bool,
) -> dict[tuple[int, str], int]:

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

        insert_stmt = insert_stmt.returning(
            models.Datum.id,
            models.Datum.dataset_id,
            models.Datum.uid,
        )

        datum_row_info = db.execute(insert_stmt).all()
        db.commit()
        return {
            (dataset_id, datum_uid): datum_id
            for datum_id, dataset_id, datum_uid in datum_row_info
        }

    except IntegrityError as e:
        db.rollback()
        raise e
