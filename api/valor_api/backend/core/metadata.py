from typing import Any

from sqlalchemy import and_, desc, func, or_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import InstrumentedAttribute, Query, Session

from valor_api import api_utils, schemas
from valor_api.backend import models
from valor_api.backend.query import generate_query, generate_select
from valor_api.backend.query.types import TableTypeAlias


def _insert_metadatum(
    db: Session,
    metadata_id: int,
    values: dict[str, Any],
    table: Any,
):
    items = [
        {
            "metadata_id": metadata_id,
            "key": key,
            "value": value,
        }
        for key, value in values.items()
    ]

    try:
        db.execute(insert(table).values(items))
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise e


def insert_metadata(
    db: Session,
    metadata: schemas.Metadata | None,
) -> int | None:

    if metadata is None:
        return None

    try:
        metadata_id = db.execute(
            insert(models.Metadata).returning(models.Metadata.id)
        ).scalar()
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise e

    if not isinstance(metadata_id, int):
        raise RuntimeError

    if metadata.string:
        _insert_metadatum(
            db=db,
            metadata_id=metadata_id,
            values=metadata.string,
            table=models.String,
        )
    if metadata.integer is not None:
        _insert_metadatum(
            db=db,
            metadata_id=metadata_id,
            values=metadata.integer,
            table=models.Integer,
        )
    if metadata.floating is not None:
        _insert_metadatum(
            db=db,
            metadata_id=metadata_id,
            values=metadata.floating,
            table=models.Float,
        )
    if metadata.geospatial is not None:
        _insert_metadatum(
            db=db,
            metadata_id=metadata_id,
            values=metadata.geospatial,
            table=models.Geospatial,
        )
    if metadata.datetime is not None:
        _insert_metadatum(
            db=db,
            metadata_id=metadata_id,
            values=metadata.datetime,
            table=models.DateTime,
        )
    if metadata.date is not None:
        _insert_metadatum(
            db=db,
            metadata_id=metadata_id,
            values=metadata.date,
            table=models.Date,
        )
    if metadata.time is not None:
        _insert_metadatum(
            db=db,
            metadata_id=metadata_id,
            values=metadata.time,
            table=models.Time,
        )

    return metadata_id
