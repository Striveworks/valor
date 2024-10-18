from datetime import datetime, date, time

from typing import Any

from sqlalchemy import delete
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from valor_api import schemas
from valor_api.backend import models


def _insert_basic_metadata(
    db: Session,
    metadata_id: int,
    values: dict[str, Any],
    table: Any
):
    rows = [
        {
            "metadata_id": metadata_id,
            "key": key,
            "value": value,
        }
        for key, value in values.items()
    ]
    try:
        db.execute(insert(table).values(rows))
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise e
    

def _insert_geospatial_metdata(
    db: Session,
    metadata_id: int,
    values: dict[str, schemas.GeoJSON],
):
    rows = [
        {
            "metadata_id": metadata_id,
            "key": key,
            "value": value.to_wkt(),
        }
        for key, value in values.items()
    ]
    try:
        db.execute(insert(models.Geospatial).values(rows))
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
        _insert_basic_metadata(
            db=db,
            metadata_id=metadata_id,
            values=metadata.string,
            table=models.String,
        )
    if metadata.integer is not None:
        _insert_basic_metadata(
            db=db,
            metadata_id=metadata_id,
            values=metadata.integer,
            table=models.Integer,
        )
    if metadata.floating is not None:
        _insert_basic_metadata(
            db=db,
            metadata_id=metadata_id,
            values=metadata.floating,
            table=models.Float,
        )
    if metadata.geospatial is not None:
        _insert_geospatial_metdata(
            db=db,
            metadata_id=metadata_id,
            values=metadata.geospatial,
        )
    if metadata.datetime is not None:
        _insert_basic_metadata(
            db=db,
            metadata_id=metadata_id,
            values=metadata.datetime,
            table=models.DateTime,
        )
    if metadata.date is not None:
        _insert_basic_metadata(
            db=db,
            metadata_id=metadata_id,
            values=metadata.date,
            table=models.Date,
        )
    if metadata.time is not None:
        _insert_basic_metadata(
            db=db,
            metadata_id=metadata_id,
            values=metadata.time,
            table=models.Time,
        )

    return metadata_id


def delete_metadata(
    db: Session,
    metadata_id: int,
):
    try:
        db.execute(
            delete(models.String)
            .where(models.String.metadata_id == metadata_id)
        )
        db.execute(
            delete(models.Integer)
            .where(models.Integer.metadata_id == metadata_id)
        )
        db.execute(
            delete(models.Float)
            .where(models.Float.metadata_id == metadata_id)
        )
        db.execute(
            delete(models.Geospatial)
            .where(models.Geospatial.metadata_id == metadata_id)
        )
        db.execute(
            delete(models.DateTime)
            .where(models.DateTime.metadata_id == metadata_id)
        )
        db.execute(
            delete(models.Date)
            .where(models.Date.metadata_id == metadata_id)
        )
        db.execute(
            delete(models.Time)
            .where(models.Time.metadata_id == metadata_id)
        )
        db.execute(
            delete(models.Metadata)
            .where(models.Metadata.id == metadata_id)
        )
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise e
