from typing import Any

from sqlalchemy import and_, desc, func, or_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import InstrumentedAttribute, Query, Session

from valor_api import api_utils, schemas
from valor_api.backend import models
from valor_api.backend.query import generate_query, generate_select
from valor_api.backend.query.types import TableTypeAlias


def insert_labels(db: Session, labels: list[str]) -> list[int]:
    try:
        label_ids = db.execute(
            insert(models.Label)
            .values([{"value": label} for label in labels])
            .on_conflict_do_nothing(index_elements=["value"])
            .returning(models.Label.id)
        ).all()
        db.commit()
        return [label_id[0] for label_id in label_ids]

    except IntegrityError as e:
        db.rollback()
        raise e
