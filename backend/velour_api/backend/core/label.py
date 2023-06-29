from sqlalchemy import Select, TextualSelect, select
from sqlalchemy.orm import Session

from velour_api.backend import models, state
from velour_api.schemas.label import Label


def get_labels_from_query(
    db: Session, query_statement: TextualSelect
) -> list[str]:
    """Queries a provided subquery for any related labels."""
    label_ids = db.scalars(
        select(query_statement.alias().c.label_id).distinct()
    ).all()
    return db.scalars(
        select(models.Label).where(models.Label.id.in_(label_ids))
    ).all()


def get_label_keys_from_query(
    db: Session, query_statement: Select
) -> list[str]:
    """Queries a provided subquery for any related label keys."""
    return db.scalars(
        (select(models.Label.key).join(query_statement.subquery())).distinct()
    ).all()


def query_by_labels(labels: list[Label]):
    """Returns a subquery of ground truths / predictions grouped by the list of provided labels."""
    pass
