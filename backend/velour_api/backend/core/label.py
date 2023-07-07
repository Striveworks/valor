from sqlalchemy import Select, TextualSelect, and_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from velour_api import schemas
from velour_api.backend import models


def get_label(
    db: Session,
    label_id: int,
) -> schemas.Label:
    label = (
        db.query(models.Label).where(models.Label.id == label_id).one_or_none()
    )
    return schemas.Label(key=label.key, value=label.value) if label else None


def get_label_id(
    db: Session,
    label: schemas.Label,
) -> int:
    return (
        db.query(models.Label)
        .where(
            and_(
                models.Label.key == label.key,
                models.Label.value == label.value,
            )
        )
        .one_or_none()
    )


def create_label(
    db: Session,
    label: schemas.Label,
) -> models.Label:
    """Create label always commits a new label as it operates as a Many-To-One mapping."""

    # Get label if it already exists
    if (
        row := db.query(models.Label)
        .where(
            and_(
                models.Label.key == label.key,
                models.Label.value == label.value,
            )
        )
        .one_or_none()
    ):
        return row

    # Otherwise, create new label
    row = models.Label(key=label.key, value=label.value)
    try:
        db.add(row)
        db.commit()
    except IntegrityError:
        db.rollback()
        raise RuntimeError  # this should never be called
    return row


def create_labels(
    db: Session,
    labels: list[schemas.Label],
) -> list[models.Label]:
    return [create_label(label) for label in labels]


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


def query_by_labels(labels: list[schemas.Label]):
    """Returns a subquery of ground truths / predictions grouped by the list of provided labels."""
    pass
