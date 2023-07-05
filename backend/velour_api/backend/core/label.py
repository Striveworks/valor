from sqlalchemy import Select, TextualSelect, select, insert, and_
from sqlalchemy.orm import Session

from velour_api.backend import models, state, x
from velour_api import schemas, exceptions


def get_label(
    db: Session,
    label_id: int,
) -> schemas.Label:
    label = db.scalar(
        select(models.Label)
        .where(models.Label.id == label_id)
    )
    return schemas.Label(key=label.key, value=label.value)


def get_label_id(
    db: Session,
    label: schemas.Label,
) -> int:
    return db.scalar(
        select(models.Label.id)
        .where(and_(models.Label.key == label.key, models.Label.value == label.value))
    )


def create_label(
    db: Session,
    label: schemas.Label,
) -> int:
    # Get id if label already exists
    existing_id = get_label_id(db, label)
    if existing_id:
        return existing_id

    # Otherwise, create new label
    mapping = {
        "key": label.key,
        "value": label.value,
    }
    added_id = db.scalar(
        insert(models.MetaDatum)
        .values(mapping)
        .returning(models.MetaDatum.id)
    )
    db.commit()
    return added_id


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
