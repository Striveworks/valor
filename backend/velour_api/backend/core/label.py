from sqlalchemy import and_, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from velour_api import enums, schemas
from velour_api.backend import models, ops


def create_label(
    db: Session,
    label: schemas.Label,
) -> models.Label:
    """Create label always commits a new label as it operates as a Many-To-One mapping."""

    # Get label if it already exists
    if row := get_label(db, label):
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
    return [create_label(db, label) for label in labels]


def get_label(
    db: Session,
    label: schemas.Label,
) -> models.Label | None:
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


def get_labels(
    db: Session,
    qf: ops.QueryFilter,
) -> list[models.Label]:
    """Returns a list of labels from a intersection of relationships (dataset, model, datum, annotation) optionally filtered by (label key, task_type)."""

    # Get annotation ids
    annotation_ids = select(models.Annotation.id)

    # Filter
    annotation_ids.where(
        and_(
            *qf.filters
        )
    )

    # Get label ids
    label_ids = (
        select(models.Label.id)
        .select_from(models.Annotation)
        .join(models.GroundTruth, models.GroundTruth.annotation_id == models.Annotation.id, full=True)
        .join(models.Prediction, models.Prediction.annotation_id == models.Annotation.id, full=True)
        .join(
            models.Label,
            or_(
                models.Label.id == models.GroundTruth.label_id,
                models.Label.id == models.Prediction.label_id,
            )
        )
        .where(models.Annotation.id.in_(annotation_ids))
    )

    # Filter by label key
    if key:
        labels = (
            db.query(models.Label)
            .where(models.Label.key == key)
            .filter(models.Label.id.in_(label_ids))
            .distinct()
            .all()
        )
    else:
        labels = (
            db.query(models.Label)
            .filter(models.Label.id.in_(label_ids))
            .distinct()
            .all()
        )

    return labels
