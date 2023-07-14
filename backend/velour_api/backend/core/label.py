from sqlalchemy import and_, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from velour_api import enums, schemas
from velour_api.backend import models


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
    key: str = None,
    task_type: list[enums.TaskType] = [],
    dataset: models.Dataset = None,
    model: models.Model = None,
    datum: models.Datum = None,
    annotation: models.Annotation = None,
) -> list[models.Label]:
    """Returns a list of labels from a union of sources (dataset, model, datum, annotation) optionally filtered by (label key, task_type)."""

    # Get annotation ids
    annotation_ids = select(models.Annotation.id)

    # Build relationship(s)
    relationships = []
    if annotation:
        relationships.append(models.Annotation.id == annotation.id)
    if datum:
        relationships.append(models.Annotation.datum_id == datum.id)
    if model:
        relationships.append(models.Annotation.model_id == model.id)
    if dataset:
        relationships.append(models.Datum.dataset_id == dataset.id)
        annotation_ids.join(
            models.Datum, models.Datum.id == models.Annotation.datum_id
        )

    # Add relationship(s)
    if relationships:
        annotation_ids.where(or_(*relationships))

    # Filter by task type intersection
    if task_type:
        annotation_ids.where(models.Annotation.task_type.in_(task_type))

    # Get label ids
    label_ids = (
        select(models.Label.id)
        .join(
            models.GroundTruth, models.GroundTruth.label_id == models.Label.id
        )
        .join(models.Prediction, models.Prediction.label_id == models.Label.id)
        .join(
            models.Annotation,
            or_(
                models.Annotation.id == models.GroundTruth.annotation_id,
                models.Annotation.id == models.Prediction.annotation_id,
            ),
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
