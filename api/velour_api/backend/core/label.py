from sqlalchemy import Select, and_, select
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
    annotation: models.Annotation,
):
    labels = (
        db.query(models.Label.key, models.Label.value)
        .select_from(models.GroundTruth)
        .join(
            models.Label,
            models.GroundTruth.label_id == models.Label.id,
        )
        .where(models.GroundTruth.annotation_id == annotation.id)
        .all()
    )

    labels_with_score = (
        db.query(models.Label.key, models.Label.value, models.Prediction.score)
        .select_from(models.Prediction)
        .join(
            models.Label,
            models.Prediction.label_id == models.Label.id,
        )
        .where(models.Prediction.annotation_id == annotation.id)
        .all()
    )

    if labels:
        return [
            schemas.Label(key=label[0], value=label[1]) for label in labels
        ]
    elif labels_with_score:
        return [
            schemas.Label(key=label[0], value=label[1], score=label[2])
            for label in labels_with_score
        ]
    else:
        raise ValueError(
            f"no labels found for annotation with id: `{annotation.id}`"
        )


def get_dataset_labels_query(
    dataset_name: str,
    annotation_type: enums.AnnotationType,
    task_types: list[enums.TaskType],
) -> Select:
    return (
        select(models.Label)
        .join(
            models.GroundTruth,
            models.GroundTruth.label_id == models.Label.id,
        )
        .join(
            models.Annotation,
            models.Annotation.id == models.GroundTruth.annotation_id,
        )
        .join(models.Datum, models.Datum.id == models.Annotation.datum_id)
        .join(models.Dataset, models.Dataset.id == models.Dataset.id)
        .where(
            and_(
                models.Dataset.name == dataset_name,
                models.annotation_type_to_geometry[annotation_type].is_not(
                    None
                ),
                models.Annotation.task_type.in_(task_types),
            )
        )
        .distinct()
    )
