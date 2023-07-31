from sqlalchemy import and_, or_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from velour_api import schemas
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
        .select_from(models.Annotation)
        .join(
            models.Prediction,
            models.Prediction.annotation_id == annotation.id,
            full=True,
        )
        .join(
            models.GroundTruth,
            models.GroundTruth.annotation_id == annotation.id,
            full=True,
        )
        .join(
            models.Label,
            or_(
                models.GroundTruth.label_id == models.Label.id,
                models.Prediction.label_id == models.Label.id,
            ),
        )
        .all()
    )

    return [schemas.Label(key=label[0], value=label[1]) for label in labels]


def get_scored_labels(
    db: Session,
    annotation: models.Annotation,
) -> list[schemas.ScoredLabel]:

    scored_labels = (
        db.query(models.Prediction.score, models.Label.key, models.Label.value)
        .select_from(models.Prediction)
        .join(models.Label, models.Label.id == models.Prediction.label_id)
        .where(models.Prediction.annotation_id == annotation.id)
        .all()
    )

    return [
        schemas.ScoredLabel(
            label=schemas.Label(
                key=label[1],
                value=label[2],
            ),
            score=label[0],
        )
        for label in scored_labels
    ]
