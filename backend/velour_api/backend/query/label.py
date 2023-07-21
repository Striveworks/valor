from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from velour_api import enums, schemas
from velour_api.backend import core, models


def get_label(
    db: Session,
    label: schemas.Label
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
    dataset: models.Dataset = None,
    model: models.Model = None,
    datum: models.Datum = None,
    annotation: models.Annotation = None,
    filter_by_key: list[str] = [],
    filter_by_task_type: list[enums.TaskType] = [],
    filter_by_annotation_type: list[enums.AnnotationType] = [],
    filter_by_metadata: list[schemas.MetaDatum] = [],
) -> list[schemas.Label]:
    """Returns a list of unique labels from a union of sources (dataset, model, datum, annotation) optionally filtered by (label key, task_type)."""

    labels = core.get_labels(
        db,
        dataset=dataset,
        model=model,
        datum=datum,
        annotation=annotation,
        filter_by_key=filter_by_key,
        filter_by_task_type=filter_by_task_type,
        filter_by_annotation_type=filter_by_annotation_type,
        filter_by_metadata=filter_by_metadata,
    )

    return [
        schemas.Label(
            key=label.key,
            value=label.value,
        )
        for label in labels
    ]


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