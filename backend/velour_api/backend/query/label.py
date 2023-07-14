from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from velour_api import enums, schemas
from velour_api.backend import core, models


def get_labels(
    db: Session,
    key: str = None,
    task_type: list[enums.TaskType] = [],
    dataset: models.Dataset = None,
    model: models.Model = None,
    datum: models.Datum = None,
    annotation: models.Annotation = None,
) -> list[schemas.Label]:
    """Returns a list of labels from a union of sources (dataset, model, datum, annotation) optionally filtered by (label key, task_type)."""

    labels = core.get_labels(
        db,
        key=key,
        task_type=task_type,
        dataset=dataset,
        model=model,
        datum=datum,
        annotation=annotation,
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
