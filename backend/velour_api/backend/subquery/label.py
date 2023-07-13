from sqlalchemy import select
from sqlalchemy.orm import Session

from velour_api import schemas
from velour_api.backend import models


def get_labels(
    db: Session,
    key: str = None,
    dataset: models.Dataset = None,
    model: models.Model = None,
    datum: models.Datum = None,
    annotation: models.Annotation = None,
) -> list[schemas.Label]:

    # Filter by object
    if dataset:
        label_ids = (
            select(models.GroundTruth.label_id)
            .join(
                models.Annotation,
                models.Annotation.id == models.GroundTruth.annotation_id,
            )
            .join(models.Datum, models.Datum.id == models.Annotation.datum_id)
            .where(models.Datum.dataset_id == dataset.id)
        )
    elif model:
        label_ids = (
            select(models.Prediction.label_id)
            .join(
                models.Annotation,
                models.Annotation.id == models.Prediction.annotation_id,
            )
            .where(models.Annotation.model_id == model.id)
        )
    elif datum:
        gt_search = (
            select(models.GroundTruth.label_id)
            .join(
                models.Annotation,
                models.Annotation.id == models.GroundTruth.annotation_id,
            )
            .where(models.Annotation.datum_id == datum.id)
        )
        pd_search = (
            select(models.Prediction.label_id)
            .join(
                models.Annotation,
                models.Annotation.id == models.Prediction.annotation_id,
            )
            .where(models.Annotation.datum_id == datum.id)
        )
        label_ids = gt_search.union(pd_search)
    elif annotation:
        gt_search = select(models.GroundTruth.label_id).where(
            models.GroundTruth.annotation_id == annotation.id
        )
        pd_search = select(models.Prediction.label_id).where(
            models.Prediction.annotation_id == annotation.id
        )
        label_ids = gt_search.union(pd_search)
    else:
        label_ids = select(models.Label.id)

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
):
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
