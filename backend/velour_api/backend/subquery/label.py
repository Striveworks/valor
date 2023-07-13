from sqlalchemy.orm import Session

from velour_api import schemas
from velour_api.backend import models, core


def get_labels(
    db: Session,
    key: str = None,
    dataset: models.Dataset = None,
    model: models.Model = None,
    datum: models.Datum = None,
    annotation: models.Annotation = None,
) -> list[schemas.Label]:
    
    if key:
        labels = (
            db.query(models.Label)
            .where(models.Label.key == key)
            .subquery()
        )
    elif dataset or model or datum or annotation:
        if dataset:
            label_ids = (
                db.query(models.GroundTruth.label_id)
                .join(models.Datum, models.Datum.id == models.GroundTruth.datum_id)
                .where(models.Datum.dataset_id == dataset.id)
                .distinct()
                .subquery()
            )
        elif model:
            label_ids = (
                db.query(models.Prediction.label_id)
                .where(models.Prediction.model_id == model.id)
                .distinct()
                .subquery()
            )
        elif datum:
            # @TODO
            raise NotImplementedError
        elif annotation:
            # @TODO
            raise NotImplementedError

        labels = (
            db.query(models.Label)
            .where(models.Label.id.in_(label_ids))
            .subquery()
        )
    else:
        labels = db.query(models.Label).subquery()

    return labels


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