from sqlalchemy.orm import Session

from velour_api import schemas
from velour_api.backend import models


def get_labels(
    db: Session,
    dataset: models.Dataset = None,
    model: models.Model = None,
    datum: models.Datum = None,
    annotation: models.Annotation = None,
) -> list[schemas.Label]:
    
    if annotation is not None:
        label_ids = (
            db.query(models.GroundTruth.label_id)
            .where(models.GroundTruth.annotation_id == annotation.id)
            .distinct()
            .subquery()
        )
        labels = (
            db.query(models.Label)
            .where(models.Label.id.in_(label_ids))
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