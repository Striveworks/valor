from sqlalchemy.orm import Session
from sqlalchemy import and_

from velour_api import schemas
from velour_api.backend import models, ops

from velour_api.backend.core.label import get_labels
from velour_api.backend.core.annotation import get_annotation
from velour_api.backend.core.metadata import get_metadata

def get_prediction_annotations(
    db: Session,
    datum: models.Datum,
) -> list[schemas.PredictedAnnotation]:

    return [
        schemas.PredictedAnnotation(
            scored_labels=list(
                {
                    schemas.ScoredLabel(
                        label=schemas.Label(
                            key=scored_label[1], 
                            value=scored_label[2],
                        ),
                        score=scored_label[0],
                    )
                    for scored_label in (
                        db.query(models.Prediction.score, models.Label.key, models.Label.value)
                        .select_from(models.Label)
                        .join(models.Prediction, models.Prediction.label_id == models.Label.id)
                        .join(models.Annotation, models.Annotation.id == models.Prediction.annotation_id)
                        .where(models.Annotation.id == annotation.id)
                        .all()
                    )
                }
            ),
            annotation=get_annotation(
                db,
                datum=datum,
                annotation=annotation,
            ),
        )
        for annotation in (
            db.query(models.Annotation)
            .where(
                and_(
                    models.Annotation.datum_id == datum.id),
                    models.Annotation.model_id.isnot(None),
                )
            .all()
        )
    ]
