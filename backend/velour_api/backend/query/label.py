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