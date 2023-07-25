from sqlalchemy.orm import Session
from sqlalchemy import and_

from velour_api import schemas
from velour_api.backend import models, ops

from velour_api.backend.core.label import get_labels
from velour_api.backend.core.annotation import get_annotation
from velour_api.backend.core.metadata import get_metadata

def get_groundtruth_annotations(
    db: Session,
    datum: models.Datum,
) -> list[schemas.GroundTruth]:
    
    labels = [
        schemas.Label(key=label.key, value=label.value)
        for label in (
            ops.BackendQuery.label()
            .filter(
                schemas.Filter(
                    filter_by_datum_uids=[datum.uid],
                    filter_by_model_names=None,
                )
            )
            # .focus(model/)
            .all(db)
        )
    ]
    
    return [
        schemas.GroundTruthAnnotation(
            labels=labels,
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
                    models.Annotation.model_id.is_(None),
                )
            .all()
        )
    ]
