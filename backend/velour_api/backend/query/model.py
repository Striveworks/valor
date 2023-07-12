from sqlalchemy.orm import Session

from velour_api import schemas
from velour_api.backend import models, core

from velour_api.backend.query.metadata import get_metadata
from velour_api.backend.query.annotation import get_annotation
from velour_api.backend.query.label import get_scored_labels


def get_prediction(
    db: Session,
    model_name: str,
    datum_uid: str,
) -> schemas.Prediction:

    # Get model
    dataset = core.get_model(db, model_name)
    
    # Get datum
    datum = core.get_datum(db, datum_uid=datum_uid)

    # Get annotations with metadata
    annotation_ids = (
        db.query(models.Prediction.annotation_id)
        .where(models.Prediction.datum_id == datum.id)
        .distinct()
        .subquery()
    )
    annotations = (
        db.query(models.Annotation)
        .where(models.Annotation.id.in_(annotation_ids))
        .all()
    )

    return schemas.Prediction(
        model_name=model_name,
        datum=schemas.Datum(
            uid=datum_uid,
            metadata=get_metadata(db, datum=datum),
        ),
        annotations=[
            schemas.PredictedAnnotation(
                annotation=get_annotation(db, datum=datum, annotation=annotation),
                scored_labels=get_scored_labels(db, annotation=annotation),
            )
            for annotation in annotations
        ],
    )


def get_model(
    db: Session,
    name: str,
) -> schemas.Model:
    
    model = db.query(models.Model).where(models.Model.name == name).one_or_none()
    if not model:
        return None

    metadata = []
    for row in db.query(models.MetaDatum).where(models.MetaDatum.model_id == model.id).all():
        if row.string_value:
            metadata.append(
                schemas.MetaDatum(
                    name=row.name,
                    value=row.string_value
                )
            )
        elif row.numeric_value:
            metadata.append(
                schemas.MetaDatum(
                    name=row.name,
                    value=row.numeric_value
                )
            )
        elif row.geo:
            metadata.append(
                schemas.MetaDatum(
                    name=row.name,
                    value=row.geo,
                )
            )
        elif row.image_id:
            if image_metadatum := db.query(models.ImageMetadata).where(models.ImageMetadata.id == row.image_id).one_or_none():
                metadata.append(
                    schemas.MetaDatum(
                        name=row.name,
                        value=schemas.ImageMetadata(
                            height=image_metadatum.height,
                            width=image_metadatum.width,
                            frame=image_metadatum.frame,
                        ),
                    )
                )

    return schemas.Model(id=model.id, name=model.name, metadata=metadata)


def get_models(
    db: Session,
) -> list[schemas.Model]:
    return [
        get_model(db, name)
        for name in db.query(models.Model.name).all()
    ]