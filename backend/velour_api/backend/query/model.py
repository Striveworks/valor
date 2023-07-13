from sqlalchemy import and_
from sqlalchemy.orm import Session

from velour_api import schemas
from velour_api.backend import core, models, subquery


def get_model(
    db: Session,
    name: str,
) -> schemas.Model:

    model = (
        db.query(models.Model).where(models.Model.name == name).one_or_none()
    )
    if not model:
        return None

    metadata = []
    for row in (
        db.query(models.MetaDatum)
        .where(models.MetaDatum.model_id == model.id)
        .all()
    ):
        if row.string_value:
            metadata.append(
                schemas.MetaDatum(name=row.name, value=row.string_value)
            )
        elif row.numeric_value:
            metadata.append(
                schemas.MetaDatum(name=row.name, value=row.numeric_value)
            )
        elif row.geo:
            metadata.append(
                schemas.MetaDatum(
                    name=row.name,
                    value=row.geo,
                )
            )
        elif row.image_id:
            if (
                image_metadatum := db.query(models.ImageMetadata)
                .where(models.ImageMetadata.id == row.image_id)
                .one_or_none()
            ):
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
    return [get_model(db, name) for name in db.query(models.Model.name).all()]


def get_prediction(
    db: Session,
    model_name: str,
    datum_uid: str,
) -> schemas.Prediction:

    # Retrieve sql models
    model = core.get_model(db, model_name)
    datum = core.get_datum(db, datum_uid)

    # Get annotations with metadata
    annotations = (
        db.query(models.Annotation)
        .where(
            and_(
                models.Annotation.model_id == model.id,
                models.Annotation.datum_id == datum.id,
            )
        )
        .all()
    )

    return schemas.Prediction(
        model_name=model.name,
        datum=schemas.Datum(
            uid=datum.uid,
            metadata=subquery.get_metadata(db, datum=datum),
        ),
        annotations=[
            schemas.PredictedAnnotation(
                annotation=subquery.get_annotation(
                    db, datum=datum, annotation=annotation
                ),
                scored_labels=subquery.get_scored_labels(
                    db, annotation=annotation
                ),
            )
            for annotation in annotations
        ],
    )


def get_predictions(
    db: Session,
    model_name: str,
) -> list[schemas.GroundTruth]:

    datums = (
        db.query(models.Datum.uid)
        .join(models.Prediction, models.Prediction.datum_id == models.Datum.id)
        .join(models.Model, models.Model.id == models.Prediction.model_id)
        .where(models.Model.name == model_name)
        .distinct()
        .all()
    )

    return [
        get_prediction(db, model_name, datum_uid[0]) for datum_uid in datums
    ]
