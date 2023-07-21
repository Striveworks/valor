from sqlalchemy import and_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from velour_api import exceptions, schemas
from velour_api.backend import core, models, query


def create_prediction(
    db: Session,
    prediction: schemas.Prediction,
):
    model = core.get_model(db, name=prediction.model_name)
    if not model:
        raise exceptions.ModelDoesNotExistError(prediction.model_name)

    datum = core.get_datum(db, prediction.datum.uid)
    if not datum:
        raise exceptions.DatumDoesNotExistError(prediction.datum.uid)

    rows = []
    for pd in prediction.annotations:
        annotation = core.create_annotation(
            db,
            annotation=pd.annotation,
            datum=datum,
            model=model,
        )
        rows += [
            models.Prediction(
                annotation=annotation,
                label=core.create_label(db, scored_label.label),
                score=scored_label.score,
            )
            for scored_label in pd.scored_labels
        ]
    try:
        db.add_all(rows)
        db.commit()
    except IntegrityError:
        db.rollback()
        raise exceptions.PredictionAlreadyExistsError
    return rows


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
            metadata=query.get_metadata(db, datum=datum),
        ),
        annotations=[
            schemas.PredictedAnnotation(
                annotation=query.get_annotation(
                    db, datum=datum, annotation=annotation
                ),
                scored_labels=query.get_scored_labels(
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
