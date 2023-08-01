from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from velour_api import exceptions, schemas
from velour_api.backend import core, models, ops


def create_prediction(
    db: Session,
    prediction: schemas.Prediction,
):
    model = core.get_model(db, name=prediction.model)
    if not model:
        raise exceptions.ModelDoesNotExistError(prediction.model)

    datum = core.get_datum(db, prediction.datum.uid)
    if not datum:
        raise exceptions.DatumDoesNotExistError(prediction.datum.uid)

    rows = []
    for predicted_annotation in prediction.annotations:
        annotation = core.create_annotation(
            db,
            annotation=predicted_annotation,
            datum=datum,
            model=model,
        )
        rows += [
            models.Prediction(
                annotation=annotation,
                label=core.create_label(db, scored_label.label),
                score=scored_label.score,
            )
            for scored_label in predicted_annotation.scored_labels
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

    datum = core.get_datum(db, datum_uid)
    model = core.get_model(db, model_name)

    return schemas.Prediction(
        model=model_name,
        datum=schemas.Datum(
            uid=datum.uid,
            metadata=core.get_metadata(db, datum=datum),
        ),
        annotations=core.get_scored_annotations(db, model=model, datum=datum),
    )


def get_predictions(
    db: Session,
    request: schemas.Filter,
) -> list[schemas.Prediction]:

    datums = ops.BackendQuery.datum().filter(request).all(db)

    return [core.get_scored_annotations(db, datum) for datum in datums]
