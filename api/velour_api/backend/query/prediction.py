from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from velour_api import schemas
from velour_api.backend import core, models, ops


def create_prediction(
    db: Session,
    prediction: schemas.Prediction,
):
    # retrieve existing table entries
    model = core.get_model(db, name=prediction.model)
    dataset = core.get_dataset(db, name=prediction.datum.dataset)
    datum = core.get_datum(db, dataset_id=dataset.id, uid=prediction.datum.uid)

    # create tables entries
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
                annotation_id=annotation.id,
                label_id=core.create_label(db, scored_label).id,
                score=scored_label.score,
            )
            for scored_label in predicted_annotation.labels
        ]
    try:
        db.add_all(rows)
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise e
    return rows


def get_prediction(
    db: Session,
    model_name: str,
    dataset_name: str,
    datum_uid: str,
) -> schemas.Prediction:
    """Returns prediction schema."""
    model = core.get_model(db, name=model_name)
    dataset = core.get_dataset(db, name=dataset_name)
    datum = core.get_datum(db, dataset_id=dataset.id, uid=datum_uid)
    return schemas.Prediction(
        model=model_name,
        datum=schemas.Datum(
            uid=datum.uid,
            dataset=dataset.name,
            metadata=core.get_metadata(db, datum=datum),
        ),
        annotations=core.get_annotations(db, datum=datum, model=model),
    )


def get_predictions(
    db: Session,
    request: schemas.Filter,
) -> list[schemas.Prediction]:
    datums = ops.BackendQuery.datum().filter(request).all(db)
    return [core.get_scored_annotations(db, datum) for datum in datums]
