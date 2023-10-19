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

    annotation_list, label_list = core.create_annotations_and_labels(
        db=db, annotations=prediction.annotations, datum=datum, model=model
    )

    # create tables entries
    rows = []

    for i, annotation in enumerate(annotation_list):
        for j, label in enumerate(label_list[i]):
            rows += [
                models.Prediction(
                    annotation_id=annotation.id,
                    label_id=label.id,
                    score=prediction.annotations[i].labels[j].score,
                )
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
            metadata=core.serialize_meta(datum.meta),
        ),
        annotations=core.get_annotations(db, datum=datum, model=model),
    )


def get_predictions(
    db: Session,
    request: schemas.Filter,
) -> list[schemas.Prediction]:
    datums = ops.BackendQuery.datum().filter(request).all(db)
    return [core.get_scored_annotations(db, datum) for datum in datums]
