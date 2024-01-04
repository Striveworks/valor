import json

from geoalchemy2.functions import ST_AsGeoJSON
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from velour_api import schemas
from velour_api.backend import core, models


def create_prediction(
    db: Session,
    prediction: schemas.Prediction,
):
    """
    Creates a prediction.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    prediction : schemas.Prediction
        The prediction to create.
    """
    # retrieve existing table entries
    model = core.fetch_model(db, name=prediction.model)
    dataset = core.fetch_dataset(db, name=prediction.datum.dataset)
    datum = core.fetch_datum(
        db, dataset_id=dataset.id, uid=prediction.datum.uid
    )

    # create labels
    all_labels = [
        label
        for annotation in prediction.annotations
        for label in annotation.labels
    ]
    label_list = core.create_labels(db=db, labels=all_labels)

    # create annotations
    annotation_list = core.create_annotations(
        db=db,
        annotations=prediction.annotations,
        datum=datum,
        model=model,
    )

    # create predictions
    label_idx = 0
    prediction_list = []
    for i, annotation in enumerate(prediction.annotations):
        indices = slice(
            label_idx, 
            label_idx + len(annotation.labels)
        )
        for j, label in enumerate(label_list[indices]):
            prediction_list.append(
                models.Prediction(
                    annotation_id=annotation_list[i].id,
                    label_id=label.id,
                    score=annotation.labels[j].score,
                )
            )
        label_idx += len(annotation.labels)

    try:
        db.add_all(prediction_list)
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise e


def get_prediction(
    db: Session,
    model_name: str,
    dataset_name: str,
    datum_uid: str,
) -> schemas.Prediction:
    """
    Fetch a prediction.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    model_name : str
        The name of the model.
    dataset_name : str
        The name of the dataset.
    datum_uid: str
        The UID of the datum to fetch.

    Returns
    ----------
    schemas.Prediction
        The requested prediction.
    """
    model = core.fetch_model(db, name=model_name)
    dataset = core.fetch_dataset(db, name=dataset_name)
    datum = core.fetch_datum(db, dataset_id=dataset.id, uid=datum_uid)
    geo_dict = (
        schemas.geojson.from_dict(
            json.loads(db.scalar(ST_AsGeoJSON(datum.geo)))
        )
        if datum.geo
        else None
    )

    return schemas.Prediction(
        model=model_name,
        datum=schemas.Datum(
            uid=datum.uid,
            dataset=dataset.name,
            metadata=datum.meta,
            geospatial=geo_dict,
        ),
        annotations=core.get_annotations(db, datum=datum, model=model),
    )
