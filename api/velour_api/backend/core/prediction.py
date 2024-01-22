import json

from geoalchemy2.functions import ST_AsGeoJSON
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from velour_api import enums, exceptions, schemas
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
    # check model status
    model_status = core.get_model_status(
        db=db,
        dataset_name=prediction.datum.dataset_name,
        model_name=prediction.model_name,
    )
    if model_status != enums.TableStatus.CREATING:
        raise exceptions.ModelFinalizedError(
            dataset_name=prediction.datum.dataset_name,
            model_name=prediction.model_name,
        )

    # retrieve existing table entries
    model = core.fetch_model(db, name=prediction.model_name)
    dataset = core.fetch_dataset(db, name=prediction.datum.dataset_name)
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
        indices = slice(label_idx, label_idx + len(annotation.labels))
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
    except IntegrityError:
        db.rollback()
        raise exceptions.PredictionAlreadyExistsError


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
        model_name=model_name,
        datum=schemas.Datum(
            uid=datum.uid,
            dataset_name=dataset.name,
            metadata=datum.meta,
            geospatial=geo_dict,
        ),
        annotations=core.get_annotations(db, datum=datum, model=model),
    )


def delete_dataset_predictions(
    db: Session,
    dataset: models.Dataset,
):
    """
    Delete all predictions over a dataset.

    Parameters
    ----------
    db : Session
        The database session.
    dataset : models.Dataset
        The dataset row that is being deleted.

    Raises
    ------
    RuntimeError
        If dataset is not in deletion state.
    """

    if dataset.status != enums.TableStatus.DELETING:
        raise RuntimeError(
            f"Attempted to delete predictions from dataset `{dataset.name}` which has status `{dataset.status}`"
        )

    subquery = (
        select(models.Prediction.id.label("id"))
        .join(
            models.Annotation,
            models.Annotation.id == models.Prediction.annotation_id,
        )
        .join(models.Datum, models.Datum.id == models.Annotation.datum_id)
        .where(models.Datum.dataset_id == dataset.id)
        .subquery()
    )
    delete_stmt = delete(models.Prediction).where(
        models.Prediction.id == subquery.c.id
    )

    try:
        db.execute(delete_stmt)
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise e


def delete_model_predictions(
    db: Session,
    model: models.Model,
):
    """
    Delete all predictions of a model.

    Parameters
    ----------
    db : Session
        The database session.
    model : models.Model
        The model row that is being deleted.

    Raises
    ------
    RuntimeError
        If dataset is not in deletion state.
    """

    if model.status != enums.ModelStatus.DELETING:
        raise RuntimeError(
            f"Attempted to delete annotations from dataset `{model.name}` which is not being deleted."
        )

    subquery = (
        select(models.Prediction.id.label("id"))
        .join(
            models.Annotation,
            models.Annotation.id == models.Prediction.annotation_id,
        )
        .where(models.Annotation.model_id == model.id)
        .subquery()
    )
    delete_stmt = delete(models.Prediction).where(
        models.Prediction.id == subquery.c.id
    )

    try:
        db.execute(delete_stmt)
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise e
