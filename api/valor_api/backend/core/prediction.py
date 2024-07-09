from collections import defaultdict

from sqlalchemy import and_, delete, func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from valor_api import enums, exceptions, schemas
from valor_api.backend import core, models


def create_predictions(
    db: Session,
    predictions: list[schemas.Prediction],
):
    """
    Creates a prediction.

    Parameters
    ----------
    db
        The database Session to query against.
    predictions
        The predictions to create.
    """

    # check model status
    dataset_and_model_names = set(
        [
            (prediction.dataset_name, prediction.model_name)
            for prediction in predictions
        ]
    )
    for dataset_name, model_name in dataset_and_model_names:
        model_status = core.get_model_status(
            db=db,
            dataset_name=dataset_name,
            model_name=model_name,
        )
        if model_status != enums.TableStatus.CREATING:
            raise exceptions.ModelFinalizedError(
                dataset_name=dataset_name,
                model_name=model_name,
            )

    # check no predictions have already been added
    map_dataset_model_to_datum = defaultdict(list)
    for prediction in predictions:
        map_dataset_model_to_datum[
            (prediction.dataset_name, prediction.model_name)
        ].append(prediction.datum.uid)

    if (
        db.scalar(
            select(func.count(models.Annotation.id))
            .join(
                models.Datum,
                models.Datum.id == models.Annotation.datum_id,
            )
            .join(
                models.Dataset,
                models.Dataset.id == models.Datum.dataset_id,
            )
            .join(
                models.Model,
                models.Model.id == models.Annotation.model_id,
            )
            .where(
                or_(
                    *[
                        and_(
                            models.Dataset.name == dataset_name,
                            models.Model.name == model_name,
                            models.Datum.uid.in_(datum_uids),
                        )
                        for (
                            dataset_name,
                            model_name,
                        ), datum_uids in map_dataset_model_to_datum.items()
                    ]
                )
            )
        )
        != 0
    ):
        raise exceptions.PredictionAlreadyExistsError

    model_names = set([dm[1] for dm in dataset_and_model_names])
    model_name_to_model = {
        model_name: core.fetch_model(db=db, name=model_name)
        for model_name in model_names
    }

    # retrieve datum ids
    dataset_to_datum_uids = defaultdict(list)
    for prediction in predictions:
        dataset_to_datum_uids[prediction.dataset_name].append(
            prediction.datum.uid
        )
    dataset_datum_to_datum_id = {
        (dataset_name, datum_uid): datum_id
        for datum_id, dataset_name, datum_uid in (
            db.query(
                models.Datum.id,
                models.Dataset.name,
                models.Datum.uid,
            )
            .join(
                models.Dataset,
                models.Dataset.id == models.Datum.dataset_id,
            )
            .where(
                or_(
                    *[
                        and_(
                            models.Dataset.name == dataset_name,
                            models.Datum.uid.in_(datum_uids),
                        )
                        for dataset_name, datum_uids in dataset_to_datum_uids.items()
                    ]
                )
            )
            .all()
        )
    }
    try:
        datum_ids = [
            dataset_datum_to_datum_id[
                (prediction.dataset_name, prediction.datum.uid)
            ]
            for prediction in predictions
        ]
    except KeyError as e:
        raise exceptions.DatumDoesNotExistError(e.args[0])

    # create labels
    all_labels = [
        label
        for prediction in predictions
        for annotation in prediction.annotations
        for label in annotation.labels
    ]
    label_dict = core.create_labels(db=db, labels=all_labels)

    # create annotations
    annotation_ids = core.create_annotations(
        db=db,
        annotations=[prediction.annotations for prediction in predictions],
        datum_ids=datum_ids,
        models_=[
            model_name_to_model[prediction.model_name]
            for prediction in predictions
        ],
    )

    # create predictions
    prediction_mappings = []
    for prediction, annotation_ids_per_prediction in zip(
        predictions, annotation_ids
    ):
        for i, annotation in enumerate(prediction.annotations):
            for label in annotation.labels:
                prediction_mappings.append(
                    models.Prediction(
                        annotation_id=annotation_ids_per_prediction[i],
                        label_id=label_dict[(label.key, label.value)],
                        score=label.score,
                    )
                )
    try:
        db.add_all(prediction_mappings)
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
    annotations = core.get_annotations(db, datum=datum, model=model)
    if len(annotations) == 0:
        raise exceptions.PredictionDoesNotExistError(
            model_name=model_name,
            dataset_name=dataset_name,
            datum_uid=datum_uid,
        )
    return schemas.Prediction(
        dataset_name=dataset.name,
        model_name=model_name,
        datum=schemas.Datum(
            uid=datum.uid,
            metadata=datum.meta,
        ),
        annotations=annotations,
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
