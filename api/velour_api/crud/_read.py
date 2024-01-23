from sqlalchemy.orm import Session

from velour_api import backend, enums, schemas


def get_table_status(
    *,
    db: Session,
    dataset_name: str,
    model_name: str | None = None,
) -> enums.TableStatus:
    """
    Fetch dataset or dataset + model status.

    Parameters
    ----------
    db : Session
        The database session.
    dataset_name : str
        Name of the dataset.
    model_name : str, optional
        Name of the model.

    Returns
    ----------
    enums.TableStatus
        The requested status.
    """
    if dataset_name and model_name:
        return backend.get_model_status(
            db=db, dataset_name=dataset_name, model_name=model_name
        )
    else:
        return backend.get_dataset_status(db=db, name=dataset_name)


def get_evaluation_status(
    *,
    db: Session,
    evaluation_id: int,
) -> enums.EvaluationStatus:
    """
    Fetch evaluation status.

    Parameters
    ----------
    db : Session
        The database session.
    evaluation_id : int
        Unique identifer of an evaluation.

    Returns
    ----------
    enums.EvaluationStatus
        The requested evaluation status.
    """
    return backend.get_evaluation_status(db=db, evaluation_id=evaluation_id)


""" Labels """


def get_all_labels(
    *,
    db: Session,
    filters: schemas.Filter | None = None,
) -> list[schemas.Label]:
    """
    Fetch all labels from the database.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    filters : schemas.Filter
        An optional filter to apply.

    Returns
    ----------
    list[schemas.Label]
        A list of labels.
    """
    return list(backend.get_labels(db, filters))


def get_dataset_labels(
    *,
    db: Session,
    filters: schemas.Filter | None = None,
) -> list[schemas.Label]:
    """
    Fetch all labels associated with dataset groundtruths from the database.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    filters : schemas.Filter
        An optional filter to apply.

    Returns
    ----------
    list[schemas.Label]
        A list of labels.
    """
    return list(
        backend.get_labels(
            db=db,
            filters=filters,
            ignore_predictions=True,
        )
    )


def get_model_labels(
    *,
    db: Session,
    filters: schemas.Filter | None = None,
) -> list[schemas.Label]:
    """
    Fetch all labels associated with dataset predictions from the database.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    filters : schemas.Filter
        An optional filter to apply.

    Returns
    ----------
    list[schemas.Label]
        A list of labels.
    """
    return list(
        backend.get_labels(
            db=db,
            filters=filters,
            ignore_groundtruths=True,
        )
    )


""" Datum """


def get_datums(
    *,
    db: Session,
    request: schemas.Filter = None,
) -> list[schemas.Datum]:
    """
    Return all datums in the database.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    request : schemas.Filter
        An optional filter to apply.

    Returns
    ----------
    List[schemas.Datum]
        A list of datums.
    """
    return backend.get_datums(db, request)


""" Datasets """


def get_dataset(
    *,
    db: Session,
    dataset_name: str,
) -> schemas.Dataset:
    """
    Fetch a dataset.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    dataset_name : str
        The name of the dataset.

    Returns
    ----------
    schemas.Dataset
        The requested dataset.
    """
    return backend.get_dataset(db, dataset_name)


def get_datasets(
    *,
    db: Session,
) -> list[schemas.Dataset]:
    """
    Fetch all datasets.

    Parameters
    ----------
    db : Session
        The database Session to query against.

    Returns
    ----------
    List[schemas.Dataset]
        A list of all datasets.
    """
    return backend.get_all_datasets(db)


def get_dataset_summary(*, db: Session, name: str) -> schemas.DatasetSummary:
    return backend.get_dataset_summary(db, name)


def get_groundtruth(
    *,
    db: Session,
    dataset_name: str,
    datum_uid: str,
) -> schemas.GroundTruth:
    """
    Fetch a groundtruth.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    dataset_name : str
        The name of the dataset.
    datum_uid: str
        The UID of the datum to fetch.


    Returns
    ----------
    schemas.GroundTruth
        The requested groundtruth.
    """
    return backend.get_groundtruth(
        db,
        dataset_name=dataset_name,
        datum_uid=datum_uid,
    )


""" Models """


def get_model(*, db: Session, model_name: str) -> schemas.Model:
    """
    Fetch a model.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    model_name : str
        The name of the model.

    Returns
    ----------
    schemas.Model
        The requested model.
    """
    return backend.get_model(db, model_name)


def get_models(
    *,
    db: Session,
) -> list[schemas.Model]:
    """
    Fetch all models.

    Parameters
    ----------
    db : Session
        The database Session to query against.

    Returns
    ----------
    List[schemas.Model]
        A list of all models.
    """
    return backend.get_models(db)


def get_prediction(
    *,
    db: Session,
    dataset_name: str,
    model_name: str,
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
    return backend.get_prediction(
        db,
        model_name=model_name,
        dataset_name=dataset_name,
        datum_uid=datum_uid,
    )


""" Evaluations """


def get_evaluations(
    *,
    db: Session,
    evaluation_ids: list[int] | None = None,
    dataset_names: list[str] | None = None,
    model_names: list[str] | None = None,
) -> list[schemas.EvaluationResponse]:
    """
    Returns all evaluations that conform to user-supplied constraints.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    evaluation_ids
        A list of evaluation job id constraints.
    dataset_names
        A list of dataset names to constrain by.
    model_names
        A list of model names to constrain by.

    Returns
    ----------
    list[schemas.EvaluationResponse]
        A list of evaluations.
    """
    # get evaluations that conform to input args
    return backend.get_evaluations(
        db=db,
        evaluation_ids=evaluation_ids,
        dataset_names=dataset_names,
        model_names=model_names,
    )
