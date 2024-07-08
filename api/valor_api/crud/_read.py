from sqlalchemy.orm import Session

from valor_api import backend, enums, schemas


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


def get_labels(
    *,
    db: Session,
    filters: schemas.Filter | None = None,
    ignore_prediction_labels=False,
    ignore_groundtruth_labels=False,
    offset: int = 0,
    limit: int = -1,
) -> tuple[set[schemas.Label], dict[str, str]]:
    """
    Fetch a list of labels from the database.

    The default behavior is return a list of all existing labels.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    filters : schemas.Filter, optional
        An optional filter to apply.
    ignore_prediction_labels : bool, default=False
        Option to ignore prediction labels in the result.
    ignore_groundtruths : bool, default=False
        Option to ignore ground truth labels in the result.
    offset : int, optional
        The start index of the items to return.
    limit : int, optional
        The number of items to return. Returns all models when set to -1.

    Returns
    ----------
    tuple[set[schemas.Label], dict[str, str]]
        A tuple containing the labels and response headers to return to the user.
    """
    return backend.get_paginated_labels(
        db=db,
        filters=filters,
        ignore_predictions=ignore_prediction_labels,
        ignore_groundtruths=ignore_groundtruth_labels,
        offset=offset,
        limit=limit,
    )


""" Datum """


def get_datums(
    *,
    db: Session,
    filters: schemas.Filter | None = None,
    offset: int = 0,
    limit: int = -1,
) -> tuple[list[schemas.Datum], dict[str, str]]:
    """
    Get datums with optional filter.

    Default behavior is to return all existing datums.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    filters : schemas.Filter, optional
        An optional filter to apply.
    offset : int, optional
        The start index of the items to return.
    limit : int, optional
        The number of items to return. Returns all items when set to -1.


    Returns
    ----------
    tuple[list[schemas.Datum], dict[str, str]]
        A tuple containing the datums and response headers to return to the user.
    """
    return backend.get_paginated_datums(
        db=db, filters=filters, offset=offset, limit=limit
    )


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
    filters: schemas.Filter,
    offset: int = 0,
    limit: int = -1,
) -> tuple[list[schemas.Dataset], dict[str, str]]:
    """
    Get datasets with optional filter.

    Default behavior is to return all existing datasets.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    filters : schemas.Filter
        A filter object to constrain the results by.
    offset : int
        The start index of the items to return.
    limit : int
        The number of items to return. Returns all items when set to -1.


    Returns
    ----------
    tuple[list[schemas.Dataset], dict[str, str]]
        A tuple containing the datasets and response headers to return to the user.
    """
    return backend.get_paginated_datasets(
        db=db, filters=filters, offset=offset, limit=limit
    )


def get_dataset_summary(*, db: Session, name: str) -> schemas.DatasetSummary:
    return backend.get_dataset_summary(db, name)


def get_groundtruth(
    *,
    db: Session,
    dataset_name: str,
    datum_uid: str,
) -> schemas.GroundTruth:
    """
    Fetch a ground truth.

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
        The requested ground truth.
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
    filters: schemas.Filter | None = None,
    offset: int = 0,
    limit: int = -1,
) -> tuple[list[schemas.Model], dict[str, str]]:
    """
    Get models with optional filter.

    Default behavior is to return all existing models.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    filters : schemas.FilterQueryParams, optional
        An optional filter to constrain results by.
    offset : int, optional
        The start index of the items to return.
    limit : int, optional
        The number of items to return. Returns all items when set to -1.
    db : Session
        The database session to use. This parameter is a sqlalchemy dependency and shouldn't be submitted by the user.

    Returns
    ----------
    tuple[list[schemas.Model], dict[str, str]]
        A tuple containing the models and response headers to return to the user.
    """
    return backend.get_paginated_models(
        db=db, filters=filters, offset=offset, limit=limit
    )


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
    offset: int = 0,
    limit: int = -1,
    metrics_to_sort_by: dict[str, dict[str, str] | str] | None = None,
) -> tuple[list[schemas.EvaluationResponse], dict[str, str]]:
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
    offset : int, optional
        The start index of the items to return.
    limit : int, optional
        The number of items to return. Returns all items when set to -1.
    metrics_to_sort_by: dict[str, dict[str, str] | str], optional
        An optional dict of metric types to sort the evaluations by.

    Returns
    ----------
    tuple[list[schemas.EvaluationResponse], dict[str, str]]
        A tuple containing the evaluations and response headers to return to the user.
    """
    # get evaluations that conform to input args
    print("ENTERING CRUD")
    return backend.get_paginated_evaluations(
        db=db,
        evaluation_ids=evaluation_ids,
        dataset_names=dataset_names,
        model_names=model_names,
        offset=offset,
        limit=limit,
        metrics_to_sort_by=metrics_to_sort_by,
    )


def get_evaluation_requests_from_model(
    db: Session, model_name: str
) -> list[schemas.EvaluationResponse]:
    """
    Returns all evaluation settings for a given model.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    model_name : str
        The model name to find evaluations of

    Returns
    ----------
    list[schemas.EvaluationResponse]
        A list of evaluations.
    """
    return backend.get_evaluation_requests_from_model(db, model_name)
