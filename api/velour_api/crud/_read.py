from sqlalchemy.orm import Session

from velour_api import backend, enums, schemas
from velour_api.crud.jobs import get_status_from_names


def get_job_status(
    *,
    dataset_name: str = None,
    model_name: str = None,
    evaluation_id: int = None,
) -> enums.JobStatus:
    """
    Fetch job status.

    The input must conform to one of the following sets to properly fetch a job:

    - Dataset Job: {dataset_name}
    - Model Job: {model_name}
    - Model Prediction Job: {dataset_name, model_name}
    - Evaluation Job: {evaluation_id} or {dataset_name, model_name, evaluation_id}

    Parameters
    ----------
    dataset_name : str, optional
        Name of a dataset.
    model_name : str, optional
        Name of a model.
    evaluation_id : int, optional
        Unique identifer of an Evaluation.

    Returns
    ----------
    enums.JobStatus
        The requested evaluation status.
    """
    return get_status_from_names(dataset_name, model_name, evaluation_id)


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


def get_joint_labels(
    *,
    db: Session,
    dataset_name: str,
    model_name: str,
    task_types: list[enums.TaskType],
    groundtruth_type: enums.AnnotationType,
    prediction_type: enums.AnnotationType,
) -> list[schemas.Label]:
    """
    Returns all unique labels that are shared between both predictions and groundtruths.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    dataset_name: str
        The name of a dataset.
    model_name: str
        The name of a model.
    task_types: list[enums.TaskType]
        The task types to filter on.
    groundtruth_type: enums.AnnotationType
        The groundtruth type to filter on.
    prediction_type: enums.AnnotationType
        The prediction type to filter on

    Returns
    ----------
    list[schemas.Label]
        A list of labels.
    """
    return backend.get_joint_labels(
        db,
        dataset_name=dataset_name,
        model_name=model_name,
        task_types=task_types,
        groundtruth_type=groundtruth_type,
        prediction_type=prediction_type,
    )


def get_disjoint_labels(
    *,
    db: Session,
    dataset_name: str,
    model_name: str,
    task_types: list[enums.TaskType],
    groundtruth_type: enums.AnnotationType,
    prediction_type: enums.AnnotationType,
) -> tuple[list[schemas.Label], list[schemas.Label]]:
    """
    Returns all unique labels that are not shared between both predictions and groundtruths.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    dataset_name: str
        The name of a dataset.
    model_name: str
        The name of a model.
    task_types: list[enums.TaskType]
        The task types to filter on.
    groundtruth_type: enums.AnnotationType
        The groundtruth type to filter on.
    prediction_type: enums.AnnotationType
        The prediction type to filter on

    Returns
    ----------
    Tuple[list[schemas.Label], list[schemas.Label]]
        A tuple of disjoint labels, where the first element is those labels which are present in groundtruths but absent in predictions.
    """
    return backend.get_disjoint_labels(
        db,
        dataset_name=dataset_name,
        model_name=model_name,
        task_types=task_types,
        groundtruth_type=groundtruth_type,
        prediction_type=prediction_type,
    )


def get_disjoint_keys(
    *,
    db: Session,
    dataset_name: str,
    model_name: str,
    task_type: enums.TaskType,
) -> tuple[list[str], list[str]]:
    """
    Returns all unique label keys that are not shared between both predictions and groundtruths.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    dataset_name: str
        The name of a dataset.
    model_name: str
        The name of a model.
    task_types: list[enums.TaskType]
        The task types to filter on.

    Returns
    ----------
    Tuple[list[schemas.Label], list[schemas.Label]]
        A tuple of disjoint label key, where the first element is those labels which are present in groundtruths but absent in predictions.
    """
    return backend.get_disjoint_keys(
        db,
        dataset_name=dataset_name,
        model_name=model_name,
        task_type=task_type,
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


def get_dataset(*, db: Session, dataset_name: str) -> schemas.Dataset:
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
    return backend.get_datasets(db)


def get_dataset_summary(db: Session, name: str) -> schemas.DatasetSummary:
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
    *, db: Session, model_name: str, dataset_name: str, datum_uid: str
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


""" Evaluation """


def get_model_metrics(
    *, db: Session, model_name: str, evaluation_id: int
) -> list[schemas.Metric]:
    """
    Fetch all metrics for a given model.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    model_name : str
        The name of the model.
    evaluation_id : id
        The evaluation ID.

    Returns
    ----------
    list[schemas.Metric]
        A list of metrics.
    """
    return backend.get_model_metrics(db, model_name, evaluation_id)


def get_evaluation_jobs(
    *,
    db: Session,
    job_ids: list[int] | None = None,
    dataset_names: list[str] | None = None,
    model_names: list[str] | None = None,
    settings: list[schemas.EvaluationSettings] | None = None,
) -> list[schemas.EvaluationJob]:
    """
    Returns all evaluation jobs that conform to user-supplied args.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    job_ids : list[int]
        A list of evaluation job id constraints.
    dataset_names | list[str]
        A list of dataset names to constrain by.
    model_names | list[str]
        A list of model names to constrain by.
    settings : list[schemas.EvaluationSettings]
        A list of `schemas.EvaluationSettings` to constrain by.

    Returns
    ----------
    list[schemas.EvaluationJob]
        A list of evaluation jobs.
    """
    return backend.get_evaluation_jobs(
        db=db,
        job_ids=job_ids,
        dataset_names=dataset_names,
        model_names=model_names,
        settings=settings,
    )


def get_evaluations(
    *,
    db: Session,
    job_ids: list[int] | None = None,
    dataset_names: list[str] | None = None,
    model_names: list[str] | None = None,
    settings: list[schemas.EvaluationSettings] | None = None,
) -> list[schemas.Evaluation]:
    """
    Returns all evaluations that conform to user-supplied constraints.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    job_ids
        A list of evaluation job id constraints.
    dataset_names
        A list of dataset names to constrain by.
    model_names
        A list of model names to constrain by.
    settings:
        A list of `schemas.EvaluationSettings` to constrain by.

    Returns
    ----------
    list[schemas.Evaluations]
        A list of evaluations.
    """
    # get evaluations that conform to input args
    evaluations = backend.get_evaluations(
        db=db,
        job_ids=job_ids,
        dataset_names=dataset_names,
        model_names=model_names,
        settings=settings,
    )

    # set evaluation status (redis only available in crud)
    for evaluation in evaluations:
        evaluation.status = get_status_from_names(
            dataset_name=evaluation.dataset,
            model_name=evaluation.model,
            evaluation_id=evaluation.job_id,
        )

    return evaluations
