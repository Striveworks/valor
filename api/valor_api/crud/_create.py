from fastapi import BackgroundTasks
from sqlalchemy.orm import Session

from valor_api import backend, enums, schemas


def create_dataset(
    *,
    db: Session,
    dataset: schemas.Dataset,
):
    """
    Creates a dataset.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    dataset : schemas.Dataset
        The dataset to create.

    Raises
    ----------
    DatasetAlreadyExistsError
        If the dataset name already exists.
    """
    backend.create_dataset(db, dataset)


def create_model(
    *,
    db: Session,
    model: schemas.Model,
):
    """
    Creates a model.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    model : schemas.Model
        The model to create.

    Raises
    ----------
    ModelAlreadyExistsError
        If the model name already exists.
    """
    backend.create_model(db, model)


def create_groundtruth(
    *,
    db: Session,
    groundtruth: schemas.GroundTruth,
):
    """
    Creates a ground truth.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    groundtruth: schemas.GroundTruth
        The ground truth to create.
    """
    backend.create_groundtruth(db, groundtruth=groundtruth)


def create_prediction(
    *,
    db: Session,
    prediction: schemas.Prediction,
):
    """
    Creates a prediction.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    prediction: schemas.Prediction
        The prediction to create.
    """
    backend.create_prediction(db, prediction=prediction)


def create_or_get_evaluations(
    *,
    db: Session,
    job_request: schemas.EvaluationRequest,
    task_handler: BackgroundTasks | None = None,
    allow_retries: bool = False,
) -> list[schemas.EvaluationResponse]:
    """
    Create or get evaluations.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    job_request: schemas.EvaluationRequest
        The evaluation request.
    task_handler: BackgroundTasks, optional
        An optional FastAPI background task handler.
    allow_retries: bool, default = False
        Allow restarting of failed evaluations.

    Returns
    ----------
    list[schemas.EvaluatationResponse]
        A list of evaluations in response format.
    """
    evaluations = backend.create_or_get_evaluations(
        db=db, job_request=job_request, allow_retries=allow_retries
    )
    run_conditions = (
        {
            enums.EvaluationStatus.PENDING,
            enums.EvaluationStatus.FAILED,
        }
        if allow_retries
        else {enums.EvaluationStatus.PENDING}
    )
    for evaluation in evaluations:
        if evaluation.status in run_conditions:
            match evaluation.parameters.task_type:
                case enums.TaskType.CLASSIFICATION:
                    compute_func = backend.compute_clf_metrics
                case enums.TaskType.OBJECT_DETECTION:
                    compute_func = backend.compute_detection_metrics
                case enums.TaskType.SEMANTIC_SEGMENTATION:
                    compute_func = (
                        backend.compute_semantic_segmentation_metrics
                    )
                case _:
                    raise RuntimeError

            if task_handler:
                task_handler.add_task(
                    compute_func,
                    db=db,
                    evaluation_id=evaluation.id,
                )
            else:
                compute_func(db=db, evaluation_id=evaluation.id)

    return evaluations
