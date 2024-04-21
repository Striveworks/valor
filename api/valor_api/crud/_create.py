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
) -> list[schemas.EvaluationResponse]:
    """
    Create or get evaluations.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    job_request: schemas.EvaluationRequest
        The evaluation request.

    Returns
    ----------
    tuple[list[schemas.EvaluatationResponse], list[schemas.EvaluatationResponse]]
        Tuple of evaluation id lists following the form ([created], [existing])
    """
    evaluations = backend.create_or_get_evaluations(db, job_request)
    for evaluation in evaluations:
        if evaluation.status == enums.EvaluationStatus.PENDING:
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
