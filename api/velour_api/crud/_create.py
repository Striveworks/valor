from fastapi import BackgroundTasks
from sqlalchemy.orm import Session

from velour_api import backend, enums, schemas


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
    Creates a groundtruth.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    groundtruth: schemas.GroundTruth
        The groundtruth to create.
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
    created, existing = backend.create_or_get_evaluations(db, job_request)

    # start computations
    for evaluation in created:
        match evaluation.evaluation_filter.task_types[0]:
            case enums.TaskType.CLASSIFICATION:
                compute_func = backend.compute_clf_metrics
            case enums.TaskType.DETECTION:
                compute_func = backend.compute_detection_metrics
            case enums.TaskType.SEGMENTATION:
                compute_func = backend.compute_semantic_segmentation_metrics
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

    return created + existing
