from fastapi import BackgroundTasks
from sqlalchemy.orm import Session

from velour_api import backend, enums, exceptions, schemas
from velour_api.crud._read import get_disjoint_keys, get_disjoint_labels


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


def create_evaluations(
    *,
    db: Session,
    job_request: schemas.EvaluationRequest,
    task_handler: BackgroundTasks,
) -> list[
    schemas.CreateClfEvaluationResponse
    | schemas.CreateDetectionEvaluationResponse
    | schemas.CreateSemanticSegmentationEvaluationResponse
]:
    """
    Creates evaluations.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    job_request: schemas.EvaluationRequest
        The evaluation request.


    Returns
    ----------
    tuple[list[int], list[int]]
        Tuple of evaluation id lists following the form ([created], [existing])
    """
    created, existing = backend.create_evaluations(db, job_request)

    # start computations
    for resp in created:
        match type(resp):
            case schemas.CreateClfEvaluationResponse:
                task_handler.add_task(
                    backend.compute_clf_metrics,
                    db=db,
                    evaluation_id=resp.evaluation_id,
                    job_request=job_request,
                )
            case schemas.CreateDetectionEvaluationResponse:
                task_handler.add_task(
                    backend.compute_detection_metrics,
                    db=db,
                    evaluation_id=resp.evaluation_id,
                    job_request=job_request,
                )
            case schemas.CreateSemanticSegmentationEvaluationResponse:
                task_handler.add_task(
                    backend.compute_semantic_segmentation_metrics,
                    db=db,
                    evaluation_id=resp.evaluation_id,
                    job_request=job_request,
                )
            case _:
                raise RuntimeError

    return created + existing
