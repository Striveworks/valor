from sqlalchemy.orm import Session

from velour_api import backend, enums, schemas, exceptions
from velour_api.crud._read import get_disjoint_keys, get_disjoint_labels


def create_dataset(*, db: Session, dataset: schemas.Dataset):
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


def create_model(*, db: Session, model: schemas.Model):
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


def create_clf_evaluation(
    *,
    db: Session,
    job_request: schemas.EvaluationJob,
) -> schemas.CreateClfMetricsResponse:
    """
    Creates a classification evaluation.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    job_request: schemas.EvaluationJob
        The evaluation job.


    Returns
    ----------
    schemas.CreateClfMetricsResponse
        A classification metric response.
    """

    # get disjoint label sets
    missing_pred_keys, ignored_pred_keys = get_disjoint_keys(
        db=db,
        dataset_name=job_request.dataset,
        model_name=job_request.model,
        task_type=enums.TaskType.CLASSIFICATION,
    )

    # create or get evaluation
    evaluation_id = backend.create_or_get_evaluation(db, job_request)

    # create response
    return schemas.CreateClfMetricsResponse(
        missing_pred_keys=missing_pred_keys,
        ignored_pred_keys=ignored_pred_keys,
        evaluation_id=evaluation_id,
    )


def compute_clf_metrics(
    *,
    db: Session,
    evaluation_id: int,
    job_request: schemas.EvaluationJob,
):
    """
    Compute classification metrics.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    evaluation_id: int
        The job id.
    job_request: schemas.EvaluationJob
        The evaluation job.
    """
    backend.compute_clf_metrics(
        db=db,
        evaluation_id=evaluation_id,
    )


def create_semantic_segmentation_evaluation(
    *,
    db: Session,
    job_request: schemas.EvaluationJob,
) -> schemas.CreateSemanticSegmentationMetricsResponse:
    """
    Creates a semantic segmentation evaluation.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    job_request: schemas.EvaluationJob
        The evaluation job.


    Returns
    ----------
    schemas.CreateSemanticSegmentationMetricsResponse
        A semantic segmentation metric response.
    """

    # get disjoint label sets
    missing_pred_labels, ignored_pred_labels = get_disjoint_labels(
        db=db,
        dataset_name=job_request.dataset,
        model_name=job_request.model,
        task_types=[enums.TaskType.SEGMENTATION],
        groundtruth_type=enums.AnnotationType.RASTER,
        prediction_type=enums.AnnotationType.RASTER,
    )

    # create or get evaluation
    evaluation_id = backend.create_or_get_evaluation(db, job_request)

    # create response
    return schemas.CreateSemanticSegmentationMetricsResponse(
        missing_pred_labels=missing_pred_labels,
        ignored_pred_labels=ignored_pred_labels,
        evaluation_id=evaluation_id,
    )


def compute_semantic_segmentation_metrics(
    *,
    db: Session,
    job_request: schemas.EvaluationJob,
    evaluation_id: int,
):
    """
    Compute semantic segmentation metrics.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    job_request: schemas.EvaluationJob
        The evaluation job object.
    evaluation_id: int
        The job id.
    """
    backend.compute_semantic_segmentation_metrics(
        db,
        evaluation_id=evaluation_id,
        job_request=job_request,
    )


def create_detection_evaluation(
    *,
    db: Session,
    job_request: schemas.EvaluationJob,
) -> schemas.CreateDetectionMetricsResponse:
    """
    Creates a detection evaluation.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    job_request: schemas.EvaluationJob
        The evaluation job.

    Returns
    ----------
    schemas.CreateDetectionMetricsResponse
        A detection metric response.
    """

    # get disjoint label sets
    missing_pred_labels, ignored_pred_labels = backend.get_disjoint_labels_from_evaluation(db, job_request)

    # create or get evaluation
    evaluation_id = backend.create_or_get_evaluation(db, job_request)

    # create response
    return schemas.CreateDetectionMetricsResponse(
        missing_pred_labels=missing_pred_labels,
        ignored_pred_labels=ignored_pred_labels,
        evaluation_id=evaluation_id,
    )


def compute_detection_metrics(
    *,
    db: Session,
    job_request: schemas.EvaluationJob,
    evaluation_id: int,
):
    """
    Compute detection metrics.

    Parameters
    ----------
    db : Session
        The database Session to query against.
    job_request: schemas.EvaluationJob
        The evaluation job object.
    evaluation_id: int
        The job id.
    """
    backend.compute_detection_metrics(
        db=db,
        evaluation_id=evaluation_id,
    )
