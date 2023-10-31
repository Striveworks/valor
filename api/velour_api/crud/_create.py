from sqlalchemy.orm import Session

from velour_api import backend, enums, schemas
from velour_api.backend import stateflow
from velour_api.crud._read import get_disjoint_keys, get_disjoint_labels


@stateflow.create
def create_dataset(*, db: Session, dataset: schemas.Dataset):
    """Creates a dataset

    Raises
    ------
    DatasetAlreadyExistsError
        if the dataset name already exists
    """
    backend.create_dataset(db, dataset)


@stateflow.create
def create_model(*, db: Session, model: schemas.Model):
    """Creates a model

    Raises
    ------
    ModelAlreadyExistsError
        if the model uid already exists
    """
    backend.create_model(db, model)


@stateflow.create
def create_groundtruth(
    *,
    db: Session,
    groundtruth: schemas.GroundTruth,
):
    backend.create_groundtruth(db, groundtruth=groundtruth)


@stateflow.create
def create_prediction(
    *,
    db: Session,
    prediction: schemas.Prediction,
):
    backend.create_prediction(db, prediction=prediction)


@stateflow.evaluate
def create_clf_evaluation(
    *,
    db: Session,
    job_request: schemas.EvaluationJob,
) -> schemas.CreateClfMetricsResponse:
    """create clf evaluation"""

    # get disjoint label sets
    missing_pred_keys, ignored_pred_keys = get_disjoint_keys(
        db=db,
        dataset_name=job_request.dataset,
        model_name=job_request.model,
        task_type=enums.TaskType.CLASSIFICATION,
    )

    # create evaluation setting
    job_id = backend.create_clf_evaluation(db, job_request)

    # create response
    return schemas.CreateClfMetricsResponse(
        missing_pred_keys=missing_pred_keys,
        ignored_pred_keys=ignored_pred_keys,
        job_id=job_id,
    )


@stateflow.computation
def compute_clf_metrics(
    *,
    db: Session,
    job_request: schemas.EvaluationJob,
    job_id: int,
):
    """compute clf metrics"""
    backend.create_clf_metrics(
        db,
        job_request=job_request,
        evaluation_id=job_id,
    )


@stateflow.evaluate
def create_semantic_segmentation_evaluation(
    *,
    db: Session,
    job_request: schemas.EvaluationJob,
) -> schemas.CreateSemanticSegmentationMetricsResponse:
    """create a semantic segmentation evaluation"""

    # get disjoint label sets
    missing_pred_labels, ignored_pred_labels = get_disjoint_labels(
        db=db,
        dataset_name=job_request.dataset,
        model_name=job_request.model,
        task_types=[enums.TaskType.SEGMENTATION],
        groundtruth_type=enums.AnnotationType.RASTER,
        prediction_type=enums.AnnotationType.RASTER,
    )

    # create evaluation setting
    job_id = backend.create_semantic_segmentation_evaluation(db, job_request)

    # create response
    return schemas.CreateSemanticSegmentationMetricsResponse(
        missing_pred_labels=missing_pred_labels,
        ignored_pred_labels=ignored_pred_labels,
        job_id=job_id,
    )


@stateflow.computation
def compute_semantic_segmentation_metrics(
    *,
    db: Session,
    job_request: schemas.EvaluationJob,
    job_id: int,
):
    backend.create_semantic_segmentation_metrics(
        db,
        job_request=job_request,
        evaluation_id=job_id,
    )


@stateflow.evaluate
def create_detection_evaluation(
    *,
    db: Session,
    job_request: schemas.EvaluationJob,
) -> schemas.CreateAPMetricsResponse:
    """create ap evaluation"""

    # create evaluation setting
    (
        job_id,
        groundtruth_type,
        prediction_type,
    ) = backend.create_detection_evaluation(db, job_request)

    # get disjoint label sets
    missing_pred_labels, ignored_pred_labels = get_disjoint_labels(
        db=db,
        dataset_name=job_request.dataset,
        model_name=job_request.model,
        task_types=[enums.TaskType.DETECTION],
        groundtruth_type=groundtruth_type,
        prediction_type=prediction_type,
    )

    # create response
    return schemas.CreateAPMetricsResponse(
        missing_pred_labels=missing_pred_labels,
        ignored_pred_labels=ignored_pred_labels,
        job_id=job_id,
    )


@stateflow.computation
def compute_detection_metrics(
    *,
    db: Session,
    job_request: schemas.EvaluationJob,
    job_id: int,
):
    """compute ap metrics"""
    backend.create_detection_metrics(
        db=db,
        job_request=job_request,
        evaluation_id=job_id,
    )
