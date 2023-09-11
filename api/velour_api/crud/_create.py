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
    """Creates a dataset

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
    request_info: schemas.ClfMetricsRequest,
) -> schemas.CreateClfMetricsResponse:
    """create clf evaluation"""

    # get disjoint label sets
    missing_pred_keys, ignored_pred_keys = get_disjoint_keys(
        db=db,
        dataset_name=request_info.settings.dataset,
        model_name=request_info.settings.model,
        task_type=enums.TaskType.CLASSIFICATION,
    )

    # create evaluation setting
    job_id = backend.create_clf_evaluation(db, request_info)

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
    request_info: schemas.ClfMetricsRequest,
    job_id: int,
):
    """compute clf metrics"""
    backend.create_clf_metrics(
        db,
        request_info=request_info,
        evaluation_settings_id=job_id,
    )


@stateflow.evaluate
def create_ap_evaluation(
    *,
    db: Session,
    request_info: schemas.APRequest,
) -> schemas.CreateAPMetricsResponse:
    """create ap evaluation"""

    # get disjoint label sets
    missing_pred_labels, ignored_pred_labels = get_disjoint_labels(
        db=db,
        dataset_name=request_info.settings.dataset,
        model_name=request_info.settings.model,
        task_types=[
            enums.TaskType.DETECTION,
            enums.TaskType.INSTANCE_SEGMENTATION,
        ],
        gt_type=request_info.settings.gt_type,
        pd_type=request_info.settings.pd_type,
    )

    # create evaluation setting
    job_id = backend.create_ap_evaluation(db, request_info)

    # create response
    return schemas.CreateAPMetricsResponse(
        missing_pred_labels=missing_pred_labels,
        ignored_pred_labels=ignored_pred_labels,
        job_id=job_id,
    )


@stateflow.computation
def compute_ap_metrics(
    *,
    db: Session,
    request_info: schemas.APRequest,
    job_id: int,
):
    """compute ap metrics"""
    backend.create_ap_metrics(
        db=db,
        request_info=request_info,
        evaluation_settings_id=job_id,
    )
