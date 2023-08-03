from sqlalchemy.orm import Session

from velour_api import backend, schemas
from velour_api.backend import jobs
from velour_api.crud._read import get_disjoint_keys, get_disjoint_labels


@jobs.create
def create_dataset(db: Session, dataset: schemas.Dataset):
    """Creates a dataset

    Raises
    ------
    DatasetAlreadyExistsError
        if the dataset name already exists
    """
    backend.create_dataset(db, dataset)


@jobs.create
def create_model(db: Session, model: schemas.Model):
    """Creates a dataset

    Raises
    ------
    ModelAlreadyExistsError
        if the model uid already exists
    """
    backend.create_model(db, model)


@jobs.create
def create_groundtruth(
    db: Session,
    groundtruth: schemas.GroundTruth,
):
    backend.create_groundtruth(db, groundtruth=groundtruth)


@jobs.create
def create_prediction(
    db: Session,
    prediction: schemas.Prediction,
):
    backend.create_prediction(db, prediction=prediction)


@jobs.evaluate(persist=True)
def create_clf_evaluation(
    db: Session,
    request_info: schemas.ClfMetricsRequest,
) -> schemas.CreateClfMetricsResponse:
    """create evaluation"""

    # get disjoint label sets
    missing_pred_keys, ignored_pred_keys = get_disjoint_keys(
        db,
        dataset_name=request_info.settings.dataset,
        model_name=request_info.settings.model,
    )

    # create evaluation setting
    evaluation_settings_id = backend.create_clf_evaluation(db, request_info)

    # create response
    return schemas.CreateClfMetricsResponse(
        missing_pred_keys=missing_pred_keys,
        ignored_pred_keys=ignored_pred_keys,
        job_id=evaluation_settings_id,
    )


@jobs.evaluate()
def create_clf_metrics(
    db: Session,
    request_info: schemas.ClfMetricsRequest,
    evaluation_settings_id: int,
):
    """compute metrics"""
    backend.create_clf_metrics(
        db,
        request_info=request_info,
        evaluation_settings_id=evaluation_settings_id,
    )


@jobs.evaluate(persist=True)
def create_ap_evaluation(
    db: Session,
    request_info: schemas.APRequest,
) -> schemas.CreateAPMetricsResponse:
    """create evaluation"""

    # get disjoint label sets
    missing_pred_labels, ignored_pred_labels = get_disjoint_labels(
        db,
        dataset_name=request_info.settings.dataset,
        model_name=request_info.settings.model,
    )

    # create evaluation setting
    evaluation_settings_id = backend.create_clf_evaluation(db, request_info)

    # create response
    return schemas.CreateAPMetricsResponse(
        missing_pred_labels=missing_pred_labels,
        ignored_pred_labels=ignored_pred_labels,
        evaluation_settings_id=evaluation_settings_id,
    )


@jobs.evaluate()
def create_ap_metrics(
    db: Session,
    request_info: schemas.APRequest,
    evaluation_settings_id: int,
):
    """compute metrics"""
    backend.create_ap_metrics(
        db=Session,
        request_info=request_info,
        evaluation_settings_id=evaluation_settings_id,
    )
