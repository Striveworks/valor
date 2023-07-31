from sqlalchemy.orm import Session

from velour_api import backend, schemas
from velour_api.backend import state


@state.create
def create_dataset(db: Session, dataset: schemas.Dataset):
    """Creates a dataset

    Raises
    ------
    DatasetAlreadyExistsError
        if the dataset name already exists
    """
    backend.create_dataset(db, dataset)


@state.create
def create_model(db: Session, model: schemas.Model):
    """Creates a dataset

    Raises
    ------
    ModelAlreadyExistsError
        if the model uid already exists
    """
    backend.create_model(db, model)


@state.create
def create_groundtruth(
    db: Session,
    groundtruth: schemas.GroundTruth,
):
    backend.create_groundtruth(db, groundtruth=groundtruth)


@state.create
def create_prediction(
    db: Session,
    prediction: schemas.Prediction,
):
    backend.create_prediction(db, prediction=prediction)


@state.create
def create_clf_metrics(
    db: Session,
    request_info: schemas.ClfMetricsRequest,
) -> int:
    return backend.create_clf_metrics(db, request_info)


# @TODO: Make this return a job id
@state.create
def create_ap_metrics(
    db: Session,
    request_info: schemas.APRequest,
) -> int:
    return backend.create_ap_metrics(db, request_info)
