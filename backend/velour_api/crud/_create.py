from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from velour_api import exceptions, schemas
from velour_api.backend import core, models, state


@state.create
def create_dataset(db: Session, dataset: schemas.Dataset):
    """Creates a dataset

    Raises
    ------
    DatasetAlreadyExistsError
        if the dataset name already exists
    """
    core.create_dataset(db, dataset)


@state.create
def create_model(db: Session, model: schemas.Model):
    """Creates a dataset

    Raises
    ------
    ModelAlreadyExistsError
        if the model uid already exists
    """
    core.create_model(db, model)


@state.create
def create_groundtruths(
    db: Session,
    groundtruth: schemas.GroundTruth,
):
    # 
    dataset = core.get_dataset(db, groundtruth.dataset_name)
    core.create_groundtruth(
        db, dataset=dataset, groundtruth=groundtruth
    )


@state.create
def create_predictions(
    db: Session,
    prediction: schemas.Prediction,
):
    model = core.get_model(db, prediction.model_name)
    core.create_prediction(db, prediction=prediction, model=model)
