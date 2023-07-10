from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from velour_api import exceptions, schemas
from velour_api.backend import core, models, state, io


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
    data: dict,
):
    # unpack
    name = data["name"],
    groundtruths = schemas.GroundTruth(**data["data"])

    # 
    dataset = io.request_dataset(db, name)
    core.create_groundtruths(
        db, dataset=dataset, groundtruths=groundtruths
    )


@state.create
def create_predictions(
    db: Session,
    name: str,
    predictions: list[schemas.Prediction],
):
    model = io.request_model(db, name)
    core.create_predictions(db, annotated_datums=predictions, model=model)
