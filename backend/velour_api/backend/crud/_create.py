from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from velour_api import exceptions, schemas
from velour_api.backend import models, state


@state.create
def create_dataset(db: Session, dataset: schemas.Dataset):
    """Creates a dataset

    Raises
    ------
    DatasetAlreadyExistsError
        if the dataset name already exists
    """
    try:
        db.add(models.Dataset(**dataset.dict()))
        db.commit()
    except IntegrityError:
        db.rollback()
        raise exceptions.DatasetAlreadyExistsError(dataset.name)


@state.create
def create_model(db: Session, model: schemas.Model):
    """Creates a dataset

    Raises
    ------
    ModelAlreadyExistsError
        if the model uid already exists
    """
    try:
        db.add(models.Model(**model.dict()))
        db.commit()
    except IntegrityError:
        db.rollback()
        raise exceptions.ModelAlreadyExistsError(model.name)


@state.create
def create_groundtruth(db: Session, gt: schemas.GroundTruth):
    pass


@state.create
def create_prediction(db: Session, pd: schemas.Prediction):
    pass
