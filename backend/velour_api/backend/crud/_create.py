from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from velour_api import exceptions
from velour_api.backend import models, state
from velour_api import schemas




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
def create_ground_truth_for_dataset(db: Session, schemas.GroundTruth)