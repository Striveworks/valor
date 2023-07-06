from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from velour_api import exceptions
from velour_api.backend import models, state, core
from velour_api.schemas import Dataset, DatasetInfo, Model, ModelInfo


@state.read
def get_datasets(db: Session) -> list[DatasetInfo]:
    return core.get_datasets()


@state.read
def get_dataset(db: Session, name: str) -> Dataset:
    return core.get_dataset(db, name)


@state.read
def get_models(db: Session) -> list[ModelInfo]:
    return core.get_models(db)


@state.read
def get_model(db: Session, name: str) -> Model:
    return core.get_model(db, name)


@state.read
def get_labels(db: Session):
    """Retrieves all existing labels."""
    pass
