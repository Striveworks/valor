from sqlalchemy.orm import Session

from velour_api import schemas
from velour_api.backend import core, state, io


@state.read
def get_labels(db: Session):
    """Retrieves all existing labels."""
    pass


# Datasets

@state.read
def get_datasets(db: Session) -> list[schemas.Dataset]:
    return io.request_datasets()


@state.read
def get_dataset(db: Session, name: str) -> schemas.Dataset:
    return io.request_dataset(db, name)


@state.read
def get_dataset_labels(db: Session, name: str) -> list[schemas.LabelDistribution]:
    pass


# Models

@state.read
def get_models(db: Session) -> list[schemas.Model]:
    return io.request_models(db)


@state.read
def get_model(db: Session, name: str) -> schemas.Model:
    return io.request_model(db, name)


@state.read
def get_model_labels(db: Session, name:str) -> list[schemas.ScoredLabelDistribution]:
    pass



