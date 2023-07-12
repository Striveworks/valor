from sqlalchemy.orm import Session

from velour_api import schemas
from velour_api.backend import core, state, query


@state.read
def get_labels(db: Session):
    """Retrieves all existing labels."""
    pass


# Datasets

@state.read
def get_datasets(db: Session) -> list[schemas.Dataset]:
    return query.get_datasets()


@state.read
def get_dataset(db: Session, name: str) -> schemas.Dataset:
    return query.get_dataset(db, name)


@state.read
def get_groundtruth(db: Session, dataset_name: str, datum_uid: str) -> schemas.GroundTruth:
    return query.get_groundtruth(db, dataset_name, datum_uid)


@state.read
def get_dataset_labels(db: Session, name: str) -> list[schemas.LabelDistribution]:
    pass


# Models


@state.read
def get_models(db: Session) -> list[schemas.Model]:
    return query.get_models(db)


@state.read
def get_model(db: Session, name: str) -> schemas.Model:
    return query.get_model(db, name)


@state.read
def get_prediction(db: Session, model_name: str, datum_uid: str) -> schemas.Prediction:
    return query.get_prediction(db, model_name, datum_uid)


@state.read
def get_model_labels(db: Session, name:str) -> list[schemas.ScoredLabelDistribution]:
    pass



