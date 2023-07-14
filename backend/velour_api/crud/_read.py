from sqlalchemy.orm import Session

from velour_api import backend, schemas
from velour_api.backend import state


@state.read
def get_labels(
    db: Session,
    key: str = None,
    dataset_name: str = None,
    model_name: str = None,
):
    """Retrieves all existing labels."""
    return backend.get_labels(
        db,
        key=key,
        dataset_name=dataset_name,
        model_name=model_name,
    )


# Datasets


@state.read
def get_datasets(db: Session) -> list[schemas.Dataset]:
    return backend.get_datasets()


@state.read
def get_dataset(db: Session, name: str) -> schemas.Dataset:
    return backend.get_dataset(db, name)


@state.read
def get_groundtruth(
    db: Session, dataset_name: str, datum_uid: str
) -> schemas.GroundTruth:
    return backend.get_groundtruth(
        db, dataset_name=dataset_name, datum_uid=datum_uid
    )


@state.read
def get_dataset_labels(
    db: Session, name: str
) -> list[schemas.LabelDistribution]:
    pass


# Models


@state.read
def get_models(db: Session) -> list[schemas.Model]:
    return backend.get_models(db)


@state.read
def get_model(db: Session, name: str) -> schemas.Model:
    return backend.get_model(db, name)


@state.read
def get_prediction(
    db: Session, model_name: str, datum_uid: str
) -> schemas.Prediction:
    return backend.get_prediction(
        db, model_name=model_name, datum_uid=datum_uid
    )


@state.read
def get_model_labels(
    db: Session, name: str
) -> list[schemas.ScoredLabelDistribution]:
    pass
