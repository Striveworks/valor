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
    dataset_name: str,
    annotated_datums: list[schemas.AnnotatedDatum],
):
    dataset = core.get_dataset(dataset_name)
    core.create_groundtruths(
        db, dataset=dataset, annotated_datums=annotated_datums
    )


@state.create
def create_predictions(
    db: Session,
    model_name: str,
    annotated_datums: list[schemas.AnnotatedDatum],
):
    model = core.get_model(model_name)
    core.create_predictions(db, annotated_datums=annotated_datums, model=model)
