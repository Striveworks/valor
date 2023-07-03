from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from velour_api import exceptions
from velour_api.backend import models, state
from velour_api.schemas import (
    Dataset,
    DatasetInfo,
    Model,
    ModelInfo,
)


@state.read
def get_datasets() -> list[DatasetInfo]:
    pass


@state.read
def get_dataset() -> Dataset:
    pass


@state.read
def get_models() -> list[ModelInfo]:
    pass


@state.read
def get_model() -> Model:
    pass


@state.read
def get_labels():
    """Retrieves all existing labels."""
    pass
