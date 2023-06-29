from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from velour_api import exceptions
from velour_api.backend import models, state
from velour_api.schemas.core import DatasetCreate


@state.read
def get_datasets():
    pass


@state.read
def get_dataset():
    pass


@state.read
def get_models():
    pass


@state.read
def get_model():
    pass


@state.read
def get_labels():
    """Retrieves all existing labels."""
    pass
