from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from velour_api import exceptions
from velour_api.backend import models, state
from velour_api.schemas.core import DatasetCreate


@state.delete
def delete_dataset():
    pass


@state.delete
def delete_model():
    pass


@state.delete
def prune_labels():
    """@TODO (maybe) Prunes orphans."""
    pass
