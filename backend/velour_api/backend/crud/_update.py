from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from velour_api import exceptions
from velour_api.backend import models, state
from velour_api.schemas.core import DatasetCreate


@state.update
def add_groundtruth():
    pass


@state.update
def add_prediction():
    pass
