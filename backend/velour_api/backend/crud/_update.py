from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from velour_api.backend import state


@state.update
def update_something():
    pass

