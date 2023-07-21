from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from velour_api.backend import state

# @TODO
@state.update
def finalize(db: Session, dataset_name: str, model_name: str = None):
    if model_name:
        # finalize inferences
        pass
    else:
        # finalize dataset
        pass
