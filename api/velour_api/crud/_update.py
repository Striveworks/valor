from sqlalchemy.orm import Session

from velour_api.backend import stateflow


# @NOTE: Right now finalize only interacts with redis but not the sql backend.
@stateflow.finalize
def finalize(*, db: Session, dataset_name: str, model_name: str = None):
    pass
