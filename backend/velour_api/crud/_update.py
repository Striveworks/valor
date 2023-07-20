from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from velour_api.backend import state


@state.update
def finalize_dataset(db: Session, dataset_name: str):
    pass

