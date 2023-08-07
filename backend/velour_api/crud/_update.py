from sqlalchemy.orm import Session

from velour_api.backend import jobs


# @TODO
# @NOTE: Right now finalize only interacts with redis but not the sql backend.
@jobs.finalize
def finalize(*, db: Session, dataset_name: str, model_name: str = None):
    pass
