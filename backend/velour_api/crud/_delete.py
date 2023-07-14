from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from velour_api import backend, exceptions, schemas
from velour_api.backend import jobs, state


@state.background
@state.delete
def delete_dataset(db: Session, name: str) -> str:
    # job, wrapped_fn = jobs.wrap_method_for_job(crud.delete_dataset)
    # background_tasks.add_task(wrapped_fn, db=db, dataset_name=dataset_name)
    # return job.uid
    backend.delete_dataset(db, name)
    return "1"


@state.delete
def delete_model(db: Session, name: str) -> str:
    backend.delete_model(db, name)
    return "1"


@state.delete
def prune_labels():
    """@TODO (maybe) Prunes orphans."""
    pass
