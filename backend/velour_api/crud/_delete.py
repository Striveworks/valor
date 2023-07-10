from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from velour_api import exceptions, schemas
from velour_api.backend import models, state, jobs, core


@state.background
@state.delete
def delete_dataset(db: Session, name: str) -> str:
    # job, wrapped_fn = jobs.wrap_method_for_job(crud.delete_dataset)
    # background_tasks.add_task(wrapped_fn, db=db, dataset_name=dataset_name)
    # return job.uid
    core.delete_dataset(db, name)
    return "1"


@state.delete
def delete_model(db: Session, name: str) -> str:
    core.delete_model(db, name)
    return "1"


@state.delete
def prune_labels():
    """@TODO (maybe) Prunes orphans."""
    pass
