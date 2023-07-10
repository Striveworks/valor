from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from velour_api import exceptions, schemas
from velour_api.backend import models, state, jobs


@state.background
@state.delete
def delete_dataset(dataset: schemas.Dataset):
    # job, wrapped_fn = jobs.wrap_method_for_job(crud.delete_dataset)
    # background_tasks.add_task(wrapped_fn, db=db, dataset_name=dataset_name)
    # return job.uid
    pass


@state.delete
def delete_model(model: schemas.Model):
    pass


@state.delete
def prune_labels():
    """@TODO (maybe) Prunes orphans."""
    pass
