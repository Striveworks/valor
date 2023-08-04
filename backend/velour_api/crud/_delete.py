from sqlalchemy.orm import Session

from velour_api import backend
from velour_api.backend import jobs


@jobs.delete
def delete_dataset(db: Session, dataset_name: str):
    # job, wrapped_fn = jobs.wrap_method_for_job(crud.delete_dataset)
    # background_tasks.add_task(wrapped_fn, db=db, dataset_name=dataset_name)
    # return job.uid
    backend.delete_dataset(db, dataset_name)

    return None  # this return is from the background decorator


@jobs.delete
def delete_model(db: Session, model_name: str):
    backend.delete_model(db, model_name)

    return None  # this return is from the background decorator


@jobs.delete
def prune_labels():
    """@TODO (maybe) Prunes orphans."""
    pass
