from sqlalchemy.orm import Session

from velour_api import backend
from velour_api.backend import stateflow


@stateflow.delete
def delete(
    *,
    db: Session,
    dataset_name: str | None = None,
    model_name: str | None = None,
):
    if dataset_name is not None:
        backend.delete_dataset(db, dataset_name)
    elif model_name is not None:
        backend.delete_model(db, model_name)


@stateflow.delete
def prune_labels():
    """@TODO (maybe) Prunes orphans."""
    pass
