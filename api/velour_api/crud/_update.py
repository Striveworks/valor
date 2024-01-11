from sqlalchemy.orm import Session

from velour_api import backend, enums, schemas
from velour_api.backend import set_dataset_status, set_model_status


def finalize(*, db: Session, dataset_name: str, model_name: str = None):
    """
    Finalizes dataset and dataset/model pairings.
    """
    if dataset_name and model_name:
        set_model_status(
            db=db,
            dataset_name=dataset_name,
            model_name=model_name,
            status=enums.TableStatus.FINALIZED,
        )
    elif dataset_name:
        set_dataset_status(
            db=db,
            name=dataset_name,
            status=enums.TableStatus.FINALIZED,
        )
