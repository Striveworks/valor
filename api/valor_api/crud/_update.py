from fastapi import BackgroundTasks
from sqlalchemy.orm import Session

from valor_api import enums
from valor_api.backend import set_dataset_status, set_model_status
from valor_api.backend.database import vacuum_analyze


def finalize(
    *,
    db: Session,
    dataset_name: str,
    model_name: str | None = None,
    task_handler: BackgroundTasks | None = None,
):
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

    if task_handler:
        task_handler.add_task(vacuum_analyze)
    else:
        vacuum_analyze()
