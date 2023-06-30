from sqlalchemy.orm import Session

import velour_api.stateflow as stateflow
from velour_api import exceptions, models

from ._read import get_dataset, get_model


@stateflow.finalize
def finalize_dataset(db: Session, dataset_name: str) -> None:
    dataset = get_dataset(db, dataset_name)
    dataset.finalized = True
    db.commit()


@stateflow.finalize
def finalize_inferences(
    db: Session, model_name: str, dataset_name: str
) -> None:
    dataset = get_dataset(db, dataset_name)
    if not dataset.finalized:
        raise exceptions.DatasetIsNotFinalizedError(dataset_name)

    model_id = get_model(db, model_name).id
    dataset_id = dataset.id

    db.add(
        models.FinalizedInferences(dataset_id=dataset_id, model_id=model_id)
    )
    db.commit()
