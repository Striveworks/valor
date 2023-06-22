from sqlalchemy.orm import Session

import velour_api.stateflow as stateflow

from ._read import get_dataset, get_model


@stateflow.delete_dataset
def delete_dataset(db: Session, dataset_name: str):
    dset = get_dataset(db, dataset_name)

    db.delete(dset)
    db.commit()


@stateflow.delete_model
def delete_model(db: Session, model_name: str):
    model = get_model(db, model_name)

    db.delete(model)
    db.commit()
