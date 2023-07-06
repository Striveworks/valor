from sqlalchemy.orm import Session
from sqlalchemy import and_

from velour_api import schemas
from velour_api.backend import models
from velour_api.backend.core.dataset import get_datum
from velour_api.backend.core.metadata import create_metadata


def create_model_info(
    db: Session,
    info: schemas.ModelInfo
) -> models.Model:
    
    # Create model
    row = models.Model(name=info.name)
    db.add(row)
    db.commit()

    # Create metadata
    create_metadata(db, info.metadata, model=row)
    return row


def create_predictions(
    db: Session,
    pds: list[schemas.Prediction],
    datum: models.Datum,
) -> list(models.Prediction):
    pass


def create_model(
    db: Session,
    model: schemas.Model,
):
    # Check if dataset already exists.
    if not db.query(models.Dataset).where(models.Dataset.name == dataset.info.name).one_or_none():
        raise exceptions.DatasetAlreadyExistsError(dataset.info.name)
    
    dataset_row = create_dataset_info(db, dataset.info)                             # Create dataset
    for datum in dataset.datums:
        datum_row = create_datum(db, datum, dataset=dataset_row)                    # Create datums
        create_groundtruths(db, datum.gts, dataset=dataset_row, datum=datum_row)    # Create groundtruths