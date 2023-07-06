from sqlalchemy.orm import Session
from sqlalchemy import select, insert, and_

from velour_api import schemas, exceptions
from velour_api.backend import models

from velour_api.backend.core.metadata import create_metadata
from velour_api.backend.core.label import create_labels
from velour_api.backend.core.geometry import create_geometric_annotation


def get_dataset_info(
    db: Session,
    name: str,
) -> schemas.DatasetInfo:
    # return dataset info with all assoiated metadata
    pass


def get_datum(
    db: Session,
    datum: schemas.Datum,
) -> models.Datum:
    return db.query(models.Datum).where(models.Datum.uid == datum.uid).one_or_none()


def create_dataset_info(
    db: Session,
    info: schemas.DatasetInfo,
) -> models.Dataset:
    
    # Create dataset
    row = models.Dataset(name=info.name)
    db.add(row)
    db.commit()

    # Create metadata
    create_metadata(db, info.metadata, dataset=row)
    return row


def create_datum(
    db: Session, 
    datum: schemas.Datum,
    dataset: models.Dataset,
) -> models.Datum:
    
    # Create datum
    row = models.Datum(uid=datum.uid, dataset=dataset.id)
    db.add(row)
    db.commit()

    # Create metadata
    create_metadata(db, datum.metadata, datum_id=row)
    return row


def create_groundtruths(
    db: Session,
    gts: list[schemas.GroundTruth],
    datum: models.Datum,
) -> list[models.GroundTruth]:
    rows = []
    for gt in gts:
        geometry = create_geometric_annotation(db, gt.geometry) if gt.geometry else None
        rows += [
            models.GroundTruth(
                datum_id=datum.id,
                task_type=gt.task_type,
                geometry=geometry.id,
                label=label.id,
            )
            for label in create_labels(db, gt.labels)
        ]
    db.add_all(rows)
    db.commit()
    return rows


def create_dataset(
    db: Session,
    dataset: schemas.Dataset,
):
    # Check if dataset already exists.
    if not db.query(models.Dataset).where(models.Dataset.name == dataset.info.name).one_or_none():
        raise exceptions.DatasetAlreadyExistsError(dataset.info.name)
    
    dataset_row = create_dataset_info(db, dataset.info)                             # Create dataset
    for datum in dataset.datums:
        datum_row = create_datum(db, datum, dataset=dataset_row)                    # Create datums
        create_groundtruths(db, datum.gts, dataset=dataset_row, datum=datum_row)    # Create groundtruths

