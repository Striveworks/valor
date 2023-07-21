from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy import select

from velour_api import exceptions, schemas
from velour_api.backend import core, models


def create_dataset(
    db: Session,
    dataset: schemas.Dataset,
):
    # Check if dataset already exists.
    if (
        db.query(models.Dataset)
        .where(models.Dataset.name == dataset.name)
        .one_or_none()
    ):
        raise exceptions.DatasetAlreadyExistsError(dataset.name)

    # Create dataset
    try:
        row = models.Dataset(name=dataset.name)
        db.add(row)
        db.commit()
    except IntegrityError:
        db.rollback()
        raise exceptions.DatasetAlreadyExistsError(dataset.name)

    # Create metadata
    core.create_metadata(db, dataset.metadata, dataset=row)
    return row


def get_dataset(
    db: Session,
    name: str,
) -> schemas.Dataset:

    dataset = (
        db.query(models.Dataset)
        .where(models.Dataset.name == name)
        .one_or_none()
    )
    if not dataset:
        return None

    metadata = []
    for row in (
        db.query(models.MetaDatum)
        .where(models.MetaDatum.dataset_id == dataset.id)
        .all()
    ):
        if row.string_value:
            metadata.append(
                schemas.MetaDatum(name=row.name, value=row.string_value)
            )
        elif row.numeric_value:
            metadata.append(
                schemas.MetaDatum(name=row.name, value=row.numeric_value)
            )
        elif row.geo:
            metadata.append(
                schemas.MetaDatum(
                    name=row.name,
                    value=row.geo,
                )
            )
        elif row.image_id:
            if (
                image_metadatum := db.query(models.ImageMetadata)
                .where(models.ImageMetadata.id == row.image_id)
                .one_or_none()
            ):
                metadata.append(
                    schemas.MetaDatum(
                        name=row.name,
                        value=schemas.ImageMetadata(
                            height=image_metadatum.height,
                            width=image_metadatum.width,
                            frame=image_metadatum.frame,
                        ),
                    )
                )

    return schemas.Dataset(id=dataset.id, name=dataset.name, metadata=metadata)


def get_datasets(
    db: Session,
) -> list[schemas.Dataset]:
    return [
        get_dataset(db, name) 
        for name in db.scalars(select(models.Dataset.name))
    ]


def delete_dataset(
    db: Session,
    name: str,
):
    ds = (
        db.query(models.Dataset)
        .where(models.Dataset.name == name)
        .one_or_none()
    )
    if not ds:
        raise exceptions.DatasetDoesNotExistError(name)
    try:
        db.delete(ds)
        db.commit()
    except IntegrityError:
        db.rollback()
        raise RuntimeError
