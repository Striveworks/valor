from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from velour_api import exceptions, schemas
from velour_api.backend import core, models


def create_model(
    db: Session,
    model: schemas.Model,
):
    # Check if dataset already exists.
    if (
        db.query(models.Model)
        .where(models.Model.name == model.name)
        .one_or_none()
    ):
        raise exceptions.ModelAlreadyExistsError(model.name)

    # Create model
    row = models.Model(name=model.name)
    try:
        db.add(row)
        db.commit()
    except IntegrityError:
        db.rollback()
        raise exceptions.ModelAlreadyExistsError(model.name)

    # Create metadata
    core.create_metadata(db, model.metadata, model=row)
    return row


def get_model(
    db: Session,
    name: str,
) -> schemas.Model:

    model = (
        db.query(models.Model).where(models.Model.name == name).one_or_none()
    )
    if not model:
        return None

    metadata = []
    for row in (
        db.query(models.MetaDatum)
        .where(models.MetaDatum.model_id == model.id)
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

    return schemas.Model(id=model.id, name=model.name, metadata=metadata)


def get_models(
    db: Session,
) -> list[schemas.Model]:
    return [
        get_model(db, name) for name in db.scalars(select(models.Model.name))
    ]


def delete_model(
    db: Session,
    name: str,
):
    md = db.query(models.Model).where(models.Model.name == name).one_or_none()
    if not md:
        raise exceptions.ModelDoesNotExistError(name)
    try:
        db.delete(md)
        db.commit()
    except IntegrityError:
        db.rollback()
        raise RuntimeError
