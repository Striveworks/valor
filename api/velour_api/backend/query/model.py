from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from velour_api import exceptions, schemas
from velour_api.backend import core, models


def create_model(
    db: Session,
    model: schemas.Model,
):
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
    model = core.get_model(db, name=name)
    metadata = core.get_metadata(db, model=model)
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
    model = core.get_model(db, name=name)
    try:
        db.delete(model)
        db.commit()
    except IntegrityError:
        db.rollback()
        raise RuntimeError
