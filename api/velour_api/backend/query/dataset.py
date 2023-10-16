from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from velour_api import exceptions, schemas
from velour_api.backend import core, models, ops


def create_dataset(
    db: Session,
    dataset: schemas.Dataset,
):
    # Create dataset
    try:
        row = models.Dataset(
            name=dataset.name,
            meta=core.deserialize_metadatums(dataset.metadata),
        )
        db.add(row)
        db.commit()
        return row
    except IntegrityError:
        db.rollback()
        raise exceptions.DatasetAlreadyExistsError(dataset.name)


def get_dataset(
    db: Session,
    name: str,
) -> schemas.Dataset:
    # retrieve dataset
    dataset = core.get_dataset(db, name=name)
    return schemas.Dataset(
        id=dataset.id,
        name=dataset.name,
        metadata=core.serialize_metadatums(dataset.meta),
    )


def get_datasets(
    db: Session,
) -> list[schemas.Dataset]:
    return [
        get_dataset(db, name)
        for name in db.scalars(select(models.Dataset.name)).all()
    ]


# @TODO
def get_datums(
    db: Session,
    request: schemas.Filter | None = None,
) -> list[schemas.Datum]:

    if not request:
        datums = db.query(models.Datum).all()
    else:
        datums = ops.BackendQuery.datum().filter(request).all(db)

    return [
        schemas.Datum(
            dataset=db.scalar(
                select(models.Dataset.name).where(
                    models.Dataset.id == datum.dataset_id
                )
            ),
            uid=datum.uid,
            metadata=core.serialize_metadatums(datum.meta),
        )
        for datum in datums
    ]


def delete_dataset(
    db: Session,
    name: str,
):
    dataset = core.get_dataset(db, name=name)
    try:
        db.delete(dataset)
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise e
