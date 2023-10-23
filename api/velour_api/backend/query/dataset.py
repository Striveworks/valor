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
            meta=core.deserialize_meta(dataset.metadata),
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
        metadata=core.serialize_meta(dataset.meta),
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
    filters: schemas.Filter | None = None,
) -> list[schemas.Datum]:

    if not filters:
        datums = db.query(models.Datum).all()
    else:
        q = ops.Query(models.Datum).filter(filters).query()
        datums = db.query(q).all()

    return [
        schemas.Datum(
            dataset=db.scalar(
                select(models.Dataset.name).where(
                    models.Dataset.id == datum.dataset_id
                )
            ),
            uid=datum.uid,
            metadata=core.serialize_meta(datum.meta),
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
