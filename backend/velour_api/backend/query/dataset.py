from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from velour_api import exceptions, schemas
from velour_api.backend import core, models, ops


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
        raise exceptions.DatasetDoesNotExistError(name)

    metadata = []
    for row in (
        db.query(models.MetaDatum)
        .where(models.MetaDatum.dataset_id == dataset.id)
        .all()
    ):
        if row.string_value:
            metadata.append(
                schemas.MetaDatum(key=row.key, value=row.string_value)
            )
        elif row.numeric_value:
            metadata.append(
                schemas.MetaDatum(key=row.key, value=row.numeric_value)
            )
        elif row.geo:
            metadata.append(
                schemas.MetaDatum(
                    key=row.key,
                    value=row.geo,
                )
            )

    return schemas.Dataset(id=dataset.id, name=dataset.name, metadata=metadata)


def get_dataset_name_from_datum_uid(
    db: Session,
    uid: str,
) -> str | None:
    """Returns dataset name"""
    return db.scalar(
        select(models.Dataset.name)
        .join(models.Datum, models.Datum.dataset_id == models.Dataset.id)
        .where(models.Datum.uid == uid)
    )


def get_datasets(
    db: Session,
) -> list[schemas.Dataset]:
    return [
        get_dataset(db, name)
        for name in db.scalars(select(models.Dataset.name)).all()
    ]


def get_datums(
    db: Session,
    request: schemas.Filter | None = None,
) -> list[schemas.Datum]:

    if not request:
        datums = db.query(models.Datum).all()
    else:
        datums = ops.BackendQuery.datum().filter(request).all(db)

    print(ops.BackendQuery.datum().filter(request))

    return [
        schemas.Datum(
            dataset=db.scalar(
                select(models.Dataset.name).where(
                    models.Dataset.id == datum.dataset_id
                )
            ),
            uid=datum.uid,
            metadata=core.get_metadata(db, datum=datum),
        )
        for datum in datums
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
