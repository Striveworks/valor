import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from valor_api import exceptions, schemas
from valor_api.backend import core, models


@pytest.fixture
def created_dataset(db: Session, dataset_name: str) -> str:
    dataset = schemas.Dataset(name=dataset_name)
    core.create_dataset(db, dataset=dataset)
    return dataset_name


def test_create_datum(
    db: Session,
    created_dataset: str,
):
    assert db.scalar(select(func.count()).select_from(models.Datum)) == 0
    dataset = core.fetch_dataset(db=db, name=created_dataset)

    # test successful
    core.create_datum(
        db=db,
        datum=schemas.Datum(uid="uid1"),
        dataset=dataset,
    )

    assert db.scalar(select(func.count()).select_from(models.Datum)) == 1

    # test catch duplicate
    with pytest.raises(exceptions.DatumAlreadyExistsError):
        core.create_datum(
            db=db,
            datum=schemas.Datum(uid="uid1"),
            dataset=dataset,
        )

    assert db.scalar(select(func.count()).select_from(models.Datum)) == 1

    # test successful 2nd datum
    core.create_datum(
        db=db,
        datum=schemas.Datum(uid="uid2"),
        dataset=dataset,
    )

    assert db.scalar(select(func.count()).select_from(models.Datum)) == 2


def test_get_datums(
    db: Session,
    created_dataset: str,
):
    dataset = core.fetch_dataset(db=db, name=created_dataset)

    core.create_datum(
        db=db,
        datum=schemas.Datum(uid="uid1"),
        dataset=dataset,
    )
    core.create_datum(
        db=db,
        datum=schemas.Datum(uid="uid2"),
        dataset=dataset,
    )
    core.create_datum(
        db=db,
        datum=schemas.Datum(uid="uid3"),
        dataset=dataset,
    )

    # basic query
    assert {datum.uid for datum in core.get_datums(db=db)} == {
        "uid1",
        "uid2",
        "uid3",
    }
