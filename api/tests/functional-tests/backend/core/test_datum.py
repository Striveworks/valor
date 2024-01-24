import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from velour_api import exceptions, schemas
from velour_api.backend import core, models


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

    # test successful
    core.create_datum(
        db=db,
        datum=schemas.Datum(
            uid="uid1",
            dataset_name=created_dataset,
        ),
    )

    assert db.scalar(select(func.count()).select_from(models.Datum)) == 1

    # test catch `DatasetDoesNotExistError`
    with pytest.raises(exceptions.DatasetDoesNotExistError):
        core.create_datum(
            db=db,
            datum=schemas.Datum(
                uid="uid2",
                dataset_name="dataset_that_doesnt_exist",
            ),
        )

    # test catch duplicate
    with pytest.raises(exceptions.DatumAlreadyExistsError):
        core.create_datum(
            db=db,
            datum=schemas.Datum(
                uid="uid1",
                dataset_name=created_dataset,
            ),
        )

    assert db.scalar(select(func.count()).select_from(models.Datum)) == 1

    # test successful 2nd datum
    core.create_datum(
        db=db,
        datum=schemas.Datum(
            uid="uid2",
            dataset_name=created_dataset,
        ),
    )

    assert db.scalar(select(func.count()).select_from(models.Datum)) == 2


def test_get_datums(
    db: Session,
    created_dataset: str,
):
    core.create_datum(
        db=db,
        datum=schemas.Datum(
            uid="uid1",
            dataset_name=created_dataset,
        ),
    )
    core.create_datum(
        db=db,
        datum=schemas.Datum(
            uid="uid2",
            dataset_name=created_dataset,
        ),
    )
    core.create_datum(
        db=db,
        datum=schemas.Datum(
            uid="uid3",
            dataset_name=created_dataset,
        ),
    )

    # basic query
    assert {datum.uid for datum in core.get_datums(db)} == {
        "uid1",
        "uid2",
        "uid3",
    }
