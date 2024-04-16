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


def test_get_paginated_datums(
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
    datums, _ = core.get_paginated_datums(db=db)
    assert {datum.uid for datum in datums} == {
        "uid1",
        "uid2",
        "uid3",
    }

    # test that we can reconstitute the full set using paginated calls
    first, header = core.get_paginated_datums(db, offset=1, limit=2)
    assert len(first) == 2
    assert header == {"content-range": "items 1-2/3"}

    second, header = core.get_paginated_datums(db, offset=0, limit=1)
    assert len(second) == 1
    assert header == {"content-range": "items 0-0/3"}

    combined = [entry.uid for entry in first + second]

    assert set(combined) == set([f"uid{i}" for i in range(1, 4)])
