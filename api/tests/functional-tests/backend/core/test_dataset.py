import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from velour_api import enums, exceptions, schemas
from velour_api.backend import core, models


@pytest.fixture
def created_dataset(db: Session, dataset_name: str) -> str:
    dataset = schemas.Dataset(name=dataset_name)
    core.create_dataset(db, dataset=dataset)
    return dataset_name


@pytest.fixture
def created_datasets(db: Session) -> list[str]:
    dataset1 = schemas.Dataset(name="dataset1")
    dataset2 = schemas.Dataset(name="dataset2")
    core.create_dataset(db, dataset=dataset1)
    core.create_dataset(db, dataset=dataset2)
    return ["dataset1", "dataset2"]


def test_created_dataset(db: Session, created_dataset):
    dataset = db.query(
        select(models.Dataset)
        .where(models.Dataset.name == created_dataset)
        .subquery()
    ).one_or_none()
    assert dataset is not None
    assert dataset.name == created_dataset
    assert dataset.meta == {}


def test_fetch_dataset(db: Session, created_dataset):
    dataset = core.fetch_dataset(db, created_dataset)
    assert dataset is not None
    assert dataset.name == created_dataset
    assert dataset.meta == {}

    with pytest.raises(exceptions.DatasetDoesNotExistError):
        core.fetch_dataset(db, "some_nonexistent_dataset")


def test_get_dataset(db: Session, created_dataset):
    dataset = core.get_dataset(db, created_dataset)
    assert dataset is not None
    assert dataset.name == created_dataset
    assert dataset.metadata == {}
    assert dataset.geospatial is None

    with pytest.raises(exceptions.DatasetDoesNotExistError):
        core.get_dataset(db, "some_nonexistent_dataset")


def test_get_datasets(db: Session, created_datasets):
    datasets = core.get_datasets(db)
    for dataset in datasets:
        assert dataset.name in created_datasets


def test_dataset_status(db: Session, created_dataset):
    # creating
    assert (
        core.get_dataset_status(db, created_dataset)
        == enums.TableStatus.CREATING
    )

    # finalized
    core.set_dataset_status(db, created_dataset, enums.TableStatus.FINALIZED)
    assert (
        core.get_dataset_status(db, created_dataset)
        == enums.TableStatus.FINALIZED
    )

    # test others
    core.set_dataset_status(db, created_dataset, enums.TableStatus.FINALIZED)
    with pytest.raises(exceptions.DatasetStateError):
        core.set_dataset_status(
            db, created_dataset, enums.TableStatus.CREATING
        )

    # deleting
    core.set_dataset_status(db, created_dataset, enums.TableStatus.DELETING)
    assert (
        core.get_dataset_status(db, created_dataset)
        == enums.TableStatus.DELETING
    )

    # test others
    with pytest.raises(exceptions.DatasetStateError):
        core.set_dataset_status(
            db, created_dataset, enums.TableStatus.CREATING
        )
    with pytest.raises(exceptions.DatasetStateError):
        core.set_dataset_status(
            db, created_dataset, enums.TableStatus.FINALIZED
        )


def test_dataset_status_create_to_delete(db: Session, created_dataset):
    # creating
    assert (
        core.get_dataset_status(db, created_dataset)
        == enums.TableStatus.CREATING
    )

    # deleting
    core.set_dataset_status(db, created_dataset, enums.TableStatus.DELETING)
    assert (
        core.get_dataset_status(db, created_dataset)
        == enums.TableStatus.DELETING
    )


def test_delete_dataset(db: Session):
    core.create_dataset(db=db, dataset=schemas.Dataset(name="dataset1"))

    assert (
        db.scalar(
            select(func.count())
            .select_from(models.Dataset)
            .where(models.Dataset.name == "dataset1")
        )
        == 1
    )

    core.delete_dataset(db=db, name="dataset1")

    assert (
        db.scalar(
            select(func.count())
            .select_from(models.Dataset)
            .where(models.Dataset.name == "dataset1")
        )
        == 0
    )
