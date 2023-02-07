import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from velour_api import crud, models, schemas


def test_create_and_get_dataset(db: Session):
    dset_name = "test dataset"
    crud.create_dataset(db, schemas.DatasetCreate(name=dset_name))

    all_datasets = db.scalars(select(models.Dataset)).all()
    assert len(all_datasets) == 1
    assert all_datasets[0].name == dset_name

    with pytest.raises(crud.DatasetAlreadyExistsError):
        crud.create_dataset(db, schemas.DatasetCreate(name=dset_name))

    crud.create_dataset(db, schemas.DatasetCreate(name="other dataset"))
    datasets = crud.get_datasets(db)
    assert len(datasets) == 2
    assert set([d.name for d in datasets]) == {dset_name, "other dataset"}
