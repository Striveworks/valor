import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from velour_api import crud, models, schemas


def test_create_and_get_datasets(db: Session):
    dset_name = "test dataset"
    crud.create_dataset(db, schemas.DatasetCreate(name=dset_name))

    all_datasets = db.scalars(select(models.Dataset)).all()
    assert len(all_datasets) == 1
    assert all_datasets[0].name == dset_name

    with pytest.raises(crud.DatasetAlreadyExistsError) as exc_info:
        crud.create_dataset(db, schemas.DatasetCreate(name=dset_name))
    assert "already exists" in str(exc_info)

    crud.create_dataset(db, schemas.DatasetCreate(name="other dataset"))
    datasets = crud.get_datasets(db)
    assert len(datasets) == 2
    assert set([d.name for d in datasets]) == {dset_name, "other dataset"}


def test_get_dataset(db: Session):
    dset_name = "test dataset"

    with pytest.raises(crud.DatasetDoesNotExistError) as exc_info:
        crud.get_dataset(db, dset_name)
    assert "does not exist" in str(exc_info)

    crud.create_dataset(db, schemas.DatasetCreate(name=dset_name))
    dset = crud.get_dataset(db, dset_name)
    assert dset.name == dset_name


def test_delete_dataset(db: Session):
    dset_name = "test dataset"
    crud.create_dataset(db, schemas.DatasetCreate(name=dset_name))

    # sanity check nothing in db
    for model_cls in [
        models.Image,
        models.GroundTruthDetection,
        models.LabeledGroundTruthDetection,
        models.Label,
    ]:
        assert db.scalars(select(func.count(model_cls.id))).first() == 0

    crud.create_groundtruth_detections(
        db,
        data=schemas.GroundTruthDetectionsCreate(
            dataset_name=dset_name,
            detections=[
                schemas.DetectionBase(
                    boundary=[(10, 20), (10, 30), (20, 30), (20, 20)],
                    labels=[schemas.Label(key="k", value="v")],
                    image=schemas.Image(uri="uri1"),
                )
            ],
        ),
    )

    # should have one row for all of these tables
    for model_cls in [
        models.Image,
        models.GroundTruthDetection,
        models.LabeledGroundTruthDetection,
    ]:
        assert db.scalars(select(func.count(model_cls.id))).first() == 1

    # delete dataset and check the cascade worked
    crud.delete_dataset(db, dataset_name=dset_name)
    for model_cls in [
        models.Dataset,
        models.Image,
        models.GroundTruthDetection,
        models.LabeledGroundTruthDetection,
    ]:
        assert db.scalars(select(func.count(model_cls.id))).first() == 0

    # make åsure labels are still there`
    assert db.scalars(select(func.count(models.Label.id))).first() == 1
