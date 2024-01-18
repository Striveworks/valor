import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from velour_api import enums, exceptions, schemas
from velour_api.backend import core, models


@pytest.fixture
def created_dataset(db: Session, dataset_name: str) -> str:
    dataset = schemas.Dataset(name=dataset_name)
    core.create_dataset(db, dataset=dataset)
    core.create_groundtruth(
        db=db,
        groundtruth=schemas.GroundTruth(
            datum=schemas.Datum(uid="uid1", dataset_name=dataset_name),
            annotations=[
                schemas.Annotation(
                    task_type=enums.TaskType.CLASSIFICATION,
                    labels=[schemas.Label(key="k1", value="v1")],
                )
            ],
        ),
    )
    core.create_groundtruth(
        db=db,
        groundtruth=schemas.GroundTruth(
            datum=schemas.Datum(uid="uid2", dataset_name=dataset_name),
            annotations=[
                schemas.Annotation(
                    task_type=enums.TaskType.DETECTION,
                    labels=[schemas.Label(key="k1", value="v1")],
                )
            ],
        ),
    )
    core.create_groundtruth(
        db=db,
        groundtruth=schemas.GroundTruth(
            datum=schemas.Datum(uid="uid3", dataset_name=dataset_name),
            annotations=[
                schemas.Annotation(
                    task_type=enums.TaskType.SEGMENTATION,
                    labels=[schemas.Label(key="k1", value="v1")],
                )
            ],
        ),
    )
    return dataset_name


@pytest.fixture
def created_model(db: Session, model_name: str, created_dataset: str) -> str:
    model = schemas.Model(name=model_name)
    core.create_model(db, model=model)
    core.create_prediction(
        db=db,
        prediction=schemas.Prediction(
            model_name=model_name,
            datum=schemas.Datum(uid="uid1", dataset_name=created_dataset),
            annotations=[
                schemas.Annotation(
                    task_type=enums.TaskType.CLASSIFICATION,
                    labels=[schemas.Label(key="k1", value="v1", score=1.0)],
                )
            ],
        ),
    )
    core.create_prediction(
        db=db,
        prediction=schemas.Prediction(
            model_name=model_name,
            datum=schemas.Datum(uid="uid2", dataset_name=created_dataset),
            annotations=[
                schemas.Annotation(
                    task_type=enums.TaskType.DETECTION,
                    labels=[schemas.Label(key="k1", value="v1", score=1.0)],
                )
            ],
        ),
    )
    core.create_prediction(
        db=db,
        prediction=schemas.Prediction(
            model_name=model_name,
            datum=schemas.Datum(uid="uid3", dataset_name=created_dataset),
            annotations=[
                schemas.Annotation(
                    task_type=enums.TaskType.SEGMENTATION,
                    labels=[schemas.Label(key="k1", value="v1")],
                )
            ],
        ),
    )
    return model_name


@pytest.fixture
def created_datasets(db: Session) -> list[str]:
    dataset1 = schemas.Dataset(name="dataset1")
    dataset2 = schemas.Dataset(name="dataset2")
    core.create_dataset(db, dataset=dataset1)
    core.create_dataset(db, dataset=dataset2)
    return ["dataset1", "dataset2"]


def test_create_dataset(db: Session, created_dataset):
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


def test_get_all_datasets(db: Session, created_datasets):
    datasets = core.get_all_datasets(db)
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


def test_dataset_status_with_evaluations(
    db: Session,
    created_dataset: str,
    created_model: str,
):
    # create an evaluation
    core.set_dataset_status(db, created_dataset, enums.TableStatus.FINALIZED)
    evaluations, _ = core.create_or_get_evaluations(
        db,
        schemas.EvaluationRequest(
            model_filter=schemas.Filter(model_names=[created_model]),
            evaluation_filter=schemas.Filter(
                dataset_names=[created_dataset],
                task_types=[enums.TaskType.CLASSIFICATION],
            ),
        ),
    )
    assert len(evaluations) == 1
    evaluation_id = evaluations[0].id

    # set the evaluation to the running state
    core.set_evaluation_status(
        db, evaluation_id, enums.EvaluationStatus.RUNNING
    )

    # test that deletion is blocked while evaluation is running
    with pytest.raises(exceptions.EvaluationRunningError):
        core.set_dataset_status(
            db, created_dataset, enums.TableStatus.DELETING
        )

    # set the evaluation to the done state
    core.set_evaluation_status(db, evaluation_id, enums.EvaluationStatus.DONE)

    # test that deletion is unblocked when evaluation is DONE
    core.set_dataset_status(db, created_dataset, enums.TableStatus.DELETING)


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


def test_get_n_datums_in_dataset(
    db: Session, dataset_name: str, dataset_model_create
):
    assert core.get_n_datums_in_dataset(db=db, name=dataset_name) == 2


def test_get_n_groundtruth_annotations(
    db: Session, dataset_name: str, dataset_model_create
):
    assert core.get_n_groundtruth_annotations(db=db, name=dataset_name) == 6


def test_get_n_groundtruth_bounding_boxes_in_dataset(
    db: Session, dataset_name: str, dataset_model_create
):
    assert (
        core.get_n_groundtruth_bounding_boxes_in_dataset(
            db=db, name=dataset_name
        )
        == 3
    )


def test_get_n_groundtruth_polygons_in_dataset(
    db: Session, dataset_name: str, dataset_model_create
):
    assert (
        core.get_n_groundtruth_polygons_in_dataset(db=db, name=dataset_name)
        == 1
    )


def test_get_n_groundtruth_multipolygons_in_dataset(
    db: Session, dataset_name: str, dataset_model_create
):
    assert (
        core.get_n_groundtruth_multipolygons_in_dataset(
            db=db, name=dataset_name
        )
        == 0
    )


def test_get_n_groundtruth_rasters_in_dataset(
    db: Session, dataset_name: str, dataset_model_create
):
    assert (
        core.get_n_groundtruth_rasters_in_dataset(db=db, name=dataset_name)
        == 1
    )


def test_get_unique_task_types_in_dataset(
    db: Session, dataset_name: str, dataset_model_create
):
    assert set(
        core.get_unique_task_types_in_dataset(db=db, name=dataset_name)
    ) == set(
        [enums.TaskType.DETECTION.value, enums.TaskType.CLASSIFICATION.value]
    )


def test_get_unique_datum_metadata_in_dataset(
    db: Session, dataset_name: str, dataset_model_create
):
    unique_metadata = core.get_unique_datum_metadata_in_dataset(
        db=db, name=dataset_name
    )
    unique_metadata.sort(key=lambda x: x["width"])
    assert unique_metadata == [
        {"width": 32.0, "height": 80.0},
        {"width": 200.0, "height": 100.0},
    ]


def test_get_unique_groundtruth_annotation_metadata_in_dataset(
    db: Session, dataset_name: str, dataset_model_create
):
    unique_metadata = (
        core.get_unique_groundtruth_annotation_metadata_in_dataset(
            db=db, name=dataset_name
        )
    )

    assert len(unique_metadata) == 2
    assert {"int_key": 1} in unique_metadata
    assert {"string_key": "string_val", "int_key": 1} in unique_metadata
