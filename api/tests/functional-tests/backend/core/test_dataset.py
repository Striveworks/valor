import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from valor_api import enums, exceptions, schemas
from valor_api.backend import core, models


@pytest.fixture
def created_datasets(db: Session) -> list[str]:
    datasets = []
    for i in range(10):
        dataset = schemas.Dataset(name=f"dataset{i}")
        core.create_dataset(db, dataset=dataset)
        datasets.append(f"dataset{i}")

    return datasets


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

    with pytest.raises(exceptions.DatasetDoesNotExistError):
        core.get_dataset(db, "some_nonexistent_dataset")


def test_get_paginated_datasets(db: Session, created_datasets):
    datasets, headers = core.get_paginated_datasets(db)
    for dataset in datasets:
        assert dataset.name in created_datasets
    assert headers == {"content-range": "items 0-9/10"}

    # test pagination
    with pytest.raises(ValueError):
        # offset is greater than the number of items returned in query
        datasets, headers = core.get_paginated_datasets(
            db, offset=100, limit=2
        )

    datasets, headers = core.get_paginated_datasets(db, offset=5, limit=2)
    assert [dataset.name for dataset in datasets] == [
        "dataset4",
        "dataset3",
    ]  # newest items are returned first
    assert headers == {"content-range": "items 5-6/10"}

    datasets, headers = core.get_paginated_datasets(db, offset=2, limit=7)
    assert [dataset.name for dataset in datasets] == [
        f"dataset{i}" for i in range(7, 0, -1)
    ]
    assert headers == {"content-range": "items 2-8/10"}

    # test that we can reconstitute the full set using paginated calls
    first, header = core.get_paginated_datasets(db, offset=1, limit=2)
    assert len(first) == 2
    assert header == {"content-range": "items 1-2/10"}

    second, header = core.get_paginated_datasets(db, offset=0, limit=1)
    assert len(second) == 1
    assert header == {"content-range": "items 0-0/10"}

    third, header = core.get_paginated_datasets(db, offset=3, limit=20)
    assert len(third) == 7
    assert header == {"content-range": "items 3-9/10"}

    combined = [entry.name for entry in first + second + third]

    assert set(combined) == set([f"dataset{i}" for i in range(0, 10)])


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
            model_names=[created_model],
            datum_filter=schemas.Filter(dataset_names=[created_dataset]),
            parameters=schemas.EvaluationParameters(
                task_type=enums.TaskType.CLASSIFICATION,
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
        [
            enums.TaskType.OBJECT_DETECTION.value,
            enums.TaskType.CLASSIFICATION.value,
        ]
    )


def test_get_unique_datum_metadata_in_dataset(
    db: Session, dataset_name: str, dataset_model_create
):
    def _get_width(dct):
        return dct["width"]

    unique_metadata = core.get_unique_datum_metadata_in_dataset(
        db=db, name=dataset_name
    )
    unique_metadata.sort(key=_get_width)
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
