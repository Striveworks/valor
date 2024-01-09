from sqlalchemy.orm import Session

from velour_api import backend, enums


def test_get_n_datums_in_dataset(
    db: Session, dataset_name: str, dataset_model_create
):
    assert backend.get_n_datums_in_dataset(db=db, name=dataset_name) == 2


def test_get_n_groundtruth_annotations(
    db: Session, dataset_name: str, dataset_model_create
):
    assert backend.get_n_groundtruth_annotations(db=db, name=dataset_name) == 6


def test_get_n_groundtruth_bounding_boxes_in_dataset(
    db: Session, dataset_name: str, dataset_model_create
):
    assert (
        backend.get_n_groundtruth_bounding_boxes_in_dataset(
            db=db, name=dataset_name
        )
        == 3
    )


def test_get_n_groundtruth_polygons_in_dataset(
    db: Session, dataset_name: str, dataset_model_create
):
    assert (
        backend.get_n_groundtruth_polygons_in_dataset(db=db, name=dataset_name)
        == 1
    )


def test_get_n_groundtruth_multipolygons_in_dataset(
    db: Session, dataset_name: str, dataset_model_create
):
    assert (
        backend.get_n_groundtruth_multipolygons_in_dataset(
            db=db, name=dataset_name
        )
        == 0
    )


def test_get_n_groundtruth_rasters_in_dataset(
    db: Session, dataset_name: str, dataset_model_create
):
    assert (
        backend.get_n_groundtruth_rasters_in_dataset(db=db, name=dataset_name)
        == 1
    )


def test_get_unique_task_types_in_dataset(
    db: Session, dataset_name: str, dataset_model_create
):
    assert set(
        backend.get_unique_task_types_in_dataset(db=db, name=dataset_name)
    ) == set(
        [enums.TaskType.DETECTION.value, enums.TaskType.CLASSIFICATION.value]
    )


def test_get_unique_datum_metadata_in_dataset(
    db: Session, dataset_name: str, dataset_model_create
):
    unique_metadata = backend.get_unique_datum_metadata_in_dataset(
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
        backend.get_unique_groundtruth_annotation_metadata_in_dataset(
            db=db, name=dataset_name
        )
    )

    assert len(unique_metadata) == 2
    assert {"int_key": 1} in unique_metadata
    assert {"string_key": "string_val", "int_key": 1} in unique_metadata
