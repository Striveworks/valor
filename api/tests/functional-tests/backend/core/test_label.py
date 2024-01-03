import pytest
from sqlalchemy.orm import Session

from velour_api import enums, schemas
from velour_api.backend import models
from velour_api.backend.core.label import (
    create_labels,
    fetch_label,
    fetch_matching_labels,
    get_disjoint_keys,
    get_disjoint_labels,
    get_joint_keys,
    get_joint_labels,
    get_label_keys,
    get_labels,
)
from velour_api.crud import (
    create_dataset,
    create_groundtruth,
    create_model,
    create_prediction,
)


@pytest.fixture
def simple_labels(db: Session) -> list[schemas.Label]:
    labels = [
        schemas.Label(key="animal", value="dog"),
        schemas.Label(key="animal", value="cat"),
    ]
    create_labels(db, labels)
    return labels


@pytest.fixture
def labels_with_common_values(db: Session) -> list[schemas.Label]:
    labels = [
        schemas.Label(key="stoplight_color", value="red"),
        schemas.Label(key="stoplight_color", value="green"),
        schemas.Label(key="stoplight_color", value="yellow"),
        schemas.Label(key="car_color", value="red"),
        schemas.Label(key="car_color", value="green"),
        schemas.Label(key="car_color", value="blue"),
    ]
    create_labels(db, labels)
    return labels


@pytest.fixture
def create_dataset_model(db: Session, dataset_name: str, model_name: str):
    create_dataset(db=db, dataset=schemas.Dataset(name=dataset_name))
    create_model(db=db, model=schemas.Model(name=model_name))
    create_groundtruth(
        db=db,
        groundtruth=schemas.GroundTruth(
            datum=schemas.Datum(uid="123", dataset=dataset_name),
            annotations=[
                schemas.Annotation(
                    task_type=enums.TaskType.CLASSIFICATION,
                    labels=[
                        schemas.Label(key="k1", value="v1"),
                        schemas.Label(key="k1", value="v2"),
                        schemas.Label(key="k2", value="v3"),
                    ],
                )
            ],
        ),
    )
    create_prediction(
        db=db,
        prediction=schemas.Prediction(
            model=model_name,
            datum=schemas.Datum(uid="123", dataset=dataset_name),
            annotations=[
                schemas.Annotation(
                    task_type=enums.TaskType.CLASSIFICATION,
                    labels=[
                        schemas.Label(key="k1", value="v2", score=0.1),
                        schemas.Label(key="k1", value="v3", score=0.9),
                        schemas.Label(key="k3", value="v3", score=1.0),
                    ],
                )
            ],
        ),
    )


def test_fetch_label(db: Session, simple_labels: list[schemas.Label]):
    for label in simple_labels:
        fetched_label = fetch_label(db, label)
        assert fetched_label.key == label.key
        assert fetched_label.value == label.value

    # fetch label that doesnt exist
    assert fetch_label(db, schemas.Label(key="k1234", value="v1234")) is None


def test_fetch_matching_labels(db: Session, simple_labels):
    # check all exist
    db_labels = {
        (label.key, label.value)
        for label in fetch_matching_labels(db, simple_labels)
    }
    og_labels = {(label.key, label.value) for label in simple_labels}
    assert db_labels == og_labels

    # search for specific label (animal, dog)
    search = [schemas.Label(key="animal", value="dog")]
    labels = fetch_matching_labels(db, search)
    assert len(labels) == 1
    assert labels[0].key == "animal"
    assert labels[0].value == "dog"

    # search for label that doesnt exist
    search = [schemas.Label(key="animal", value="elephant")]
    assert len(fetch_matching_labels(db, search)) == 0


def test_fetch_matching_labels_from_labels_with_common_values(
    db: Session,
    labels_with_common_values,
):
    # check all exist
    db_labels = {
        (label.key, label.value)
        for label in fetch_matching_labels(db, labels_with_common_values)
    }
    og_labels = {
        (label.key, label.value) for label in labels_with_common_values
    }
    assert db_labels == og_labels

    # search for specific label (car_color, red)
    search = [schemas.Label(key="car_color", value="red")]
    labels = fetch_matching_labels(db, search)
    assert len(labels) == 1
    assert labels[0].key == "car_color"
    assert labels[0].value == "red"

    # search for label that doesnt exist
    search = [schemas.Label(key="animal", value="elephant")]
    assert len(fetch_matching_labels(db, search)) == 0

    # search for labels (car_color, red) and (stoplight_color, green)
    # validates that key-value pairings are respected.
    search = [
        schemas.Label(key="car_color", value="red"),
        schemas.Label(key="stoplight_color", value="green"),
    ]
    labels = fetch_matching_labels(db, search)
    assert {("car_color", "red"), ("stoplight_color", "green")} == {
        (label.key, label.value) for label in labels
    }
    assert len(labels) == 2


def test_create_labels_with_duplicates(db: Session):
    labels = [
        schemas.Label(key="stoplight_color", value="red"),
        schemas.Label(key="stoplight_color", value="red"),
    ]
    created_labels = create_labels(db, labels)
    assert len(db.query(models.Label).all()) == 1
    assert len(created_labels) == 2
    assert created_labels[0].id == created_labels[1].id


def test_get_labels(
    db: Session,
    create_dataset_model,
):
    assert len(db.query(models.Label).all()) == 5

    labels = get_labels(db)
    assert len(labels) == 5
    assert set(labels) == {
        schemas.Label(key="k1", value="v1"),
        schemas.Label(key="k1", value="v2"),
        schemas.Label(key="k2", value="v3"),
        schemas.Label(key="k1", value="v2"),
        schemas.Label(key="k1", value="v3"),
        schemas.Label(key="k3", value="v3"),
    }

    pred_labels = get_labels(db, ignore_groundtruths=True)
    assert len(pred_labels) == 3
    assert set(pred_labels) == {
        schemas.Label(key="k1", value="v2"),
        schemas.Label(key="k1", value="v3"),
        schemas.Label(key="k3", value="v3"),
    }

    gt_labels = get_labels(db, ignore_predictions=True)
    assert len(gt_labels) == 3
    assert set(gt_labels) == {
        schemas.Label(key="k1", value="v1"),
        schemas.Label(key="k1", value="v2"),
        schemas.Label(key="k2", value="v3"),
    }


def test_get_labels_filtered(
    db: Session,
    create_dataset_model,
):
    assert len(db.query(models.Label).all()) == 5

    filters = schemas.Filter(label_keys=["k1"])

    labels = get_labels(db, filters=filters)
    assert len(labels) == 3
    assert set(labels) == {
        schemas.Label(key="k1", value="v1"),
        schemas.Label(key="k1", value="v2"),
        schemas.Label(key="k1", value="v3"),
    }

    pred_labels = get_labels(db, filters=filters, ignore_groundtruths=True)
    assert len(pred_labels) == 2
    assert set(pred_labels) == {
        schemas.Label(key="k1", value="v2"),
        schemas.Label(key="k1", value="v3"),
    }

    gt_labels = get_labels(db, filters=filters, ignore_predictions=True)
    assert len(gt_labels) == 2
    assert set(gt_labels) == {
        schemas.Label(key="k1", value="v1"),
        schemas.Label(key="k1", value="v2"),
    }


def test_get_label_keys(
    db: Session,
    create_dataset_model,
):
    assert len(db.query(models.Label).all()) == 5

    labels = get_label_keys(db)
    assert len(labels) == 3
    assert set(labels) == {"k1", "k2", "k3"}

    pred_labels = get_label_keys(db, ignore_groundtruths=True)
    assert len(pred_labels) == 2
    assert set(pred_labels) == {"k1", "k3"}

    gt_labels = get_label_keys(db, ignore_predictions=True)
    assert len(gt_labels) == 2
    assert set(gt_labels) == {"k1", "k2"}


def test_get_label_keys_filtered(
    db: Session,
    create_dataset_model,
):
    assert len(db.query(models.Label).all()) == 5

    filters = schemas.Filter(label_keys=["k1"])

    labels = get_label_keys(db, filters=filters)
    assert len(labels) == 1
    assert set(labels) == {"k1"}

    pred_labels = get_label_keys(db, filters=filters, ignore_groundtruths=True)
    assert len(pred_labels) == 1
    assert set(pred_labels) == {"k1"}

    gt_labels = get_label_keys(db, filters=filters, ignore_predictions=True)
    assert len(gt_labels) == 1
    assert set(gt_labels) == {"k1"}


def test_get_joint_labels(
    db: Session,
    dataset_name: str,
    model_name: str,
    create_dataset_model,
):
    labels = get_joint_labels(
        db=db,
        dataset_name=dataset_name,
        model_name=model_name,
        task_types=[enums.TaskType.CLASSIFICATION],
        groundtruth_type=enums.AnnotationType.NONE,
        prediction_type=enums.AnnotationType.NONE,
    )
    assert len(labels) == 1
    assert labels[0] == schemas.Label(
        key="k1",
        value="v2",
    )


def test_get_joint_keys(
    db: Session,
    dataset_name: str,
    model_name: str,
    create_dataset_model,
):
    keys = get_joint_keys(
        db=db,
        dataset_name=dataset_name,
        model_name=model_name,
        task_type=enums.TaskType.CLASSIFICATION,
    )
    assert len(keys) == 1
    assert set(keys) == {"k1"}


def test_get_disjoint_labels(
    db: Session,
    dataset_name: str,
    model_name: str,
    create_dataset_model,
):
    ds_unique, md_unique = get_disjoint_labels(
        db=db,
        dataset_name=dataset_name,
        model_name=model_name,
        task_types=[enums.TaskType.CLASSIFICATION],
        groundtruth_type=enums.AnnotationType.NONE,
        prediction_type=enums.AnnotationType.NONE,
    )
    assert len(ds_unique) == 2
    assert set(ds_unique) == {
        schemas.Label(key="k1", value="v1"),
        schemas.Label(key="k2", value="v3"),
    }

    assert len(md_unique) == 2
    assert set(md_unique) == {
        schemas.Label(key="k1", value="v3"),
        schemas.Label(key="k3", value="v3"),
    }


def test_get_disjoint_keys(
    db: Session,
    dataset_name: str,
    model_name: str,
    create_dataset_model,
):
    ds_unique, md_unique = get_disjoint_keys(
        db=db,
        dataset_name=dataset_name,
        model_name=model_name,
        task_type=enums.TaskType.CLASSIFICATION,
    )
    assert len(ds_unique) == 1
    assert set(ds_unique) == {"k2"}

    assert len(md_unique) == 1
    assert set(md_unique) == {"k3"}
