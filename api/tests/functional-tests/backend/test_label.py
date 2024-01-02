import pytest
from sqlalchemy.orm import Session

from velour_api import schemas, enums
from velour_api.crud import create_dataset, create_model, create_groundtruth, create_prediction
from velour_api.backend import models
from velour_api.backend.core.label import (
    fetch_matching_labels,
    create_labels,
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
    create_dataset(
        db=db,
        dataset=schemas.Dataset(name=dataset_name)
    )
    create_model(
        db=db,
        model=schemas.Model(name=model_name)
    )
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
                    ]
                )
            ]
        )
    )
    create_prediction(
        db=db,
        prediction=schemas.Prediction(
            datum=schemas.Datum(uid="123", dataset=dataset_name),
            annotations=[
                schemas.Annotation(
                    task_type=enums.TaskType.CLASSIFICATION,
                    labels=[
                        schemas.Label(key="k1", value="v2"),
                        schemas.Label(key="k1", value="v3"),
                        schemas.Label(key="k3", value="v3")
                    ]
                )
            ]
        )
    )


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
    create_labels(db, labels)
    assert len(db.query(models.schemas.Label).all()) == 1


def test_get_labels(
    db: Session,
    create_dataset_model,
):
    assert len()
