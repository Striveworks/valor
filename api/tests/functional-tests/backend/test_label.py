import numpy
import pytest
from sqlalchemy import and_
from sqlalchemy.orm import Session
from psycopg2.errors import UniqueViolation

from velour_api.schemas import Label
from velour_api.exceptions import LabelDoesNotExistError
from velour_api.backend import models
from velour_api.backend.core.label import create_labels, _get_existing_labels


@pytest.fixture
def simple_labels(db: Session) -> list[Label]:
    labels = [
        Label(key="animal", value="dog"),
        Label(key="animal", value="cat"),
    ]
    create_labels(db, labels)
    return labels


@pytest.fixture
def labels_with_common_values(db: Session) -> list[Label]:
    labels = [
        Label(key="stoplight_color", value="red"),
        Label(key="stoplight_color", value="green"),
        Label(key="stoplight_color", value="yellow"),
        Label(key="car_color", value="red"),
        Label(key="car_color", value="green"),
        Label(key="car_color", value="blue"),
    ]
    create_labels(db, labels)
    return labels


def test__get_existing_labels(db: Session, simple_labels):
    # check all exist
    db_labels = {
        (label.key, label.value)
        for label in _get_existing_labels(db, simple_labels)
    }
    og_labels = {
        (label.key, label.value)
        for label in simple_labels
    }
    assert db_labels == og_labels

    # search for specific label (animal, dog)
    search = [Label(key="animal", value="dog")]
    labels = _get_existing_labels(db, search)
    assert len(labels) == 1
    assert labels[0].key == "animal"
    assert labels[0].value == "dog"

    # search for label that doesnt exist
    search = [Label(key="animal", value="elephant")]
    with pytest.raises(LabelDoesNotExistError):
        _get_existing_labels(db, search)


def test__get_existing_labels_from_labels_with_common_values(
    db: Session, 
    labels_with_common_values,
):
    # check all exist
    db_labels = {
        (label.key, label.value)
        for label in _get_existing_labels(db, labels_with_common_values)
    }
    og_labels = {
        (label.key, label.value)
        for label in labels_with_common_values
    }
    assert db_labels == og_labels

    # search for specific label (car_color, red)
    search = [Label(key="car_color", value="red")]
    labels = _get_existing_labels(db, search)
    assert len(labels) == 1
    assert labels[0].key == "car_color"
    assert labels[0].value == "red"

    # search for label that doesnt exist
    search = [Label(key="animal", value="elephant")]
    with pytest.raises(LabelDoesNotExistError):
        _get_existing_labels(db, search)

    # search for labels (car_color, red) and (stoplight_color, green)
    # validates that key-value pairings are respected.
    search = [
        Label(key="car_color", value="red"),
        Label(key="stoplight_color", value="green"),
    ]
    labels = _get_existing_labels(db, search)
    assert {("car_color", "red"), ("stoplight_color", "green")} == {
        (label.key, label.value)
        for label in labels
    }
    assert len(labels) == 2


def test_create_labels_with_duplicates(db: Session):
    labels = [
        Label(key="stoplight_color", value="red"),
        Label(key="stoplight_color", value="red"),
    ]
    create_labels(db, labels)