""" These integration tests should be run with a back end at http://localhost:8000
that is no auth
"""

from typing import List

import pytest

from client.valor import (
    Annotation,
    Client,
    Dataset,
    Datum,
    GroundTruth,
    Label,
    Model,
    Prediction,
)
from client.valor.client import connect
from client.valor.enums import TaskType
from client.valor.exceptions import ClientException
from client.valor.schemas import Constraint, Filter


@pytest.fixture
def dataset_labels() -> List[Label]:
    return [Label(key=f"class{i//2}", value=str(i)) for i in range(10)]


@pytest.fixture
def model_labels() -> List[Label]:
    return [
        Label(key=f"class{i//2}", value=str(i), score=0.9 if i % 2 else 0.1)
        for i in range(10)
    ]


@pytest.fixture
def created_dataset(
    client: Client,
    dataset_name: str,
    dataset_labels: List[Label],
) -> Dataset:
    dataset = Dataset.create(name=dataset_name)
    dataset.add_groundtruth(
        groundtruth=GroundTruth(
            datum=Datum(uid="1"),
            annotations=[
                Annotation(
                    task_type=TaskType.CLASSIFICATION,
                    labels=dataset_labels,
                )
            ],
        )
    )
    dataset.finalize()
    return dataset


@pytest.fixture
def created_model(
    client: Client,
    model_name: str,
    model_labels: List[Label],
    created_dataset: Dataset,
) -> Model:
    model = Model.create(name=model_name)
    model.add_prediction(
        dataset=created_dataset,
        prediction=Prediction(
            datum=Datum(uid="1"),
            annotations=[
                Annotation(
                    task_type=TaskType.CLASSIFICATION,
                    labels=model_labels,
                )
            ],
        ),
    )
    return model


def test_connect():
    bad_url = "localhost:8000"
    with pytest.raises(ValueError):
        connect(host=bad_url, reconnect=True)

    bad_url2 = "http://localhost:8111"
    with pytest.raises(Exception):
        connect(host=bad_url2, reconnect=True)

    good_url = "http://localhost:8000"
    connect(host=good_url, reconnect=True)


def test_version_mismatch_warning(caplog):
    # test client being older than api
    Client().conn._validate_version(
        client_version="1.1.1", api_version="9.9.9"
    )

    assert all(
        record.levelname == "WARNING" and "older" in record.message
        for record in caplog.records
    )

    caplog.clear()

    # test client being newer than api
    Client().conn._validate_version(
        client_version="9.9.9", api_version="1.1.1"
    )

    assert all(
        record.levelname == "WARNING" and "newer" in record.message
        for record in caplog.records
    )

    caplog.clear()

    # test client and API being the same version
    Client().conn._validate_version(
        client_version="1.1.1", api_version="1.1.1"
    )

    assert all(
        record.levelname == "DEBUG"
        and "matches client version" in record.message
        for record in caplog.records
    )
    caplog.clear()

    # test missing client or API versions
    Client().conn._validate_version(client_version=None, api_version="1.1.1")  # type: ignore - purposefully throwing error

    assert all(
        record.levelname == "WARNING"
        and "client isn't versioned" in record.message
        for record in caplog.records
    )
    caplog.clear()

    Client().conn._validate_version(client_version="1.1.1", api_version=None)  # type: ignore - purposefully throwing error

    assert all(
        record.levelname == "WARNING"
        and "API didn't return a version" in record.message
        for record in caplog.records
    )

    caplog.clear()

    # test that semantic versioning works correctly
    # client_version > api_version when comparing strings, but
    # client_version < api_version when comparing semantic versions
    Client().conn._validate_version(
        client_version="1.12.2", api_version="1.101.12"
    )

    assert all(
        record.levelname == "WARNING" and "older" in record.message
        for record in caplog.records
    )
    caplog.clear()


def test__requests_wrapper(client: Client):
    with pytest.raises(ValueError):
        client.conn._requests_wrapper("get", "/datasets/fake_dataset/status")

    with pytest.raises(ValueError):
        client.conn._requests_wrapper(
            "bad_method", "datasets/fake_dataset/status"
        )

    with pytest.raises(ClientException):
        client.conn._requests_wrapper("get", "not_an_endpoint")


def test_get_labels(
    client: Client,
    created_dataset: Dataset,
    created_model: Model,
    dataset_labels: List[Label],
    model_labels: List[Label],
):
    all_labels = client.get_labels()
    assert len(all_labels) == 10

    high_score_labels = client.get_labels(
        Filter(label_scores=[Constraint(value=0.5, operator=">")])
    )
    assert len(high_score_labels) == 5
    for label in high_score_labels:
        assert int(label.value) % 2 == 1

    low_score_labels = client.get_labels(
        Filter(label_scores=[Constraint(value=0.5, operator="<")])
    )
    assert len(low_score_labels) == 5
    for label in low_score_labels:
        assert int(label.value) % 2 == 0


def test_get_datasets(
    client: Client,
    created_dataset: Dataset,
    created_model: Model,
    dataset_labels: List[Label],
    model_labels: List[Label],
):
    all_datasets = client.get_datasets()
    assert len(all_datasets) == 1
    assert all_datasets[0].name == created_dataset.name

    pos_query = client.get_datasets(Filter(labels=[{"class0": "1"}]))
    assert len(pos_query) == 1
    assert pos_query[0].name == created_dataset.name

    neg_query = client.get_datasets(Filter(labels=[{"some_other_class": "1"}]))
    assert len(neg_query) == 0


def test_get_models(
    client: Client,
    created_dataset: Dataset,
    created_model: Model,
    dataset_labels: List[Label],
    model_labels: List[Label],
):
    all_models = client.get_models()
    assert len(all_models) == 1
    assert all_models[0].name == created_model.name

    pos_query = client.get_models(Filter(labels=[{"class0": "1"}]))
    assert len(pos_query) == 1
    assert pos_query[0].name == created_model.name

    neg_query = client.get_models(Filter(labels=[{"some_other_class": "1"}]))
    assert len(neg_query) == 0


def test_get_datums(
    client: Client,
    created_dataset: Dataset,
    created_model: Model,
    dataset_labels: List[Label],
    model_labels: List[Label],
):
    all_datums = client.get_datums()
    assert len(all_datums) == 1
    assert all_datums[0].uid == "1"

    pos_query = client.get_datums(Filter(labels=[{"class0": "1"}]))
    assert len(pos_query) == 1
    assert pos_query[0].uid == "1"

    neg_query = client.get_datums(Filter(labels=[{"some_other_class": "1"}]))
    assert len(neg_query) == 0
