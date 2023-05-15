from unittest.mock import MagicMock, PropertyMock, patch

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from velour_api import database, exceptions
from velour_api.schemas import Dataset, DatumTypes, Model


@pytest.fixture
def client() -> TestClient:
    database.create_db = MagicMock()
    database.make_session = MagicMock()
    from velour_api import main

    main.get_db = MagicMock()

    return TestClient(main.app)


def test_protected_routes(client: TestClient):
    """Check that all routes are protected"""
    routes = [r for r in client.app.routes if isinstance(r, APIRoute)]
    with patch(
        "velour_api.settings.AuthConfig.no_auth",
        new_callable=PropertyMock(return_value=False),
    ):
        for r in routes:
            for m in r.methods:
                resp = getattr(client, m.lower())(r.path)
                assert resp.status_code == 403, f"{r}, {m}"


@patch("velour_api.main.crud")
def _test_post_endpoints(
    crud,
    client: TestClient,
    endpoint: str,
    crud_method_name: str,
    example_json: dict,
    expected_status_code=200,
    endpoint_only_has_post=True,
):
    crud_method = getattr(crud, crud_method_name)
    # have mock method return empty list (type hint in main is satisfied)
    crud_method.return_value = []
    resp = client.post(endpoint, json=example_json)
    assert resp.status_code == expected_status_code

    crud_method.assert_called_once()

    # now send a bad payload and make sure we get a 422
    resp = client.post(endpoint, json={})
    assert resp.status_code == 422

    # send an invalid method and make sure we get a 405
    if endpoint_only_has_post:
        resp = client.get(endpoint)
        assert resp.status_code == 405


def test_post_groundtruth_detections(client: TestClient):
    example_json = {"dataset_name": "", "detections": []}
    _test_post_endpoints(
        client=client,
        endpoint="/groundtruth-detections",
        crud_method_name="create_groundtruth_detections",
        example_json=example_json,
    )

    # check we get a conflict (409) if the dataset is finalized
    with patch(
        "velour_api.main.crud.create_groundtruth_detections",
        side_effect=exceptions.DatasetIsFinalizedError("dsetname"),
    ):
        resp = client.post("/groundtruth-detections", json=example_json)
        assert resp.status_code == 409


def test_post_predicted_detections(client: TestClient):
    example_json = {"model_name": "", "dataset_name": "", "detections": []}
    _test_post_endpoints(
        client=client,
        endpoint="/predicted-detections",
        crud_method_name="create_predicted_detections",
        example_json=example_json,
    )

    # check we get a 404 if an image does not exist
    with patch(
        "velour_api.main.crud.create_predicted_detections",
        side_effect=exceptions.ImageDoesNotExistError("", ""),
    ):
        resp = client.post("/predicted-detections", json=example_json)
        assert resp.status_code == 404


def test_post_groundtruth_segmentations(client: TestClient):
    example_json = {"dataset_name": "", "segmentations": []}
    _test_post_endpoints(
        client=client,
        endpoint="/groundtruth-segmentations",
        crud_method_name="create_groundtruth_segmentations",
        example_json=example_json,
    )

    # check we get a conflict (409) if the dataset is finalized
    with patch(
        "velour_api.main.crud.create_groundtruth_segmentations",
        side_effect=exceptions.DatasetIsFinalizedError("dsetname"),
    ):
        resp = client.post("/groundtruth-segmentations", json=example_json)
        assert resp.status_code == 409


def test_post_predicted_segmentations(client: TestClient):
    example_json = {"model_name": "", "dataset_name": "", "segmentations": []}
    _test_post_endpoints(
        client=client,
        endpoint="/predicted-segmentations",
        crud_method_name="create_predicted_segmentations",
        example_json=example_json,
    )

    # check we get a 404 if an image does not exist
    with patch(
        "velour_api.main.crud.create_predicted_segmentations",
        side_effect=exceptions.ImageDoesNotExistError("", ""),
    ):
        resp = client.post("/predicted-segmentations", json=example_json)
        assert resp.status_code == 404


def test_post_groundtruth_classifications(client: TestClient):
    example_json = {"dataset_name": "", "classifications": []}
    _test_post_endpoints(
        client=client,
        endpoint="/groundtruth-classifications",
        crud_method_name="create_ground_truth_image_classifications",
        example_json=example_json,
    )

    # check we get a conflict (409) if the dataset is finalized
    with patch(
        "velour_api.main.crud.create_ground_truth_image_classifications",
        side_effect=exceptions.DatasetIsFinalizedError("dsetname"),
    ):
        resp = client.post("/groundtruth-classifications", json=example_json)
        assert resp.status_code == 409


def test_post_predicted_classifications(client: TestClient):
    example_json = {
        "model_name": "",
        "dataset_name": "",
        "classifications": [],
    }
    _test_post_endpoints(
        client=client,
        endpoint="/predicted-classifications",
        crud_method_name="create_predicted_image_classifications",
        example_json=example_json,
    )

    # check we get a 404 if an image does not exist
    with patch(
        "velour_api.main.crud.create_predicted_image_classifications",
        side_effect=exceptions.ImageDoesNotExistError("", ""),
    ):
        resp = client.post("/predicted-classifications", json=example_json)
        assert resp.status_code == 404


def test_post_datasets(client: TestClient):
    example_json = {"name": "", "type": DatumTypes.IMAGE.value}
    _test_post_endpoints(
        client=client,
        endpoint="/datasets",
        crud_method_name="create_dataset",
        example_json=example_json,
        expected_status_code=201,
        endpoint_only_has_post=False,
    )

    with patch(
        "velour_api.main.crud.create_dataset",
        side_effect=exceptions.DatasetAlreadyExistsError(""),
    ):
        resp = client.post("/datasets", json=example_json)
        assert resp.status_code == 409


def test_post_models(client: TestClient):
    example_json = {"name": ""}
    _test_post_endpoints(
        client=client,
        endpoint="/models",
        crud_method_name="create_model",
        example_json=example_json,
        expected_status_code=201,
        endpoint_only_has_post=False,
    )

    with patch(
        "velour_api.main.crud.create_model",
        side_effect=exceptions.ModelAlreadyExistsError(""),
    ):
        resp = client.post("/models", json=example_json)
        assert resp.status_code == 409


@patch("velour_api.main.crud")
def test_get_datasets(crud, client: TestClient):
    crud.get_datasets.return_value = []
    resp = client.get("/datasets")
    assert resp.status_code == 200
    crud.get_datasets.assert_called_once()


@patch("velour_api.main.crud")
def test_get_dataset_by_name(crud, client: TestClient):
    crud.get_dataset.return_value = Dataset(
        name="", draft=True, type=DatumTypes.TABULAR
    )

    resp = client.get("/datasets/dsetname")
    assert resp.status_code == 200
    crud.get_dataset.assert_called_once()

    with patch(
        "velour_api.main.crud.get_dataset",
        side_effect=exceptions.DatasetDoesNotExistError(""),
    ):
        resp = client.get("/datasets/dsetname")
        assert resp.status_code == 404

    resp = client.post("/datasets/dsetname")
    assert resp.status_code == 405


@patch("velour_api.main.crud")
def test_get_model_by_name(crud, client: TestClient):
    crud.get_model.return_value = Model(name="")
    resp = client.get("/models/modelname")
    assert resp.status_code == 200
    crud.get_model.assert_called_once()

    with patch(
        "velour_api.main.crud.get_model",
        side_effect=exceptions.ModelDoesNotExistError(""),
    ):
        resp = client.get("/models/modelname")
        assert resp.status_code == 404

    resp = client.post("/models/modelname")
    assert resp.status_code == 405


@patch("velour_api.main.crud")
def test_finalize_datasets(crud, client: TestClient):
    resp = client.put("/datasets/dsetname/finalize")
    assert resp.status_code == 200
    crud.finalize_dataset.assert_called_once()

    with patch(
        "velour_api.main.crud.finalize_dataset",
        side_effect=exceptions.DatasetDoesNotExistError(""),
    ):
        resp = client.put("datasets/dsetname/finalize")
        assert resp.status_code == 404

    resp = client.get("/datasets/dsetname/finalize")
    assert resp.status_code == 405


@patch("velour_api.main.crud")
@patch("velour_api.main.schemas")
def test_get_dataset_labels(schemas, crud, client: TestClient):
    resp = client.get("/datasets/dsetname/labels")
    assert resp.status_code == 200
    crud.get_all_labels_in_dataset.assert_called_once()

    with patch(
        "velour_api.main.crud.get_all_labels_in_dataset",
        side_effect=exceptions.DatasetDoesNotExistError(""),
    ):
        resp = client.get("datasets/dsetname/labels")
        assert resp.status_code == 404

    resp = client.post("/datasets/dsetname/labels")
    assert resp.status_code == 405


@patch("velour_api.main.crud")
@patch("velour_api.main.schemas")
def test_get_dataset_images(schemas, crud, client: TestClient):
    resp = client.get("/datasets/dsetname/images")
    assert resp.status_code == 200
    crud.get_datums_in_dataset.assert_called_once()

    with patch(
        "velour_api.main.crud.get_datums_in_dataset",
        side_effect=exceptions.DatasetDoesNotExistError(""),
    ):
        resp = client.get("datasets/dsetname/images")
        assert resp.status_code == 404

    resp = client.post("/datasets/dsetname/images")
    assert resp.status_code == 405


@patch("velour_api.main.crud")
def test_delete_dataset(crud, client: TestClient):
    resp = client.delete("/datasets/dsetname")
    assert resp.status_code == 200
    crud.delete_dataset.assert_called_once()


@patch("velour_api.main.crud")
def test_get_models(crud, client: TestClient):
    crud.get_models.return_value = []
    resp = client.get("/models")
    assert resp.status_code == 200
    crud.get_models.assert_called_once()


@patch("velour_api.main.crud")
def test_delete_model(crud, client: TestClient):
    resp = client.delete("/models/modelname")
    assert resp.status_code == 200
    crud.delete_model.assert_called_once()


@patch("velour_api.main.crud")
def test_get_labels(crud, client: TestClient):
    crud.get_all_labels.return_value = []
    resp = client.get("/labels")
    assert resp.status_code == 200
    crud.get_all_labels.assert_called_once()

    resp = client.post("/labels")
    assert resp.status_code == 405


def test_user(client: TestClient):
    resp = client.get("/user")
    assert resp.json() == {"email": None}


@patch("velour_api.main.crud")
@patch("velour_api.main.schemas")
def test_get_model_metrics(schemas, crud, client: TestClient):
    crud.get_model_metrics.return_value = []
    resp = client.get("/models/modelname/metrics")
    assert resp.status_code == 200
    crud.get_model_metrics.assert_called_once()

    with patch(
        "velour_api.main.crud.get_model_metrics",
        side_effect=exceptions.ModelDoesNotExistError(""),
    ):
        resp = client.get("models/modelname/metrics")
        assert resp.status_code == 404

    resp = client.post("/models/modelname/metrics")
    assert resp.status_code == 405
