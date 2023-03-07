from unittest.mock import MagicMock, PropertyMock, patch

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from velour_api import database


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
    resp = client.post(endpoint, json=example_json)
    assert resp.status_code == expected_status_code
    getattr(crud, crud_method_name).assert_called_once()

    # now send a bad payload and make sure we get a 422
    resp = client.post(endpoint, json={})
    assert resp.status_code == 422

    # send an invalid method and make sure we get a 405
    if endpoint_only_has_post:
        resp = client.get(endpoint)
        assert resp.status_code == 405


def test_post_groundtruth_detections(client: TestClient):
    _test_post_endpoints(
        client=client,
        endpoint="/groundtruth-detections",
        crud_method_name="create_groundtruth_detections",
        example_json={"dataset_name": "", "detections": []},
    )


def test_post_predicted_detections(client: TestClient):
    _test_post_endpoints(
        client=client,
        endpoint="/predicted-detections",
        crud_method_name="create_predicted_detections",
        example_json={"model_name": "", "detections": []},
    )


def test_post_groundtruth_segmentations(client: TestClient):
    _test_post_endpoints(
        client=client,
        endpoint="/groundtruth-segmentations",
        crud_method_name="create_groundtruth_segmentations",
        example_json={"dataset_name": "", "segmentations": []},
    )


def test_post_predicted_segmentations(client: TestClient):
    _test_post_endpoints(
        client=client,
        endpoint="/predicted-segmentations",
        crud_method_name="create_predicted_segmentations",
        example_json={"model_name": "", "segmentations": []},
    )


def test_post_groundtruth_classifications(client: TestClient):
    _test_post_endpoints(
        client=client,
        endpoint="/groundtruth-classifications",
        crud_method_name="create_ground_truth_image_classifications",
        example_json={"dataset_name": "", "classifications": []},
    )


def test_post_predicted_classifications(client: TestClient):
    _test_post_endpoints(
        client=client,
        endpoint="/predicted-classifications",
        crud_method_name="create_predicted_image_classifications",
        example_json={"model_name": "", "classifications": []},
    )


def test_post_datasets(client: TestClient):
    _test_post_endpoints(
        client=client,
        endpoint="/datasets",
        crud_method_name="create_dataset",
        example_json={"name": ""},
        expected_status_code=201,
        endpoint_only_has_post=False,
    )


def test_post_models(client: TestClient):
    _test_post_endpoints(
        client=client,
        endpoint="/models",
        crud_method_name="create_model",
        example_json={"name": ""},
        expected_status_code=201,
        endpoint_only_has_post=False,
    )


@patch("velour_api.main.crud")
def test_get_datasets(crud, client: TestClient):
    resp = client.get("/datasets")
    assert resp.status_code == 200
    crud.get_datasets.assert_called_once()


@patch("velour_api.main.crud")
@patch("velour_api.main.schemas")
def test_get_dataset_by_name(schemas, crud, client: TestClient):
    resp = client.get("/datasets/dsetname")
    assert resp.status_code == 200
    crud.get_dataset.assert_called_once()
    schemas.Dataset.assert_called_once()

    resp = client.post("/datasets/dsetname")
    assert resp.status_code == 405


@patch("velour_api.main.crud")
def test_finalize_datasets(crud, client: TestClient):
    resp = client.put("/datasets/dsetname/finalize")
    assert resp.status_code == 200
    crud.finalize_dataset.assert_called_once()

    resp = client.get("/datasets/dsetname/finalize")
    assert resp.status_code == 405


@patch("velour_api.main.crud")
@patch("velour_api.main.schemas")
def test_get_dataset_labels(schemas, crud, client: TestClient):
    resp = client.get("/datasets/dsetname/labels")
    assert resp.status_code == 200
    crud.get_labels_in_dataset.assert_called_once()

    resp = client.post("/datasets/dsetname/labels")
    assert resp.status_code == 405


@patch("velour_api.main.crud")
@patch("velour_api.main.schemas")
def test_get_dataset_images(schemas, crud, client: TestClient):
    resp = client.get("/datasets/dsetname/images")
    assert resp.status_code == 200
    crud.get_images_in_dataset.assert_called_once()

    resp = client.post("/datasets/dsetname/images")
    assert resp.status_code == 405


@patch("velour_api.main.crud")
def test_delete_dataset(crud, client: TestClient):
    resp = client.delete("/datasets/dsetname")
    assert resp.status_code == 200
    crud.delete_dataset.assert_called_once()


@patch("velour_api.main.crud")
def test_get_models(crud, client: TestClient):
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
    resp = client.get("/labels")
    assert resp.status_code == 200
    crud.get_all_labels.assert_called_once()

    resp = client.post("/labels")
    assert resp.status_code == 405


def test_user(client: TestClient):
    resp = client.get("/user")
    assert resp.json() == {"email": None}
