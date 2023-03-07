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
    crud, client: TestClient, endpoint: str, example_json: dict
):
    pass


@patch("velour_api.main.crud")
def test_post_groundtruth_detections(crud, client: TestClient):
    resp = client.post(
        "/groundtruth-detections", json={"dataset_name": "", "detections": []}
    )
    assert resp.status_code == 200
    crud.create_groundtruth_detections.assert_called_once()

    # now send a bad payload and make sure we get a 422
    resp = client.post("/groundtruth-detections", json={"detections": []})
    assert resp.status_code == 422

    # send an invalid method and make sure we get a 405
    resp = client.get("/groundtruth-detections")
    assert resp.status_code == 405


@patch("velour_api.main.crud")
def test_post_predicted_detections(crud, client: TestClient):
    resp = client.post(
        "/predicted-detections", json={"model_name": "", "detections": []}
    )
    assert resp.status_code == 200
    crud.create_predicted_detections.assert_called_once()

    # now send a bad payload and make sure we get a 422
    resp = client.post("/predicted-detections", json={"detections": []})
    assert resp.status_code == 422

    # send an invalid method and make sure we get a 405
    resp = client.get("/predicted-detections")
    assert resp.status_code == 405


@patch("velour_api.main.crud")
def test_get_datasets(crud, client: TestClient):
    client.get("/datasets")
    crud.get_datasets.assert_called_once()


def test_user(client: TestClient):
    resp = client.get("/user")
    assert resp.json() == {"email": None}
