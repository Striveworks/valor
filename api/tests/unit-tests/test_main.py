from unittest.mock import MagicMock, PropertyMock, patch

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from velour_api import exceptions, schemas
from velour_api.backend import database


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


""" POST /groundtruths """


def test_post_groundtruth(client: TestClient):
    example_json = {
        "datum": {
            "uid": "file_uid",
            "dataset": "dataset1",
            "metadata": [],
        },
        "annotations": [
            {
                "labels": [
                    {"key": "k1", "value": "v1"},
                ],
                "task_type": "classification",
                "metadata": [],
            }
        ],
    }

    # check we get a conflict (409) if the dataset is finalized
    with patch(
        "velour_api.main.crud.create_groundtruth",
        side_effect=exceptions.DatasetFinalizedError("dsetname"),
    ):
        resp = client.post("/groundtruths", json=example_json)
        assert resp.status_code == 409


def test_post_groundtruth_classification(client: TestClient):
    example_json = {
        "datum": {
            "uid": "file_uid",
            "dataset": "dataset1",
            "metadata": [
                {"key": "meta1", "value": 0.4},
                {"key": "meta1", "value": "v1"},
            ],
        },
        "annotations": [
            {
                "labels": [
                    {"key": "k1", "value": "v1"},
                    {"key": "k1", "value": "v2"},
                ],
                "task_type": "classification",
                "metadata": [
                    {"key": "meta1", "value": 0.4},
                    {"key": "meta1", "value": "v1"},
                ],
            },
            {
                "labels": [
                    {"key": "k2", "value": "v1"},
                    {"key": "k2", "value": "v2"},
                ],
                "task_type": "classification",
                "metadata": [
                    {"key": "meta2", "value": 0.4},
                    {"key": "meta2", "value": "v1"},
                ],
            },
        ],
    }

    _test_post_endpoints(
        client=client,
        endpoint="/groundtruths",
        crud_method_name="create_groundtruth",
        example_json=example_json,
    )


def test_post_groundtruth_bbox_detection(client: TestClient):
    example_json = {
        "datum": {
            "uid": "file_uid",
            "dataset": "dataset1",
            "metadata": [
                {"key": "meta1", "value": 0.4},
                {"key": "meta1", "value": "v1"},
            ],
        },
        "annotations": [
            {
                "labels": [
                    {"key": "k1", "value": "v1"},
                    {"key": "k1", "value": "v2"},
                ],
                "task_type": "detection",
                "metadata": [
                    {"key": "meta1", "value": 0.4},
                    {"key": "meta1", "value": "v1"},
                ],
                "bounding_box": {
                    "polygon": {
                        "points": [
                            {"x": 0, "y": 0},
                            {"x": 0, "y": 1},
                            {"x": 1, "y": 1},
                            {"x": 1, "y": 0},
                        ]
                    }
                },
            },
        ],
    }
    _test_post_endpoints(
        client=client,
        endpoint="/groundtruths",
        crud_method_name="create_groundtruth",
        example_json=example_json,
    )


def test_post_groundtruth_polygon_detection(client: TestClient):
    example_json = {
        "datum": {
            "uid": "file_uid",
            "dataset": "dataset1",
            "metadata": [
                {"key": "meta1", "value": 0.4},
                {"key": "meta1", "value": "v1"},
            ],
        },
        "annotations": [
            {
                "labels": [
                    {"key": "k1", "value": "v1"},
                    {"key": "k1", "value": "v2"},
                ],
                "task_type": "detection",
                "metadata": [
                    {"key": "meta1", "value": 0.4},
                    {"key": "meta1", "value": "v1"},
                ],
                "polygon": {
                    "boundary": {
                        "points": [
                            {"x": 0, "y": 0},
                            {"x": 0, "y": 10},
                            {"x": 10, "y": 10},
                            {"x": 10, "y": 0},
                        ]
                    },
                    "holes": [
                        {
                            "points": [
                                {"x": 1, "y": 1},
                                {"x": 1, "y": 2},
                                {"x": 3, "y": 3},
                                {"x": 2, "y": 1},
                            ]
                        },
                        {
                            "points": [
                                {"x": 4, "y": 4},
                                {"x": 4, "y": 5},
                                {"x": 4.5, "y": 5.5},
                                {"x": 5, "y": 5},
                                {"x": 5, "y": 4},
                            ]
                        },
                    ],
                },
            },
        ],
    }
    _test_post_endpoints(
        client=client,
        endpoint="/groundtruths",
        crud_method_name="create_groundtruth",
        example_json=example_json,
    )


def test_post_groundtruth_raster_segmentation(client: TestClient):
    example_json = {
        "datum": {
            "uid": "file_uid",
            "dataset": "dataset1",
            "metadata": [
                {"key": "height", "value": 20},
                {"key": "width", "value": 20},
            ],
        },
        "annotations": [
            {
                "labels": [
                    {"key": "k1", "value": "v1"},
                    {"key": "k1", "value": "v2"},
                ],
                "task_type": "instance_segmentation",
                "metadata": [
                    {"key": "meta1", "value": 0.4},
                    {"key": "meta1", "value": "v1"},
                ],
                "raster": {
                    "mask": "iVBORw0KGgoAAAANSUhEUgAAABQAAAAUAQAAAACl8iCgAAAAF0lEQVR4nGP4f4CBiYGBIGZgsP9AjDoAuysDE0GVDN8AAAAASUVORK5CYII=",
                },
            },
            {
                "labels": [
                    {"key": "k1", "value": "v1"},
                    {"key": "k1", "value": "v2"},
                ],
                "task_type": "semantic_segmentation",
                "metadata": [
                    {"key": "meta1", "value": 0.4},
                    {"key": "meta1", "value": "v1"},
                ],
                "raster": {
                    "mask": "iVBORw0KGgoAAAANSUhEUgAAABQAAAAUAQAAAACl8iCgAAAAF0lEQVR4nGP4f4CBiYGBIGZgsP9AjDoAuysDE0GVDN8AAAAASUVORK5CYII=",
                },
            },
        ],
    }
    _test_post_endpoints(
        client=client,
        endpoint="/groundtruths",
        crud_method_name="create_groundtruth",
        example_json=example_json,
    )


""" POST /predictions """


def test_post_prediction(client: TestClient):
    example_json = {
        "model": "model1",
        "datum": {
            "uid": "file_uid",
            "dataset": "dataset1",
            "metadata": [],
        },
        "annotations": [
            {
                "scored_labels": [
                    {"label": {"key": "k1", "value": "v1"}, "score": 0.9},
                    {
                        "label": {"key": "k1", "value": "v2"},
                        "score": 0.1,
                    },
                ],
                "task_type": "classification",
                "metadata": [],
            }
        ],
    }

    # check we get a code (404) if the model does not exist
    with patch(
        "velour_api.main.crud.create_prediction",
        side_effect=exceptions.ModelDoesNotExistError("model1"),
    ):
        resp = client.post("/predictions", json=example_json)
        assert resp.status_code == 404

    # check we get a code (409) if the datum does not exist
    with patch(
        "velour_api.main.crud.create_prediction",
        side_effect=exceptions.DatumDoesNotExistError("uid1"),
    ):
        resp = client.post("/predictions", json=example_json)
        assert resp.status_code == 404


def test_post_prediction_classification(client: TestClient):
    example_json = {
        "model": "model1",
        "datum": {
            "uid": "file_uid",
            "dataset": "dataset1",
            "metadata": [
                {"key": "meta1", "value": 0.4},
                {"key": "meta1", "value": "v1"},
            ],
        },
        "annotations": [
            {
                "scored_labels": [
                    {"label": {"key": "k1", "value": "v1"}, "score": 0.9},
                    {
                        "label": {"key": "k1", "value": "v2"},
                        "score": 0.1,
                    },
                ],
                "task_type": "classification",
                "metadata": [
                    {"key": "meta1", "value": 0.4},
                    {"key": "meta1", "value": "v1"},
                ],
            },
            {
                "scored_labels": [
                    {"label": {"key": "k1", "value": "v1"}, "score": 0.9},
                    {
                        "label": {"key": "k1", "value": "v2"},
                        "score": 0.1,
                    },
                ],
                "task_type": "classification",
                "metadata": [
                    {"key": "meta2", "value": 0.4},
                    {"key": "meta2", "value": "v1"},
                ],
            },
        ],
    }
    _test_post_endpoints(
        client=client,
        endpoint="/predictions",
        crud_method_name="create_prediction",
        example_json=example_json,
    )


def test_post_prediction_bbox_detection(client: TestClient):
    example_json = {
        "model": "model1",
        "datum": {
            "uid": "file_uid",
            "dataset": "dataset1",
            "metadata": [
                {"key": "meta1", "value": 0.4},
                {"key": "meta1", "value": "v1"},
            ],
        },
        "annotations": [
            {
                "scored_labels": [
                    {"label": {"key": "k1", "value": "v1"}, "score": 0.9},
                    {
                        "label": {"key": "k1", "value": "v2"},
                        "score": 0.1,
                    },
                ],
                "task_type": "detection",
                "metadata": [
                    {"key": "meta1", "value": 0.4},
                    {"key": "meta1", "value": "v1"},
                ],
                "bounding_box": {
                    "polygon": {
                        "points": [
                            {"x": 0, "y": 0},
                            {"x": 0, "y": 1},
                            {"x": 1, "y": 1},
                            {"x": 1, "y": 0},
                        ]
                    }
                },
            },
        ],
    }
    _test_post_endpoints(
        client=client,
        endpoint="/predictions",
        crud_method_name="create_prediction",
        example_json=example_json,
    )


def test_post_prediction_polygon_detection(client: TestClient):
    example_json = {
        "model": "model1",
        "datum": {
            "uid": "file_uid",
            "dataset": "dataset1",
            "metadata": [
                {"key": "meta1", "value": 0.4},
                {"key": "meta1", "value": "v1"},
            ],
        },
        "annotations": [
            {
                "scored_labels": [
                    {"label": {"key": "k1", "value": "v1"}, "score": 0.9},
                    {
                        "label": {"key": "k1", "value": "v2"},
                        "score": 0.1,
                    },
                ],
                "task_type": "detection",
                "metadata": [
                    {"key": "meta1", "value": 0.4},
                    {"key": "meta1", "value": "v1"},
                ],
                "polygon": {
                    "boundary": {
                        "points": [
                            {"x": 0, "y": 0},
                            {"x": 0, "y": 10},
                            {"x": 10, "y": 10},
                            {"x": 10, "y": 0},
                        ]
                    },
                    "holes": [
                        {
                            "points": [
                                {"x": 1, "y": 1},
                                {"x": 1, "y": 2},
                                {"x": 3, "y": 3},
                                {"x": 2, "y": 1},
                            ]
                        },
                        {
                            "points": [
                                {"x": 4, "y": 4},
                                {"x": 4, "y": 5},
                                {"x": 4.5, "y": 5.5},
                                {"x": 5, "y": 5},
                                {"x": 5, "y": 4},
                            ]
                        },
                    ],
                },
            },
        ],
    }

    from velour_api import schemas

    schemas.Prediction(**example_json)

    _test_post_endpoints(
        client=client,
        endpoint="/predictions",
        crud_method_name="create_prediction",
        example_json=example_json,
    )


def test_post_prediction_raster_segmentation(client: TestClient):
    example_json = {
        "model": "model1",
        "datum": {
            "uid": "file_uid",
            "dataset": "dataset1",
            "metadata": [
                {"key": "height", "value": 20},
                {"key": "width", "value": 20},
            ],
        },
        "annotations": [
            {
                "scored_labels": [
                    {"label": {"key": "k1", "value": "v1"}, "score": 0.9},
                    {
                        "label": {"key": "k1", "value": "v2"},
                        "score": 0.1,
                    },
                ],
                "task_type": "instance_segmentation",
                "metadata": [
                    {"key": "meta1", "value": 0.4},
                    {"key": "meta1", "value": "v1"},
                ],
                "raster": {
                    "mask": "iVBORw0KGgoAAAANSUhEUgAAABQAAAAUAQAAAACl8iCgAAAAF0lEQVR4nGP4f4CBiYGBIGZgsP9AjDoAuysDE0GVDN8AAAAASUVORK5CYII=",
                },
            },
            {
                "scored_labels": [
                    {"label": {"key": "k1", "value": "v1"}, "score": 0.9},
                    {
                        "label": {"key": "k1", "value": "v2"},
                        "score": 0.1,
                    },
                ],
                "task_type": "semantic_segmentation",
                "metadata": [
                    {"key": "meta1", "value": 0.4},
                    {"key": "meta1", "value": "v1"},
                ],
                "raster": {
                    "mask": "iVBORw0KGgoAAAANSUhEUgAAABQAAAAUAQAAAACl8iCgAAAAF0lEQVR4nGP4f4CBiYGBIGZgsP9AjDoAuysDE0GVDN8AAAAASUVORK5CYII=",
                },
            },
        ],
    }
    _test_post_endpoints(
        client=client,
        endpoint="/predictions",
        crud_method_name="create_prediction",
        example_json=example_json,
    )


""" POST /datasets  """


def test_post_datasets(client: TestClient):
    example_json = {
        "id": 1,
        "name": "dataset1",
        "metadata": [
            {"key": "meta1", "value": 0.4},
            {"key": "meta1", "value": "v1"},
        ],
    }
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


""" POST /models """


def test_post_models(client: TestClient):
    example_json = {
        "id": 1,
        "name": "model1",
        "metadata": [
            {"key": "meta1", "value": 0.4},
            {"key": "meta1", "value": "v1"},
        ],
    }
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


""" GET /datasets """


@patch("velour_api.main.crud")
def test_get_datasets(crud, client: TestClient):
    crud.get_datasets.return_value = []
    resp = client.get("/datasets")
    assert resp.status_code == 200
    crud.get_datasets.assert_called_once()


""" GET /datasets/{dataset_name}"""


@patch("velour_api.main.crud")
def test_get_dataset_by_name(crud, client: TestClient):
    crud.get_dataset.return_value = schemas.Dataset(
        id=1, name="name", metadata=[]
    )
    resp = client.get("/datasets/name")
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


""" GET /models/{model_name}"""


@patch("velour_api.main.crud")
def test_get_model_by_name(crud, client: TestClient):
    crud.get_model.return_value = schemas.Model(id=1, name="name", metadata=[])
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


""" PUT /datasets/{dataset_name}/finalize """


@patch("velour_api.main.crud")
def test_finalize_datasets(crud, client: TestClient):
    resp = client.put("/datasets/dsetname/finalize")
    assert resp.status_code == 200
    crud.finalize.assert_called_once()

    with patch(
        "velour_api.main.crud.finalize",
        side_effect=exceptions.DatasetDoesNotExistError(""),
    ):
        resp = client.put("datasets/dsetname/finalize")
        assert resp.status_code == 404

    # @FIXME Not sure why this is failing
    # resp = client.get("/datasets/dsetname/finalize")
    # assert resp.status_code == 405


""" GET /dataset/{dataset_name}/labels"""


@patch("velour_api.main.crud")
def test_get_dataset_labels(crud, client: TestClient):
    crud.get_labels.return_value = []
    resp = client.get("/datasets/dsetname/labels")
    assert resp.status_code == 200
    crud.get_labels.assert_called_once()

    with patch(
        "velour_api.main.crud.get_labels",
        side_effect=exceptions.DatasetDoesNotExistError(""),
    ):
        resp = client.get("datasets/dsetname/labels")
        assert resp.status_code == 404

    resp = client.post("/datasets/dsetname/labels")
    assert resp.status_code == 405


""" GET /dataset/{dataset_name}/data """


@patch("velour_api.main.crud")
def test_get_dataset_datums(crud, client: TestClient):
    crud.get_datums.return_value = []
    resp = client.get("/datasets/dsetname/data")
    assert resp.status_code == 200
    crud.get_datums.assert_called_once()

    with patch(
        "velour_api.main.crud.get_datums",
        side_effect=exceptions.DatasetDoesNotExistError(""),
    ):
        resp = client.get("datasets/dsetname/data")
        assert resp.status_code == 404

    resp = client.post("/datasets/dsetname/data")
    assert resp.status_code == 405


""" GET /dataset/{dataset_name}/data/filter/{task_type} """


@patch("velour_api.main.crud")
def test_get_dataset_datums_by_filter(crud, client: TestClient):
    crud.get_datums.return_value = []
    resp = client.get("/datasets/dsetname/data/filter/task_type")
    assert resp.status_code == 200
    crud.get_datums.assert_called_once()

    with patch(
        "velour_api.main.crud.get_datums",
        side_effect=exceptions.DatasetDoesNotExistError(""),
    ):
        resp = client.get("datasets/dsetname/data/filter/task_type")
        assert resp.status_code == 404

    resp = client.post("/datasets/dsetname/data/filter/task_type")
    assert resp.status_code == 405


""" GET /dataset/{dataset_name}/data/{uid} """


@patch("velour_api.main.crud")
def test_get_dataset_datum(crud, client: TestClient):
    crud.get_datum.return_value = None
    resp = client.get("/datasets/dsetname/data/uid")
    assert resp.status_code == 200
    crud.get_datum.assert_called_once()

    with patch(
        "velour_api.main.crud.get_datum",
        side_effect=exceptions.DatasetDoesNotExistError(""),
    ):
        resp = client.get("datasets/dsetname/data/uid")
        assert resp.status_code == 404

    resp = client.post("/datasets/dsetname/data/uid")
    assert resp.status_code == 405


""" DELETE /datasets/{dataset_name} """


@patch("velour_api.main.crud")
def test_delete_dataset(crud, client: TestClient):
    crud.delete.return_value = None
    resp = client.delete("/datasets/dsetname")
    assert resp.status_code == 200
    crud.delete.assert_called_once()


""" GET /models """


@patch("velour_api.main.crud")
def test_get_models(crud, client: TestClient):
    crud.get_models.return_value = []
    resp = client.get("/models")
    assert resp.status_code == 200
    crud.get_models.assert_called_once()


""" DELETE /models/{model_nam,e} """


@patch("velour_api.main.crud")
def test_delete_model(crud, client: TestClient):
    crud.delete.return_value = None
    resp = client.delete("/models/modelname")
    assert resp.status_code == 200
    crud.delete.assert_called_once()


""" GET /labels """


@patch("velour_api.main.crud")
def test_get_labels(crud, client: TestClient):
    crud.get_labels.return_value = []
    resp = client.get("/labels")
    assert resp.status_code == 200
    crud.get_labels.assert_called_once()

    resp = client.post("/labels")
    assert resp.status_code == 405


""" GET /user """


def test_user(client: TestClient):
    resp = client.get("/user")
    assert resp.json() == {"email": None}
