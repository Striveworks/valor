from unittest.mock import MagicMock, PropertyMock, patch

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from velour_api import exceptions, schemas
from velour_api.api_utils import _split_query_params
from velour_api.backend import database
from velour_api.enums import TableStatus, TaskType


@pytest.fixture
def client() -> TestClient:
    database.create_db = MagicMock()
    database.make_session = MagicMock()
    from velour_api import main

    main.get_db = MagicMock()

    return TestClient(main.app)


def test__split_query_params():
    """Test helper function for splitting GET params into a list"""
    param_string = None
    assert _split_query_params(param_string) is None

    param_string = "model"
    assert _split_query_params(param_string) == ["model"]

    param_string = "model1,model2"
    assert _split_query_params(param_string) == ["model1", "model2"]


def test_protected_routes(client: TestClient):
    """Check that all non-status routes are protected"""
    routes = [
        r
        for r in client.app.routes
        if isinstance(r, APIRoute) and r.name not in {"health", "ready"}
    ]
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


@patch("velour_api.main.crud")
def _test_post_evaluation_endpoint(
    crud,
    client: TestClient,
    endpoint: str,
    crud_method_name: str,
    example_json: dict,
    metric_response: schemas.CreateDetectionMetricsResponse
    | schemas.CreateClfMetricsResponse
    | schemas.CreateSemanticSegmentationMetricsResponse,
):
    """Helper function to test our metric endpoints by patching fastapi's BackgroundTasks"""
    crud_method = getattr(crud, crud_method_name)
    crud_method.return_value = metric_response

    resp = client.post(endpoint, json=example_json)
    assert resp.status_code == 202
    crud_method.assert_called_once()

    resp = client.post(endpoint, json={})
    assert resp.status_code == 422

    with patch(
        "fastapi.BackgroundTasks.add_task",
        side_effect=ValueError(),
    ):
        resp = client.post(endpoint, json=example_json)
        assert resp.status_code == 400

    with patch(
        "fastapi.BackgroundTasks.add_task",
        side_effect=exceptions.DatasetNotFinalizedError(""),
    ):
        resp = client.post(endpoint, json=example_json)
        assert resp.status_code == 405

    with patch(
        "fastapi.BackgroundTasks.add_task",
        side_effect=exceptions.ModelStateError("a", "b", "c"),
    ):
        resp = client.post(endpoint, json=example_json)
        assert resp.status_code == 409


""" POST /groundtruths """


def test_post_groundtruth(client: TestClient):
    example_json = {
        "datum": {
            "uid": "file_uid",
            "dataset": "dataset1",
            "metadata": {},
        },
        "annotations": [
            {
                "labels": [
                    {"key": "k1", "value": "v1"},
                ],
                "task_type": TaskType.CLASSIFICATION.value,
                "metadata": {},
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

    # check that we get an error if the dataset doesn't exist
    with patch(
        "velour_api.main.crud.create_groundtruth",
        side_effect=exceptions.DatasetDoesNotExistError("fake_dsetname"),
    ):
        resp = client.post("/groundtruths", json=example_json)
        assert resp.status_code == 404


def test_post_groundtruth_classification(client: TestClient):
    example_json = {
        "datum": {
            "uid": "file_uid",
            "dataset": "dataset1",
            "metadata": {
                "meta1": 0.4,
                "meta2": "v1",
            },
        },
        "annotations": [
            {
                "labels": [
                    {"key": "k1", "value": "v1"},
                    {"key": "k1", "value": "v2"},
                ],
                "task_type": TaskType.CLASSIFICATION.value,
                "metadata": {
                    "meta1": 0.4,
                    "meta2": "v1",
                },
            },
            {
                "labels": [
                    {"key": "k2", "value": "v1"},
                    {"key": "k2", "value": "v2"},
                ],
                "task_type": TaskType.CLASSIFICATION.value,
                "metadata": {
                    "meta1": 0.4,
                    "meta2": "v1",
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


def test_post_groundtruth_bbox_detection(client: TestClient):
    example_json = {
        "datum": {
            "uid": "file_uid",
            "dataset": "dataset1",
            "metadata": {
                "meta1": 0.4,
                "meta2": "v1",
            },
        },
        "annotations": [
            {
                "labels": [
                    {"key": "k1", "value": "v1"},
                    {"key": "k1", "value": "v2"},
                ],
                "task_type": TaskType.DETECTION.value,
                "metadata": {
                    "meta1": 0.4,
                    "meta2": "v1",
                },
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
            "metadata": {
                "meta1": 0.4,
                "meta2": "v1",
            },
        },
        "annotations": [
            {
                "labels": [
                    {"key": "k1", "value": "v1"},
                    {"key": "k1", "value": "v2"},
                ],
                "task_type": TaskType.DETECTION.value,
                "metadata": {
                    "meta1": 0.4,
                    "meta2": "v1",
                },
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
            "metadata": {
                "height": 20,
                "width": 20,
            },
        },
        "annotations": [
            {
                "labels": [
                    {"key": "k1", "value": "v1"},
                    {"key": "k1", "value": "v2"},
                ],
                "task_type": TaskType.DETECTION.value,
                "metadata": {
                    "meta1": 0.4,
                    "meta2": "v1",
                },
                "raster": {
                    "mask": "iVBORw0KGgoAAAANSUhEUgAAABQAAAAUAQAAAACl8iCgAAAAF0lEQVR4nGP4f4CBiYGBIGZgsP9AjDoAuysDE0GVDN8AAAAASUVORK5CYII=",
                },
            },
            {
                "labels": [
                    {"key": "k1", "value": "v1"},
                    {"key": "k1", "value": "v2"},
                ],
                "task_type": TaskType.SEGMENTATION.value,
                "metadata": {
                    "meta1": 0.4,
                    "meta2": "v1",
                },
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


""" GET /groundtruths/dataset/{dataset_name}/datum/{uid} """


@patch("velour_api.main.crud")
def test_get_groundtruth(crud, client: TestClient):
    crud.get_groundtruth.return_value = {
        "datum": {
            "uid": "file_uid",
            "dataset": "dataset1",
            "metadata": {
                "meta1": 0.4,
                "meta2": "v1",
            },
        },
        "annotations": [
            {
                "labels": [
                    {"key": "k1", "value": "v1", "score": 0.1},
                    {"key": "k1", "value": "v2", "score": 0.1},
                ],
                "task_type": TaskType.DETECTION.value,
                "metadata": {
                    "meta1": 0.4,
                    "meta2": "v1",
                },
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

    resp = client.get("/groundtruths/dataset/dsetname/datum/1")
    assert resp.status_code == 200
    crud.get_groundtruth.assert_called_once()

    with patch(
        "velour_api.main.crud.get_groundtruth",
        side_effect=exceptions.DatasetDoesNotExistError("dsetname"),
    ):
        resp = client.get("/groundtruths/dataset/dsetname/datum/1")

        assert resp.status_code == 404


""" POST /predictions """


def test_post_prediction(client: TestClient):
    example_json = {
        "model": "model1",
        "datum": {
            "uid": "file_uid",
            "dataset": "dataset1",
            "metadata": {},
        },
        "annotations": [
            {
                "labels": [
                    {"key": "k1", "value": "v1", "score": 0.9},
                    {"key": "k1", "value": "v2", "score": 0.1},
                ],
                "task_type": TaskType.CLASSIFICATION.value,
                "metadata": {},
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

    # check we get a code (409) if the dataset hasn't been finalized
    with patch(
        "velour_api.main.crud.create_prediction",
        side_effect=exceptions.DatasetNotFinalizedError("dataset1"),
    ):
        resp = client.post("/predictions", json=example_json)
        assert resp.status_code == 409


def test_post_prediction_classification(client: TestClient):
    example_json = {
        "model": "model1",
        "datum": {
            "uid": "file_uid",
            "dataset": "dataset1",
            "metadata": {
                "meta1": 0.4,
                "meta2": "v1",
            },
        },
        "annotations": [
            {
                "labels": [
                    {"key": "k1", "value": "v1", "score": 0.9},
                    {"key": "k1", "value": "v2", "score": 0.1},
                ],
                "task_type": TaskType.CLASSIFICATION.value,
                "metadata": {
                    "meta1": 0.4,
                    "meta2": "v1",
                },
            },
            {
                "labels": [
                    {"key": "k1", "value": "v1", "score": 0.9},
                    {"key": "k1", "value": "v2", "score": 0.1},
                ],
                "task_type": TaskType.CLASSIFICATION.value,
                "metadata": {
                    "meta1": 0.4,
                    "meta2": "v1",
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


def test_post_prediction_bbox_detection(client: TestClient):
    example_json = {
        "model": "model1",
        "datum": {
            "uid": "file_uid",
            "dataset": "dataset1",
            "metadata": {
                "meta1": 0.4,
                "meta2": "v1",
            },
        },
        "annotations": [
            {
                "labels": [
                    {"key": "k1", "value": "v1", "score": 0.9},
                    {"key": "k1", "value": "v2", "score": 0.1},
                ],
                "task_type": TaskType.DETECTION.value,
                "metadata": {
                    "meta1": 0.4,
                    "meta2": "v1",
                },
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
    schemas.Prediction(**example_json)
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
            "metadata": {
                "meta1": 0.4,
                "meta2": "v1",
            },
        },
        "annotations": [
            {
                "labels": [
                    {"key": "k1", "value": "v1", "score": 0.9},
                    {"key": "k1", "value": "v2", "score": 0.1},
                ],
                "task_type": TaskType.DETECTION.value,
                "metadata": {
                    "meta1": 0.4,
                    "meta2": "v1",
                },
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
            "metadata": {
                "height": 20,
                "width": 20,
            },
        },
        "annotations": [
            {
                "labels": [
                    {"key": "k1", "value": "v1", "score": 0.9},
                    {"key": "k1", "value": "v2", "score": 0.1},
                ],
                "task_type": TaskType.DETECTION.value,
                "metadata": {
                    "meta1": 0.4,
                    "meta2": "v1",
                },
                "raster": {
                    "mask": "iVBORw0KGgoAAAANSUhEUgAAABQAAAAUAQAAAACl8iCgAAAAF0lEQVR4nGP4f4CBiYGBIGZgsP9AjDoAuysDE0GVDN8AAAAASUVORK5CYII=",
                },
            },
            {
                "labels": [
                    {"key": "k1", "value": "v1"},
                    {"key": "k1", "value": "v2"},
                ],
                "task_type": TaskType.SEGMENTATION.value,
                "metadata": {
                    "meta1": 0.4,
                    "meta2": "v1",
                },
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


""" GET /predictions/model/{model_name}/dataset/{dataset_name}/datum/{uid} """


@patch("velour_api.main.crud")
def test_get_prediction(crud, client: TestClient):
    crud.get_prediction.return_value = {
        "model": "model1",
        "datum": {
            "uid": "file_uid",
            "dataset": "dataset1",
            "metadata": {
                "meta1": 0.4,
                "meta2": "v1",
            },
        },
        "annotations": [
            {
                "labels": [
                    {"key": "k1", "value": "v1", "score": 0.1},
                    {"key": "k1", "value": "v2", "score": 0.1},
                ],
                "task_type": TaskType.DETECTION.value,
                "metadata": {
                    "meta1": 0.4,
                    "meta2": "v1",
                },
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

    resp = client.get("/predictions/model/model_name/dataset/dsetname/datum/1")
    assert resp.status_code == 200
    crud.get_prediction.assert_called_once()

    with patch(
        "velour_api.main.crud.get_prediction",
        side_effect=exceptions.DatasetDoesNotExistError("dsetname"),
    ):
        resp = client.get(
            "/predictions/model/model_name/dataset/dsetname/datum/1"
        )
        assert resp.status_code == 404


""" POST /datasets  """


def test_post_datasets(client: TestClient):
    example_json = {
        "id": 1,
        "name": "dataset1",
        "metadata": {
            "meta1": 0.4,
            "meta2": "v1",
        },
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


""" GET /datasets """


@patch("velour_api.main.crud")
def test_get_datasets(crud, client: TestClient):
    crud.get_datasets.return_value = []
    resp = client.get("/datasets")
    assert resp.status_code == 200
    crud.get_datasets.assert_called_once()


""" GET /datasets/{dataset_name} """


@patch("velour_api.main.crud")
def test_get_dataset_by_name(crud, client: TestClient):
    crud.get_dataset.return_value = schemas.Dataset(
        id=1, name="name", metadata={}
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


""" GET /datasets/{dataset_name}/status"""


@patch("velour_api.main.crud")
def test_get_dataset_status(crud, client: TestClient):
    crud.get_table_status.return_value = TableStatus.FINALIZED.value
    resp = client.get("/datasets/dsetname/status")
    assert resp.status_code == 200
    crud.get_table_status.assert_called_once()

    with patch(
        "velour_api.main.crud.get_table_status",
        side_effect=exceptions.DatasetDoesNotExistError(""),
    ):
        resp = client.get("/datasets/dsetname/status")
        assert resp.status_code == 404


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

    with patch(
        "velour_api.main.crud.finalize",
        side_effect=exceptions.DatasetIsEmptyError(""),
    ):
        resp = client.put("datasets/dsetname/finalize")
        assert resp.status_code == 400

    resp = client.get("/datasets/dsetname/finalize")
    assert resp.status_code == 405


""" DELETE /datasets/{dataset_name} """


@patch("velour_api.main.crud")
def test_delete_dataset(crud, client: TestClient):
    crud.delete.return_value = None
    resp = client.delete("/datasets/dsetname")
    assert resp.status_code == 200
    assert crud.delete.call_count == 1


""" POST /models """


def test_post_models(client: TestClient):
    example_json = {
        "id": 1,
        "name": "model1",
        "metadata": {
            "meta1": 0.4,
            "meta2": "v1",
        },
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


""" GET /models"""


@patch("velour_api.main.crud")
def test_get_models(crud, client: TestClient):
    crud.get_models.return_value = []
    resp = client.get("/models")
    assert resp.status_code == 200
    crud.get_models.assert_called_once()


""" GET /models/{model_name} """


@patch("velour_api.main.crud")
def test_get_model_by_name(crud, client: TestClient):
    crud.get_model.return_value = schemas.Model(id=1, name="name", metadata={})
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


""" PUT /models/{model_name}/finalize/datasets/{dataset_name}/finalize """


@patch("velour_api.main.crud")
def test_finalize_inferences(crud, client: TestClient):
    resp = client.put("/models/modelname/datasets/dsetname/finalize")
    assert resp.status_code == 200
    crud.finalize.assert_called_once()

    with patch(
        "velour_api.main.crud.finalize",
        side_effect=exceptions.DatasetDoesNotExistError(""),
    ):
        resp = client.put("/models/modelname/datasets/dsetname/finalize")
        assert resp.status_code == 404

    with patch(
        "velour_api.main.crud.finalize",
        side_effect=exceptions.DatasetIsEmptyError(""),
    ):
        resp = client.put("/models/modelname/datasets/dsetname/finalize")
        assert resp.status_code == 400

    resp = client.get("/models/modelname/datasets/dsetname/finalize")
    assert resp.status_code == 405


""" DELETE /models/{model_name} """


@patch("velour_api.main.crud")
def test_delete_model(crud, client: TestClient):
    crud.delete.return_value = None
    resp = client.delete("/models/modelname")
    assert resp.status_code == 200
    assert crud.delete.call_count == 1


""" POST /evaluations/ap-metrics """


def test_post_detection_metrics(client: TestClient):
    metric_response = schemas.CreateDetectionMetricsResponse(
        missing_pred_labels=[], ignored_pred_labels=[], evaluation_id=1
    )

    example_json = {
        "model": "modelname",
        "dataset": "dsetname",
        "task_type": TaskType.DETECTION.value,
        "settings": {},
    }

    _test_post_evaluation_endpoint(
        client=client,
        crud_method_name="create_detection_evaluation",
        endpoint="/evaluations",
        metric_response=metric_response,
        example_json=example_json,
    )


""" POST /evaluations/clf-metrics """


def test_post_clf_metrics(client: TestClient):
    metric_response = schemas.CreateClfMetricsResponse(
        missing_pred_keys=[], ignored_pred_keys=[], evaluation_id=1
    )

    example_json = {
        "model": "modelname",
        "dataset": "dsetname",
        "task_type": TaskType.CLASSIFICATION.value,
        "settings": {},
    }

    _test_post_evaluation_endpoint(
        client=client,
        crud_method_name="create_clf_evaluation",
        endpoint="/evaluations",
        metric_response=metric_response,
        example_json=example_json,
    )


""" POST /evaluations/semantic-segmentation-metrics """


def test_post_semenatic_segmentation_metrics(client: TestClient):
    metric_response = schemas.CreateSemanticSegmentationMetricsResponse(
        missing_pred_labels=[], ignored_pred_labels=[], evaluation_id=1
    )

    example_json = {
        "model": "modelname",
        "dataset": "dsetname",
        "task_type": TaskType.SEGMENTATION.value,
        "settings": {},
    }

    _test_post_evaluation_endpoint(
        client=client,
        crud_method_name="create_semantic_segmentation_evaluation",
        endpoint="/evaluations",
        metric_response=metric_response,
        example_json=example_json,
    )


""" GET /evaluations"""


@patch("velour_api.main.crud")
def test_get_bulk_evaluations(crud, client: TestClient):
    crud.get_evaluations.return_value = []

    resp = client.get(
        "/evaluations?models=model1,model2&datasets=dataset1,dataset2"
    )
    assert resp.status_code == 200
    crud.get_evaluations.assert_called_once()

    with patch(
        "velour_api.main.crud.get_evaluations",
        side_effect=ValueError(),
    ):
        resp = client.get(
            "/evaluations?models=model1,model2&datasets=dataset1,dataset2"
        )
        assert resp.status_code == 400

    with patch(
        "velour_api.main.crud.get_evaluations",
        side_effect=exceptions.DatasetDoesNotExistError("dataset1"),
    ):
        resp = client.get(
            "/evaluations?models=model1,model2&datasets=dataset1,dataset2"
        )
        assert resp.status_code == 404


@patch("velour_api.main.crud")
def test_get_bulk_evaluations_evaluation_ids(crud, client: TestClient):
    crud.get_evaluations.return_value = []

    def test_ids(ids_str, ids_expected):
        resp = client.get(f"/evaluations?evaluation_ids={ids_str}")
        assert resp.status_code == 200
        call_kwargs = crud.get_evaluations.call_args.kwargs
        assert call_kwargs["evaluation_ids"] == ids_expected

    test_ids("4", [4])
    test_ids("4,7", [4, 7])
    test_ids("4,7,11", [4, 7, 11])

    for query in ("foo", "4,foo", "1,,"):
        resp = client.get(f"/evaluations?evaluation_ids={query}")
        assert resp.status_code == 400


""" GET /labels/dataset/{dataset_name} """


@patch("velour_api.main.crud")
def test_get_dataset_labels(crud, client: TestClient):
    crud.get_dataset_labels.return_value = []
    resp = client.get("/labels/dataset/dsetname")
    assert resp.status_code == 200
    crud.get_dataset_labels.assert_called_once()

    with patch(
        "velour_api.main.crud.get_dataset_labels",
        side_effect=exceptions.DatasetDoesNotExistError(""),
    ):
        resp = client.get("/labels/dataset/dsetname")
        assert resp.status_code == 404

    resp = client.post("/labels/dataset/dsetname")
    assert resp.status_code == 405


""" GET /labels/model/{model_name} """


@patch("velour_api.main.crud")
def test_get_model_labels(crud, client: TestClient):
    crud.get_labels_from_model.return_value = []
    resp = client.get("/labels/model/modelname")
    assert resp.status_code == 200
    crud.get_model_labels.assert_called_once()

    with patch(
        "velour_api.main.crud.get_model_labels",
        side_effect=exceptions.ModelDoesNotExistError(""),
    ):
        resp = client.get("/labels/model/modelname")
        assert resp.status_code == 404


""" GET /data/dataset/{dataset_name} """


@patch("velour_api.main.crud")
def test_get_dataset_datums(crud, client: TestClient):
    crud.get_datums.return_value = []
    resp = client.get("/data/dataset/dsetname")
    assert resp.status_code == 200
    crud.get_datums.assert_called_once()

    with patch(
        "velour_api.main.crud.get_datums",
        side_effect=exceptions.DatasetDoesNotExistError(""),
    ):
        resp = client.get("/data/dataset/dsetname")
        assert resp.status_code == 404

    resp = client.post("/data/dataset/dsetname")
    assert resp.status_code == 405


""" GET /data/dataset/{dataset_name}/uid/{uid} """


@patch("velour_api.main.crud")
def test_get_dataset_datum(crud, client: TestClient):
    crud.get_datum.return_value = None
    resp = client.get("/data/dataset/dsetname/uid/uid")
    assert resp.status_code == 200
    crud.get_datum.assert_called_once()

    with patch(
        "velour_api.main.crud.get_datum",
        side_effect=exceptions.DatasetDoesNotExistError(""),
    ):
        resp = client.get("/data/dataset/dsetname/uid/uid")
        assert resp.status_code == 404

    resp = client.post("/data/dataset/dsetname/uid/uid")
    assert resp.status_code == 405


""" GET /labels """


@patch("velour_api.main.crud")
def test_get_all_labels(crud, client: TestClient):
    crud.get_all_labels.return_value = []
    resp = client.get("/labels")
    assert resp.status_code == 200
    crud.get_all_labels.assert_called_once()

    resp = client.post("/labels")
    assert resp.status_code == 405


""" GET /user """


def test_get_user(client: TestClient):
    resp = client.get("/user")
    assert resp.json() == {"email": None}
