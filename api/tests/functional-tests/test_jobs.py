""" These tests require a redis instance either running unauthenticated
at localhost:6379 or with the following enviornment variables set:
REDIS_HOST, REDIS_PORT, REDIS_PASSWORD, REDIS_USERNAME, REDIS_DB
"""
from unittest.mock import MagicMock  # , patch

import pytest
from fastapi.testclient import TestClient

from velour_api.backend import database
from velour_api.crud import jobs
from velour_api.crud.jobs import (
    Job,
    generate_uuid,
    get_status_from_names,
    get_status_from_uuid,
)
from velour_api.enums import JobStatus


@pytest.fixture
def client() -> TestClient:
    database.create_db = MagicMock()
    database.make_session = MagicMock()
    from velour_api import main

    main.get_db = MagicMock()

    return TestClient(main.app)


@pytest.fixture(autouse=True)
def db():
    """The setup checks that the redis db is empty and
    the teardown flushes it
    """
    jobs.connect_to_redis()
    if len(jobs.r.keys()) != 0:
        raise RuntimeError("redis database is not-empty")
    yield
    jobs.r.flushdb()


@pytest.fixture
def evaluation_id() -> int:
    return 123


@pytest.fixture
def uuid(dataset_name: str, model_name: str, evaluation_id: int) -> str:
    return generate_uuid(dataset_name, model_name, evaluation_id)


@pytest.fixture
def create_job(uuid: str) -> Job:
    job = Job.get(uuid)
    job.status = JobStatus.PROCESSING
    job.sync()
    yield job
    job.delete()


def test_get_status_from_names(
    dataset_name: str, model_name: str, evaluation_id: str, create_job: Job
):
    assert get_status_from_names(
        dataset_name=dataset_name,
        model_name=model_name,
        evaluation_id=evaluation_id,
    )


def test_get_status_from_uuid(uuid, create_job: Job):
    assert get_status_from_uuid(uuid)
