""" These tests require a redis instance either running unauthenticated
at localhost:6379 or with the following enviornment variables set:
REDIS_HOST, REDIS_PORT, REDIS_PASSWORD, REDIS_USERNAME, REDIS_DB
"""
from unittest.mock import MagicMock  # , patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from velour_api import crud, enums, exceptions, schemas
from velour_api.backend import database
from velour_api.crud import jobs


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


