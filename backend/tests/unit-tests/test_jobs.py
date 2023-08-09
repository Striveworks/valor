import json
from unittest.mock import MagicMock

import pytest

from velour_api.backend import jobs
from velour_api.enums import JobStatus
from velour_api.schemas.stateflow import BackendStateflow


@pytest.fixture(autouse=True)
def teardown():
    # reset the global redis instance (which may have been set by other tests)
    jobs.r = None


def test_get_status():
    """Checks that multiple calls of `get_status` only
    connects to redis once
    """

    jobs.redis.Redis = MagicMock()

    assert jobs.redis.Redis.call_count == 0
    assert jobs.get_status(0) is None
    assert jobs.redis.Redis.call_count == 1
    assert jobs.get_status(0) is None
    assert jobs.redis.Redis.call_count == 1


def test_set_status():
    """Checks that multiple calls of `set_status` only
    connects to redis once
    """
    jobs.redis.Redis = MagicMock()

    assert jobs.redis.Redis.call_count == 0
    jobs.set_status(0, JobStatus.PENDING)
    assert jobs.redis.Redis.call_count == 1
    jobs.set_status(1, JobStatus.PENDING)
    assert jobs.redis.Redis.call_count == 1


def test_remove_status():
    """Checks that multiple calls of `remove_status` only
    connects to redis once
    """

    r = MagicMock()
    r.get = MagicMock(
        return_value=bytes(
            json.dumps({"jobs": {0: "done"}}),
            encoding="utf-8",
        )
    )
    r.set = MagicMock()

    jobs.redis = MagicMock()
    jobs.redis.Redis = MagicMock(return_value=r)

    assert jobs.redis.Redis.call_count == 0
    jobs.remove_status(0)
    assert jobs.redis.Redis.call_count == 1
    jobs.remove_status(0)
    assert jobs.redis.Redis.call_count == 1


def test_get_backend_state():
    """Checks that multiple calls of `get_backend_state` only
    connects to redis once
    """
    jobs.redis.Redis = MagicMock()

    assert jobs.redis.Redis.call_count == 0
    jobs.get_backend_state()
    assert jobs.redis.Redis.call_count == 1
    jobs.get_backend_state()
    assert jobs.redis.Redis.call_count == 1


def test_set_backend_state():
    """Checks that multiple calls of `set_backend_state` only
    connects to redis once
    """
    jobs.redis.Redis = MagicMock()
    jobs.json = MagicMock()

    assert jobs.redis.Redis.call_count == 0
    jobs.set_backend_state(BackendStateflow())
    assert jobs.redis.Redis.call_count == 1
    jobs.set_backend_state(BackendStateflow())
    assert jobs.redis.Redis.call_count == 1
