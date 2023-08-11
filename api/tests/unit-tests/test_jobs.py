from unittest.mock import MagicMock

import pytest

from velour_api.backend import jobs
from velour_api.schemas.stateflow import Stateflow


@pytest.fixture(autouse=True)
def teardown():
    # reset the global redis instance (which may have been set by other tests)
    jobs.r = None


def test_get_stateflow():
    """Checks that multiple calls of `get_stateflow` only
    connects to redis once
    """
    jobs.redis.Redis = MagicMock()

    assert jobs.redis.Redis.call_count == 0
    jobs.get_stateflow()
    assert jobs.redis.Redis.call_count == 1
    jobs.get_stateflow()
    assert jobs.redis.Redis.call_count == 1


def test_set_stateflow():
    """Checks that multiple calls of `set_stateflow` only
    connects to redis once
    """
    jobs.redis.Redis = MagicMock()
    jobs.json = MagicMock()

    assert jobs.redis.Redis.call_count == 0
    jobs.set_stateflow(Stateflow())
    assert jobs.redis.Redis.call_count == 1
    jobs.set_stateflow(Stateflow())
    assert jobs.redis.Redis.call_count == 1
