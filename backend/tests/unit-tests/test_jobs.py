from unittest.mock import MagicMock

import pytest

from velour_api.backend import jobs


@pytest.fixture(autouse=True)
def teardown():
    # reset the global redis instance (which may have been set by other tests)
    jobs.r = None


def test_get_jobs():
    """Checks that multiple calls of `get_jobs` only
    connects to redis once
    """
    jobs.redis.Redis = MagicMock()

    assert jobs.redis.Redis.call_count == 0
    jobs.get_all_jobs()
    assert jobs.redis.Redis.call_count == 1
    jobs.get_all_jobs()
    assert jobs.redis.Redis.call_count == 1


def test_get_job():
    """Checks that multiple calls of `get_job` only
    connects to redis once
    """
    # reset the global redis instance (which may have been set by other tests)
    # jobs.r = None
    jobs.redis.Redis = MagicMock()
    jobs.json = MagicMock()

    assert jobs.redis.Redis.call_count == 0
    jobs.get_job("")
    assert jobs.redis.Redis.call_count == 1
    jobs.get_job("")
    assert jobs.redis.Redis.call_count == 1


def test_add_job():
    """Checks that multiple calls of `add_job` only
    connects to redis once
    """
    jobs.redis.Redis = MagicMock()
    jobs.json = MagicMock()

    job = MagicMock()

    assert jobs.redis.Redis.call_count == 0
    jobs.add_job(job)
    assert jobs.redis.Redis.call_count == 1
    jobs.add_job(job)
    assert jobs.redis.Redis.call_count == 1
