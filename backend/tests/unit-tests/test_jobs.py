from unittest.mock import MagicMock

import pytest

from velour_api.backend import jobs
from velour_api.schemas import BackendStatus


@pytest.fixture(autouse=True)
def teardown():
    # reset the global redis instance (which may have been set by other tests)
    jobs.r = None


def test__get_backend_status():
    """Checks that multiple calls of `_get_backend_status` only
    connects to redis once
    """
    jobs.redis.Redis = MagicMock()

    assert jobs.redis.Redis.call_count == 0
    jobs._get_backend_status()
    assert jobs.redis.Redis.call_count == 1
    jobs._get_backend_status()
    assert jobs.redis.Redis.call_count == 1


def test__set_backend_status():
    """Checks that multiple calls of `_set_backend_status` only
    connects to redis once
    """
    jobs.redis.Redis = MagicMock()
    jobs.json = MagicMock()

    assert jobs.redis.Redis.call_count == 0
    jobs._set_backend_status(BackendStatus())
    assert jobs.redis.Redis.call_count == 1
    jobs._set_backend_status(BackendStatus())
    assert jobs.redis.Redis.call_count == 1


def test__get_evaluation_status():
    pass


def test__set_evaluation_status():
    pass


def test__get_names():

    dataset_name = "ds1"
    model_name = "md1"

    assert (dataset_name, model_name) == jobs._get_names(
        dataset_name, model_name
    )
    assert (dataset_name, None) == jobs._get_names(dataset_name)
    assert (dataset_name, None) == jobs._get_names(dataset_name=dataset_name)
    assert (dataset_name, model_name) == jobs._get_names(
        dataset_name=dataset_name, model_name=model_name
    )
    assert (dataset_name, model_name) == jobs._get_names(
        dataset_name, model_name=model_name
    )

    with pytest.raises(ValueError) as e:
        jobs._get_names()
    assert "dataset_name not provided" in str(e)
    with pytest.raises(ValueError) as e:
        jobs._get_names(123)
    assert "must be of type `str`" in str(e)
    with pytest.raises(ValueError) as e:
        jobs._get_names("123", 123)
    assert "must be of type `str`" in str(e)


def test__validate_backend_status():
    pass


def test_create():
    pass


def test_read():
    pass


def test_finalize():
    pass


def test_evaluate():
    pass


def test_delete():
    pass


def test_debug():
    pass
