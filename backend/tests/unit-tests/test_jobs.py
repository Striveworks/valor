from unittest.mock import MagicMock

import pytest

from velour_api.backend import jobs
from velour_api.enums import JobStatus, Stateflow
from velour_api.exceptions import (
    DatasetDoesNotExistError,
    ModelDoesNotExistError,
    StateflowError,
)
from velour_api.schemas.jobs import BackendState, BackendStatus


@pytest.fixture(autouse=True)
def teardown():
    # reset the global redis instance (which may have been set by other tests)
    jobs.r = None


def test_get_evaluation_status():
    """Checks that multiple calls of `_get_backend_status` only
    connects to redis once
    """
    jobs.redis.Redis = MagicMock()

    assert jobs.redis.Redis.call_count == 0
    with pytest.raises(RuntimeError):
        jobs.get_evaluation_status(0)
    assert jobs.redis.Redis.call_count == 1
    with pytest.raises(RuntimeError):
        jobs.get_evaluation_status(0)
    assert jobs.redis.Redis.call_count == 1


def test_set_evaluation_status():
    """Checks that multiple calls of `_get_backend_status` only
    connects to redis once
    """
    jobs.redis.Redis = MagicMock()

    assert jobs.redis.Redis.call_count == 0
    jobs.set_evaluation_status(0, JobStatus.PENDING)
    assert jobs.redis.Redis.call_count == 1
    jobs.set_evaluation_status(0, JobStatus.PROCESSING)
    assert jobs.redis.Redis.call_count == 1


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


def test__parse_args():

    dataset_name = "ds1"
    model_name = "md1"

    assert (dataset_name, model_name) == jobs._parse_args(
        dataset_name, model_name
    )
    assert (dataset_name, None) == jobs._parse_args(dataset_name)
    assert (dataset_name, None) == jobs._parse_args(dataset_name=dataset_name)
    assert (dataset_name, model_name) == jobs._parse_args(
        dataset_name=dataset_name, model_name=model_name
    )
    assert (dataset_name, model_name) == jobs._parse_args(
        dataset_name, model_name=model_name
    )

    with pytest.raises(ValueError) as e:
        jobs._parse_args()
    assert "dataset_name not provided" in str(e)
    with pytest.raises(ValueError) as e:
        jobs._parse_args(123)
    assert "must be of type `str`" in str(e)
    with pytest.raises(ValueError) as e:
        jobs._parse_args("123", 123)
    assert "must be of type `str`" in str(e)


def test__validate_backend_status():

    jobs._get_backend_status = MagicMock()

    def _generate_status(
        ds_state: Stateflow | None, md_state: Stateflow | None = None
    ):
        if ds_state is None:
            return BackendStatus()

        if md_state is None:
            models = {}
        else:
            models = {"md": md_state}

        return BackendStatus(
            datasets={
                "ds": BackendState(
                    status=ds_state,
                    models=models,
                )
            }
        )

    def _test_permutation_neg(
        ds_state: Stateflow,
        ds_next: Stateflow | None,
        md_state: Stateflow | None = None,
        md_next: Stateflow | None = None,
        error=StateflowError,
    ):
        jobs._get_backend_status.return_value = _generate_status(
            ds_state=ds_state,
            md_state=md_state,
        )
        if ds_next is not None:
            with pytest.raises(error):
                jobs._validate_backend_status(ds_next, dataset_name="ds")
        else:
            with pytest.raises(error):
                jobs._validate_backend_status(
                    md_next, dataset_name="ds", model_name="md"
                )

    def _test_permutation_pos(
        ds_state: Stateflow,
        ds_next: Stateflow,
        md_state: Stateflow | None = None,
        md_next: Stateflow | None = None,
    ):
        jobs._get_backend_status.return_value = _generate_status(
            ds_state=ds_state,
            md_state=md_state,
        )
        if ds_next is not None:
            assert jobs._validate_backend_status(ds_next, dataset_name="ds")
        else:
            assert jobs._validate_backend_status(
                md_next, dataset_name="ds", model_name="md"
            )

    # dataset - positive cases
    _test_permutation_pos(None, Stateflow.CREATE)
    _test_permutation_pos(Stateflow.CREATE, Stateflow.CREATE)
    _test_permutation_pos(Stateflow.CREATE, Stateflow.READY)
    _test_permutation_pos(Stateflow.CREATE, Stateflow.DELETE)
    _test_permutation_pos(Stateflow.READY, Stateflow.READY)
    _test_permutation_pos(Stateflow.READY, Stateflow.EVALUATE)
    _test_permutation_pos(Stateflow.READY, Stateflow.DELETE)
    _test_permutation_pos(Stateflow.EVALUATE, Stateflow.READY)
    _test_permutation_pos(Stateflow.EVALUATE, Stateflow.EVALUATE)
    _test_permutation_pos(Stateflow.DELETE, Stateflow.DELETE)

    # dataset - negative cases
    _test_permutation_neg(
        None, Stateflow.READY, error=DatasetDoesNotExistError
    )
    _test_permutation_neg(
        None, Stateflow.EVALUATE, error=DatasetDoesNotExistError
    )
    _test_permutation_neg(
        None, Stateflow.DELETE, error=DatasetDoesNotExistError
    )
    _test_permutation_neg(Stateflow.CREATE, Stateflow.EVALUATE)
    _test_permutation_neg(Stateflow.READY, Stateflow.CREATE)
    _test_permutation_neg(Stateflow.EVALUATE, Stateflow.CREATE)
    _test_permutation_neg(Stateflow.EVALUATE, Stateflow.DELETE)
    _test_permutation_neg(Stateflow.DELETE, Stateflow.CREATE)
    _test_permutation_neg(Stateflow.DELETE, Stateflow.READY)
    _test_permutation_neg(Stateflow.DELETE, Stateflow.EVALUATE)

    # model - positive cases
    for state in [Stateflow.READY, Stateflow.EVALUATE]:
        _test_permutation_pos(state, None, None, Stateflow.CREATE)
        _test_permutation_pos(state, None, Stateflow.CREATE, Stateflow.CREATE)
        _test_permutation_pos(state, None, Stateflow.CREATE, Stateflow.READY)
        _test_permutation_pos(state, None, Stateflow.CREATE, Stateflow.DELETE)
        _test_permutation_pos(state, None, Stateflow.READY, Stateflow.READY)
        _test_permutation_pos(state, None, Stateflow.READY, Stateflow.EVALUATE)
        _test_permutation_pos(state, None, Stateflow.READY, Stateflow.DELETE)
        _test_permutation_pos(state, None, Stateflow.EVALUATE, Stateflow.READY)
        _test_permutation_pos(
            state, None, Stateflow.EVALUATE, Stateflow.EVALUATE
        )
        _test_permutation_pos(state, None, Stateflow.DELETE, Stateflow.DELETE)

    # model - negative cases
    for state in [Stateflow.READY, Stateflow.EVALUATE]:
        _test_permutation_neg(
            state, None, None, Stateflow.READY, error=ModelDoesNotExistError
        )
        _test_permutation_neg(
            state, None, None, Stateflow.EVALUATE, error=ModelDoesNotExistError
        )
        _test_permutation_neg(
            state, None, None, Stateflow.DELETE, error=ModelDoesNotExistError
        )
        _test_permutation_neg(
            state, None, Stateflow.CREATE, Stateflow.EVALUATE
        )
        _test_permutation_neg(state, None, Stateflow.READY, Stateflow.CREATE)
        _test_permutation_neg(
            state, None, Stateflow.EVALUATE, Stateflow.CREATE
        )
        _test_permutation_neg(
            state, None, Stateflow.EVALUATE, Stateflow.DELETE
        )
        _test_permutation_neg(state, None, Stateflow.DELETE, Stateflow.CREATE)
        _test_permutation_neg(state, None, Stateflow.DELETE, Stateflow.READY)
        _test_permutation_neg(
            state, None, Stateflow.DELETE, Stateflow.EVALUATE
        )
    for state in [Stateflow.CREATE, Stateflow.DELETE]:
        _test_permutation_neg(state, None, None, Stateflow.CREATE)
        _test_permutation_neg(state, None, Stateflow.CREATE, Stateflow.CREATE)
        _test_permutation_neg(state, None, Stateflow.CREATE, Stateflow.READY)
        _test_permutation_neg(state, None, Stateflow.CREATE, Stateflow.DELETE)
        _test_permutation_neg(state, None, Stateflow.READY, Stateflow.READY)
        _test_permutation_neg(state, None, Stateflow.READY, Stateflow.EVALUATE)
        _test_permutation_neg(state, None, Stateflow.READY, Stateflow.DELETE)
        _test_permutation_neg(state, None, Stateflow.EVALUATE, Stateflow.READY)
        _test_permutation_neg(
            state, None, Stateflow.EVALUATE, Stateflow.EVALUATE
        )
        _test_permutation_neg(state, None, Stateflow.DELETE, Stateflow.DELETE)

    # model & dataset - postive cases
    _test_permutation_pos(Stateflow.READY, Stateflow.DELETE, Stateflow.CREATE)
    _test_permutation_pos(Stateflow.READY, Stateflow.DELETE, Stateflow.READY)
    _test_permutation_pos(Stateflow.READY, Stateflow.DELETE, Stateflow.DELETE)

    # model & dataset - negative cases
    _test_permutation_pos(
        Stateflow.READY, Stateflow.DELETE, Stateflow.EVALUATE
    )
