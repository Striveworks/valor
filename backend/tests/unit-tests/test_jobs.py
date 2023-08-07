from unittest.mock import MagicMock

import pytest

from velour_api.backend import jobs
from velour_api.enums import JobStatus, Stateflow
from velour_api.exceptions import (
    EvaluationJobDoesNotExistError,
    StateflowError,
)
from velour_api.schemas.jobs import BackendStatus, DatasetStatus, ModelStatus


@pytest.fixture(autouse=True)
def teardown():
    # reset the global redis instance (which may have been set by other tests)
    jobs.r = None


def test_get_evaluation_job():
    """Checks that multiple calls of `_get_backend_status` only
    connects to redis once
    """

    jobs.redis.Redis = MagicMock()

    assert jobs.redis.Redis.call_count == 0
    with pytest.raises(EvaluationJobDoesNotExistError):
        jobs.get_evaluation_job(0)
    assert jobs.redis.Redis.call_count == 1
    with pytest.raises(EvaluationJobDoesNotExistError):
        jobs.get_evaluation_job(0)
    assert jobs.redis.Redis.call_count == 1


def test_set_evaluation_job():
    """Checks that multiple calls of `_get_backend_status` only
    connects to redis once
    """
    jobs.redis.Redis = MagicMock()

    assert jobs.redis.Redis.call_count == 0
    jobs.set_evaluation_job(0, JobStatus.PENDING)
    assert jobs.redis.Redis.call_count == 1
    jobs.set_evaluation_job(1, JobStatus.PENDING)
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


def test__update_backend_status():

    jobs._get_backend_status = MagicMock()
    jobs._set_backend_status = MagicMock()

    def _generate_status(
        ds_state: Stateflow | None, md_state: Stateflow | None = None
    ):
        if ds_state is None:
            return BackendStatus()

        if md_state is None:
            models = {}
        else:
            models = {"md": ModelStatus(status=md_state)}

        return BackendStatus(
            datasets={
                "ds": DatasetStatus(
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
                jobs._update_backend_status(ds_next, dataset_name="ds")
        else:
            with pytest.raises(error):
                jobs._update_backend_status(
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
        if ds_next != Stateflow.NONE:
            jobs._update_backend_status(ds_next, dataset_name="ds")
        else:
            jobs._update_backend_status(
                md_next, dataset_name="ds", model_name="md"
            )

    # dataset - positive cases
    _test_permutation_pos(Stateflow.NONE, Stateflow.CREATE)
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
    _test_permutation_neg(Stateflow.NONE, Stateflow.READY)
    _test_permutation_neg(Stateflow.NONE, Stateflow.EVALUATE)
    _test_permutation_neg(Stateflow.NONE, Stateflow.DELETE)
    _test_permutation_neg(Stateflow.CREATE, Stateflow.EVALUATE)
    _test_permutation_neg(Stateflow.READY, Stateflow.CREATE)
    _test_permutation_neg(Stateflow.EVALUATE, Stateflow.CREATE)
    _test_permutation_neg(Stateflow.EVALUATE, Stateflow.DELETE)
    _test_permutation_neg(Stateflow.DELETE, Stateflow.CREATE)
    _test_permutation_neg(Stateflow.DELETE, Stateflow.READY)
    _test_permutation_neg(Stateflow.DELETE, Stateflow.EVALUATE)

    # model - positive cases
    for state in [Stateflow.READY, Stateflow.EVALUATE]:
        _test_permutation_pos(
            state, Stateflow.NONE, Stateflow.NONE, Stateflow.CREATE
        )
        _test_permutation_pos(
            state, Stateflow.NONE, Stateflow.CREATE, Stateflow.CREATE
        )
        _test_permutation_pos(
            state, Stateflow.NONE, Stateflow.CREATE, Stateflow.READY
        )
        _test_permutation_pos(
            state, Stateflow.NONE, Stateflow.CREATE, Stateflow.DELETE
        )
        _test_permutation_pos(
            state, Stateflow.NONE, Stateflow.READY, Stateflow.READY
        )
        _test_permutation_pos(
            state, Stateflow.NONE, Stateflow.READY, Stateflow.EVALUATE
        )
        _test_permutation_pos(
            state, Stateflow.NONE, Stateflow.READY, Stateflow.DELETE
        )
        _test_permutation_pos(
            state, Stateflow.NONE, Stateflow.EVALUATE, Stateflow.READY
        )
        _test_permutation_pos(
            state, Stateflow.NONE, Stateflow.EVALUATE, Stateflow.EVALUATE
        )
        _test_permutation_pos(
            state, Stateflow.NONE, Stateflow.DELETE, Stateflow.DELETE
        )

    # model - negative cases
    for state in [Stateflow.READY, Stateflow.EVALUATE]:
        _test_permutation_neg(
            state, Stateflow.NONE, Stateflow.NONE, Stateflow.READY
        )
        _test_permutation_neg(
            state, Stateflow.NONE, Stateflow.NONE, Stateflow.EVALUATE
        )
        _test_permutation_neg(
            state, Stateflow.NONE, Stateflow.NONE, Stateflow.DELETE
        )
        _test_permutation_neg(
            state, Stateflow.NONE, Stateflow.CREATE, Stateflow.EVALUATE
        )
        _test_permutation_neg(
            state, Stateflow.NONE, Stateflow.READY, Stateflow.CREATE
        )
        _test_permutation_neg(
            state, Stateflow.NONE, Stateflow.EVALUATE, Stateflow.CREATE
        )
        _test_permutation_neg(
            state, Stateflow.NONE, Stateflow.EVALUATE, Stateflow.DELETE
        )
        _test_permutation_neg(
            state, Stateflow.NONE, Stateflow.DELETE, Stateflow.CREATE
        )
        _test_permutation_neg(
            state, Stateflow.NONE, Stateflow.DELETE, Stateflow.READY
        )
        _test_permutation_neg(
            state, Stateflow.NONE, Stateflow.DELETE, Stateflow.EVALUATE
        )
    for state in [Stateflow.CREATE, Stateflow.DELETE]:
        _test_permutation_neg(
            state, Stateflow.NONE, Stateflow.NONE, Stateflow.CREATE
        )
        _test_permutation_neg(
            state, Stateflow.NONE, Stateflow.CREATE, Stateflow.CREATE
        )
        _test_permutation_neg(
            state, Stateflow.NONE, Stateflow.CREATE, Stateflow.READY
        )
        _test_permutation_neg(
            state, Stateflow.NONE, Stateflow.CREATE, Stateflow.DELETE
        )
        _test_permutation_neg(
            state, Stateflow.NONE, Stateflow.READY, Stateflow.READY
        )
        _test_permutation_neg(
            state, Stateflow.NONE, Stateflow.READY, Stateflow.EVALUATE
        )
        _test_permutation_neg(
            state, Stateflow.NONE, Stateflow.READY, Stateflow.DELETE
        )
        _test_permutation_neg(
            state, Stateflow.NONE, Stateflow.EVALUATE, Stateflow.READY
        )
        _test_permutation_neg(
            state, Stateflow.NONE, Stateflow.EVALUATE, Stateflow.EVALUATE
        )
        _test_permutation_neg(
            state, Stateflow.NONE, Stateflow.DELETE, Stateflow.DELETE
        )

    # model & dataset - postive cases
    _test_permutation_pos(Stateflow.READY, Stateflow.DELETE, Stateflow.CREATE)
    _test_permutation_pos(Stateflow.READY, Stateflow.DELETE, Stateflow.READY)
    _test_permutation_pos(Stateflow.READY, Stateflow.DELETE, Stateflow.DELETE)

    # model & dataset - negative cases
    _test_permutation_neg(
        Stateflow.READY, Stateflow.DELETE, Stateflow.EVALUATE
    )
